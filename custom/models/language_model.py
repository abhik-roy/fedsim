"""Small Transformer language model plugin for next-token prediction.

Uses nn.TransformerEncoder with causal mask (GPT-style decoder-only).
Lightweight: ~2M params with default settings.
"""

import math
import torch
import torch.nn as nn

NAME = "Small LM"
TASK_TYPE = "language_modeling"
COMPATIBLE_TASKS = ["language_modeling"]
DESCRIPTION = "Small Transformer language model (GPT-style, ~2M params)"

PARAMS = {
    "embed_dim": {"type": "int", "default": 128, "min": 32, "max": 512, "step": 32,
                  "label": "Embedding Dimension"},
    "num_heads": {"type": "int", "default": 4, "min": 1, "max": 8, "step": 1,
                  "label": "Attention Heads"},
    "num_layers": {"type": "int", "default": 2, "min": 1, "max": 6, "step": 1,
                   "label": "Transformer Layers"},
    "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                "label": "Dropout Rate"},
}


def _get_vocab_size():
    """Read vocab size from cache, fallback to default."""
    import os
    cache_file = os.path.join(os.path.expanduser("~"), ".cache", "fedsim", "wikitext2", "vocab_size.txt")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return int(f.read().strip())
    return 10002


class SmallLM(nn.Module):
    """Transformer-based language model with causal masking."""

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2,
                 dropout=0.1, max_seq_len=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        # Restore padding_idx=0 embedding to zeros after init
        with torch.no_grad():
            self.token_embed.weight[0].zero_()

    def _generate_causal_mask(self, seq_len, device):
        """Generate upper-triangular causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        """Forward pass. x: (batch, seq_len) LongTensor."""
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        x_emb = self.dropout(tok_emb + pos_emb)

        causal_mask = self._generate_causal_mask(seq_len, x.device)
        output = self.transformer(x_emb, mask=causal_mask)
        logits = self.output_proj(output)

        return logits  # (batch, seq_len, vocab_size)


def build(dataset_info, **kwargs):
    """Build a small Transformer LM."""
    vocab_size = dataset_info.get("vocab_size") or _get_vocab_size()
    seq_length = dataset_info.get("seq_length", 128)
    embed_dim = kwargs.get("embed_dim", PARAMS["embed_dim"]["default"])
    num_heads = kwargs.get("num_heads", PARAMS["num_heads"]["default"])
    num_layers = kwargs.get("num_layers", PARAMS["num_layers"]["default"])
    dropout = kwargs.get("dropout", PARAMS["dropout"]["default"])

    return SmallLM(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=num_heads, num_layers=num_layers,
        dropout=dropout, max_seq_len=seq_length,
    )


def train_step(model, batch, optimizer, device, **kwargs):
    """Train step for next-token prediction."""
    input_ids, target_ids = batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    optimizer.zero_grad()
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # Reshape for cross entropy: (batch*seq_len, vocab_size) vs (batch*seq_len,)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        ignore_index=0,  # ignore padding
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    ppl = min(math.exp(loss.item()), 1e4)  # clamp perplexity
    return {"loss": loss.item(), "perplexity": ppl}


def eval_step(model, batch, device, **kwargs):
    """Evaluation step for next-token prediction."""
    input_ids, target_ids = batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        ignore_index=0,
    )

    ppl = min(math.exp(loss.item()), 1e4)
    return {"loss": loss.item(), "perplexity": ppl}
