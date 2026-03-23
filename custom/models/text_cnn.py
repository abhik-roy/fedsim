"""TextCNN model plugin (Kim 2014) for text classification.

Architecture:
  Embedding → 3 parallel Conv1d branches (kernel sizes 3, 4, 5) →
  ReLU → Global Max Pool → Concatenate → Dropout → Linear → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "TextCNN"
COMPATIBLE_TASKS = ["text_classification"]
PARAMS = {
    "embed_dim": {"type": "int", "default": 128, "min": 32, "max": 512, "step": 32,
                  "label": "Embedding Dimension"},
    "num_filters": {"type": "int", "default": 100, "min": 16, "max": 512, "step": 16,
                    "label": "Filters per Kernel Size"},
    "dropout": {"type": "float", "default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05,
                "label": "Dropout Rate"},
}

def _get_vocab_size():
    """Read vocab size from AG News cache file, falling back to default."""
    import os
    cache_file = os.path.join(os.path.expanduser("~"), ".cache", "fedsim", "ag_news", "vocab_size.txt")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return int(f.read().strip())
    return 25002  # fallback: 25000 words + PAD + UNK


class TextCNN(nn.Module):
    """TextCNN with multiple parallel convolutional filters."""

    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, ks) for ks in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len) LongTensor
        if x.dim() == 3:
            # Framework may add a channel dim: (batch, 1, seq_len) → squeeze
            x = x.squeeze(1)
        x = x.long()

        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)        # (batch, embed_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))         # (batch, num_filters, L')
            c = c.max(dim=2).values       # (batch, num_filters) — global max pool
            conv_outs.append(c)

        out = torch.cat(conv_outs, dim=1) # (batch, num_filters * 3)
        out = self.dropout(out)
        logits = self.fc(out)             # (batch, num_classes)
        return logits


def build(dataset_info, **kwargs):
    """Build a TextCNN model from dataset_info dict."""
    num_classes = dataset_info["num_classes"]
    vs = dataset_info.get("vocab_size") or _get_vocab_size()
    embed_dim = kwargs.get("embed_dim", PARAMS["embed_dim"]["default"])
    num_filters = kwargs.get("num_filters", PARAMS["num_filters"]["default"])
    dropout = kwargs.get("dropout", PARAMS["dropout"]["default"])
    return TextCNN(vocab_size=vs, embed_dim=embed_dim, num_classes=num_classes,
                   num_filters=num_filters, dropout=dropout)
