"""WikiText-2 language modeling dataset plugin.

Next-token prediction task. Tokenizes via whitespace, builds vocabulary,
creates fixed-length sequences for autoregressive language modeling.
"""

import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

NAME = "WikiText-2"
TASK_TYPE = "language_modeling"
NUM_CLASSES = None
INPUT_CHANNELS = 1
IMAGE_SIZE = 128
SEQ_LENGTH = 128
VOCAB_SIZE = 10002  # placeholder, updated after load()
DESCRIPTION = "WikiText-2 language modeling dataset (next-token prediction)"

_PAD_IDX = 0
_UNK_IDX = 1
_MAX_VOCAB = 10000
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fedsim", "wikitext2")

# URLs for raw text
_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
_TEST_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"


class _LMDataset(Dataset):
    """Language modeling dataset with .targets for non-IID partitioning."""

    def __init__(self, input_ids, target_ids, target_bins=None):
        self.input_ids = input_ids
        self.target_ids = target_ids
        if target_bins is not None:
            self.targets = target_bins
        else:
            self.targets = np.zeros(len(input_ids), dtype=np.int64)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def _download_text(url, cache_path):
    """Download text file, cache locally."""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        response = urllib.request.urlopen(url, timeout=120)
        text = response.read().decode("utf-8")
        with open(cache_path, "w") as f:
            f.write(text)
        return text
    except Exception:
        # Fallback: generate synthetic text for testing
        import warnings
        warnings.warn(f"Failed to download {url}. Using synthetic text.")
        words = [f"word{i}" for i in range(500)]
        import random
        random.seed(42)
        text = " ".join(random.choices(words, k=50000))
        with open(cache_path, "w") as f:
            f.write(text)
        return text


def _build_vocab(text, max_vocab=_MAX_VOCAB):
    """Build vocabulary from text. Returns word2idx dict."""
    tokens = text.lower().split()
    counter = Counter(tokens)
    most_common = counter.most_common(max_vocab)
    word2idx = {"<pad>": _PAD_IDX, "<unk>": _UNK_IDX}
    for word, _ in most_common:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
    return word2idx


def _tokenize(text, word2idx, seq_length):
    """Convert text to sequences of token IDs."""
    tokens = text.lower().split()
    token_ids = [word2idx.get(t, _UNK_IDX) for t in tokens]

    # Create fixed-length sequences
    sequences_in = []
    sequences_out = []
    for i in range(0, len(token_ids) - seq_length, seq_length):
        seq = token_ids[i:i + seq_length + 1]
        if len(seq) == seq_length + 1:
            sequences_in.append(seq[:-1])   # input: tokens 0..n-1
            sequences_out.append(seq[1:])   # target: tokens 1..n

    return sequences_in, sequences_out


def load(**kwargs):
    """Load WikiText-2 dataset for language modeling."""
    global VOCAB_SIZE

    train_text = _download_text(_TRAIN_URL, os.path.join(_CACHE_DIR, "train.txt"))
    test_text = _download_text(_TEST_URL, os.path.join(_CACHE_DIR, "test.txt"))

    # Build vocab from training data
    word2idx = _build_vocab(train_text)
    VOCAB_SIZE = len(word2idx)

    # Cache vocab size for model plugins
    vs_path = os.path.join(_CACHE_DIR, "vocab_size.txt")
    os.makedirs(os.path.dirname(vs_path), exist_ok=True)
    with open(vs_path, "w") as f:
        f.write(str(VOCAB_SIZE))

    seq_length = SEQ_LENGTH

    # Tokenize
    train_in, train_out = _tokenize(train_text, word2idx, seq_length)
    test_in, test_out = _tokenize(test_text, word2idx, seq_length)

    # Convert to tensors
    train_input = torch.tensor(train_in, dtype=torch.long)
    train_target = torch.tensor(train_out, dtype=torch.long)
    test_input = torch.tensor(test_in, dtype=torch.long)
    test_target = torch.tensor(test_out, dtype=torch.long)

    # Bin for non-IID: mean token index per sequence, 5 quantile buckets
    mean_indices = train_input.float().mean(dim=1).numpy()
    quantiles = np.quantile(mean_indices, [0.2, 0.4, 0.6, 0.8])
    train_bins = np.digitize(mean_indices, quantiles).astype(np.int64)

    test_mean = test_input.float().mean(dim=1).numpy()
    test_bins = np.digitize(test_mean, quantiles).astype(np.int64)

    train_ds = _LMDataset(train_input, train_target, train_bins)
    test_ds = _LMDataset(test_input, test_target, test_bins)

    return train_ds, test_ds
