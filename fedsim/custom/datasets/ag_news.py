"""AG News dataset plugin for text classification.

4 classes: World (0), Sports (1), Business (2), Sci/Tech (3).
Tokenizes text via whitespace splitting, builds a vocabulary from training data,
and pads/truncates sequences to a fixed length.
"""

import csv
import io
import os
import urllib.request
from collections import Counter

import torch
from torch.utils.data import Dataset

NAME = "AG News (4-class)"
NUM_CLASSES = 4
INPUT_CHANNELS = 1
IMAGE_SIZE = 256  # sequence length
TASK_TYPE = "text_classification"
SEQ_LENGTH = 256

VOCAB_SIZE = 25002  # updated after vocabulary is built (25000 + PAD + UNK)
_VOCAB_CACHE_FILE = None  # set after load() to allow TextCNN to read

_PAD_IDX = 0
_UNK_IDX = 1
_MAX_VOCAB = 25000

_TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


def _download_csv(url, cache_path):
    """Download a CSV file and cache it locally."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return list(csv.reader(f))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Downloading {url} ...")
    response = urllib.request.urlopen(url, timeout=120)
    data = response.read().decode("utf-8")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(data)
    return list(csv.reader(io.StringIO(data)))


def _parse_rows(rows):
    """Parse CSV rows into (label, text) pairs. Labels converted to 0-indexed."""
    samples = []
    for row in rows:
        if len(row) < 3:
            continue
        label = int(row[0]) - 1  # torchtext/CSV uses 1-4, convert to 0-3
        text = row[1] + " " + row[2]  # title + description
        samples.append((label, text))
    return samples


def _tokenize(text):
    """Simple whitespace tokenizer with lowercasing and basic cleanup."""
    return text.lower().split()


def _build_vocab(samples, max_vocab=_MAX_VOCAB):
    """Build a vocabulary from training samples. Returns word-to-index dict."""
    counter = Counter()
    for _, text in samples:
        counter.update(_tokenize(text))
    most_common = counter.most_common(max_vocab)
    # 0 = PAD, 1 = UNK, then words starting at index 2
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    return word2idx


def _encode(text, word2idx, seq_len):
    """Tokenize and encode text to a fixed-length list of indices."""
    tokens = _tokenize(text)
    indices = [word2idx.get(t, _UNK_IDX) for t in tokens]
    # Truncate or pad
    if len(indices) >= seq_len:
        indices = indices[:seq_len]
    else:
        indices = indices + [_PAD_IDX] * (seq_len - len(indices))
    return indices


class AGNewsDataset(Dataset):
    """AG News dataset returning (token_ids_tensor, label)."""

    def __init__(self, samples, word2idx, seq_len):
        self.samples = samples
        self.word2idx = word2idx
        self.seq_len = seq_len
        # Pre-extract targets for fast non-IID partitioning (avoids O(N) fallback)
        self.targets = [label for label, _ in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        ids = _encode(text, self.word2idx, self.seq_len)
        return torch.tensor(ids, dtype=torch.long), label


def load():
    """Return (train_dataset, test_dataset)."""
    global VOCAB_SIZE

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fedsim", "ag_news")
    train_path = os.path.join(cache_dir, "train.csv")
    test_path = os.path.join(cache_dir, "test.csv")

    train_rows = _download_csv(_TRAIN_URL, train_path)
    test_rows = _download_csv(_TEST_URL, test_path)

    train_samples = _parse_rows(train_rows)
    test_samples = _parse_rows(test_rows)

    word2idx = _build_vocab(train_samples)
    VOCAB_SIZE = len(word2idx) + 2  # +2 for PAD and UNK

    # Write vocab size to a cache file so TextCNN can read it reliably
    # (avoids fragile cross-module global mutation via exec_module)
    vocab_cache = os.path.join(cache_dir, "vocab_size.txt")
    with open(vocab_cache, "w") as f:
        f.write(str(VOCAB_SIZE))

    seq_len = IMAGE_SIZE

    train_dataset = AGNewsDataset(train_samples, word2idx, seq_len)
    test_dataset = AGNewsDataset(test_samples, word2idx, seq_len)

    return train_dataset, test_dataset
