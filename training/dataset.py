"""Dataset reading, leakage checks, and random next-token batches."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from model.tokenizer import ByteTokenizer


def read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    if not text:
        raise ValueError(f"Dataset file is empty: {path}")
    return text


def load_datasets(training_path: str | Path, validation_path: str | Path, tokenizer: ByteTokenizer):
    train_text, val_text = read_text(training_path), read_text(validation_path)
    # Exact shared non-empty lines are a common accidental leakage source.
    train_lines = {line.strip() for line in train_text.splitlines() if line.strip()}
    val_lines = {line.strip() for line in val_text.splitlines() if line.strip()}
    overlap = train_lines & val_lines
    if overlap:
        raise ValueError(f"Training/validation leakage: {len(overlap)} identical non-empty line(s)")
    train_ids = torch.tensor(tokenizer.encode(train_text, add_bos=True, add_eos=True), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text, add_bos=True, add_eos=True), dtype=torch.long)
    return train_ids, val_ids


class TokenBatcher:
    def __init__(self, tokens: torch.Tensor, context_length: int, device: torch.device):
        if tokens.numel() < context_length + 1:
            raise ValueError("dataset needs at least context_length + 1 tokens")
        self.tokens, self.context_length, self.device = tokens, context_length, device

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.tokens.numel() - self.context_length, (batch_size,))
        x = torch.stack([self.tokens[i:i + self.context_length] for i in starts])
        y = torch.stack([self.tokens[i + 1:i + self.context_length + 1] for i in starts])
        return x.to(self.device), y.to(self.device)
