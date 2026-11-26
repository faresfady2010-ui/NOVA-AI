"""Configuration for NOVA's decoder-only Transformer."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 259
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.context_length < 2 or self.d_model < 1 or self.n_layers < 1:
            raise ValueError("context_length, d_model, and n_layers must be positive")
        if self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by a positive n_heads")
        if self.vocab_size < 259:
            raise ValueError("vocab_size must include all byte and special tokens")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        return cls(**values)
