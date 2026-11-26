"""Model and default training configuration for NOVA."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 259
    context_length: int = 128
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


@dataclass
class TrainingConfig:
    """Defaults used by ``python -m training.train``; CLI flags override these."""
    steps: int = 500
    batch_size: int = 8
    learning_rate: float = 3e-4
    eval_every: int = 100
    gradient_clip: float = 1.0
    seed: int = 42
    train_path: str = "data/training.txt"
    validation_path: str = "data/validation.txt"
    checkpoint_path: str = "checkpoints/nova.pt"
    log_path: str = "checkpoints/training.jsonl"
    report_path: str = "checkpoints/training_report.json"

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.eval_every < 1:
            raise ValueError("steps, batch_size, and eval_every must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
