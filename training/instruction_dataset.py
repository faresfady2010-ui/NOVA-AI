"""JSONL supervised instruction examples and assistant-only next-token targets."""
from __future__ import annotations
import json
from pathlib import Path
import random
import torch
from model.tokenizer import ByteTokenizer

USER_MARKER = "<|user|>\n"
ASSISTANT_MARKER = "<|assistant|>\n"
END_MARKER = "\n<|end|>\n"


def load_examples(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")
    examples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            messages = item.get("messages") if isinstance(item, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number} requires a non-empty messages array")
            for i, message in enumerate(messages):
                if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                    raise ValueError(f"{path}:{line_number} has invalid message at index {i}")
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError(f"{path}:{line_number} has empty/non-text message at index {i}")
                expected = "user" if i % 2 == 0 else "assistant"
                if message["role"] != expected:
                    raise ValueError(f"{path}:{line_number} messages must alternate user/assistant beginning with user")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{line_number} must end with an assistant response")
            examples.append(item)
    if not examples:
        raise ValueError(f"Instruction dataset is empty: {path}")
    return examples


def encode_example(example: dict, tokenizer: ByteTokenizer, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-length inputs/labels; user/prompt targets are ignored (-100)."""
    sequence = [tokenizer.BOS]
    assistant_target = [False]
    for message in example["messages"]:
        is_assistant = message["role"] == "assistant"
        prefix = ASSISTANT_MARKER if is_assistant else USER_MARKER
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(message["content"].strip())
        ending_ids = tokenizer.encode(END_MARKER if is_assistant else "\n")
        sequence.extend(prefix_ids + content_ids + ending_ids)
        # Assistant content and its end marker are supervised. Role markers and
        # all user text are context only. Shift alignment happens below.
        assistant_target.extend([False] * len(prefix_ids))
        assistant_target.extend([is_assistant] * len(content_ids + ending_ids))

    # Keep the newest context if a long example exceeds the checkpoint window.
    sequence = sequence[-(context_length + 1):]
    assistant_target = assistant_target[-(context_length + 1):]
    input_ids = sequence[:-1]
    labels = [token if assistant_target[i + 1] else -100 for i, token in enumerate(sequence[1:])]
    if not any(label != -100 for label in labels):
        raise ValueError("Example has no assistant target within model context; shorten it or increase context")
    pad_count = context_length - len(input_ids)
    input_ids += [ByteTokenizer.PAD] * pad_count
    labels += [-100] * pad_count
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class InstructionBatcher:
    def __init__(self, examples: list[dict], tokenizer: ByteTokenizer, context_length: int, device: torch.device):
        if not examples:
            raise ValueError("instruction examples cannot be empty")
        self.encoded = [encode_example(ex, tokenizer, context_length) for ex in examples]
        self.device = device

    def batch(self, batch_size: int, *, rng: random.Random | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        chooser = rng or random
        selected = [self.encoded[chooser.randrange(len(self.encoded))] for _ in range(batch_size)]
        x = torch.stack([pair[0] for pair in selected]).to(self.device)
        y = torch.stack([pair[1] for pair in selected]).to(self.device)
        return x, y

    def all_batches(self, batch_size: int):
        """Yield every example once, deterministic order, for held-out loss."""
        for start in range(0, len(self.encoded), batch_size):
            chunk = self.encoded[start:start + batch_size]
            yield (torch.stack([x for x, _ in chunk]).to(self.device),
                   torch.stack([y for _, y in chunk]).to(self.device))


def evaluate_instruction_loss(model, batcher: InstructionBatcher) -> float:
    model.eval()
    total_loss, total_targets = 0.0, 0
    with torch.no_grad():
        for x, y in batcher.all_batches(8):
            _, loss = model(x, y)
            count = int((y != -100).sum().item())
            total_loss += float(loss.item()) * count
            total_targets += count
    if total_targets == 0:
        raise ValueError("instruction dataset has no supervised assistant targets")
    return total_loss / total_targets
