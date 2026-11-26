"""Train NOVA from local text, starting from random initialization."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import ByteTokenizer, ModelConfig, NOVATransformer
from model.config import TrainingConfig
from .dataset import TokenBatcher, load_datasets


def evaluate_loss(model: NOVATransformer, tokens: torch.Tensor, device: torch.device) -> float:
    """Deterministic, token-weighted next-byte loss over a held-out stream."""
    context = model.config.context_length
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        # Adjacent windows cover the stream without stochastic resampling.
        for start in range(0, tokens.numel() - 1, context):
            end = min(start + context + 1, tokens.numel())
            chunk = tokens[start:end]
            if chunk.numel() < 2:
                continue
            x = chunk[:-1].unsqueeze(0).to(device)
            y = chunk[1:].unsqueeze(0).to(device)
            _, loss = model(x, y)
            count = y.numel()
            total_loss += loss.item() * count
            total_tokens += count
    if total_tokens == 0:
        raise ValueError("dataset has no next-token targets")
    return total_loss / total_tokens


def save_checkpoint(path: str | Path, model: NOVATransformer, config: ModelConfig, step: int,
                    optimizer=None, scheduler=None, report: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "config": config.to_dict(), "step": step,
        "tokenizer": {"type": "utf8-byte", "vocab_size": ByteTokenizer.VOCAB_SIZE},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if report is not None:
        payload["training_report"] = report
    torch.save(payload, path)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=defaults.train_path)
    parser.add_argument("--validation", default=defaults.validation_path)
    parser.add_argument("--checkpoint", default=defaults.checkpoint_path)
    parser.add_argument("--report", default=defaults.report_path)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--context", type=int, default=ModelConfig.context_length)
    parser.add_argument("--layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--width", type=int, default=ModelConfig.d_model)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    try:
        settings = TrainingConfig(steps=args.steps, batch_size=args.batch_size,
                                  learning_rate=args.lr, eval_every=args.eval_every, seed=args.seed)
        config = ModelConfig(context_length=args.context, d_model=args.width,
                             n_heads=args.heads, n_layers=args.layers)
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = ByteTokenizer()
        train_tokens, val_tokens = load_datasets(args.train, args.validation, tokenizer)
        train_batcher = TokenBatcher(train_tokens, config.context_length, device)
        model = NOVATransformer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings.steps, eta_min=settings.learning_rate * 0.05
        )
        train_loss_start = evaluate_loss(model, train_tokens, device)
        val_loss_start = evaluate_loss(model, val_tokens, device)
        checkpoint_path, report_path = Path(args.checkpoint), Path(args.report)
        log_path = checkpoint_path.with_suffix(".jsonl")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        started = time.perf_counter()
        records = []

        for step in range(1, settings.steps + 1):
            model.train()
            x, y = train_batcher.batch(settings.batch_size)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), defaults.gradient_clip)
            optimizer.step()
            scheduler.step()

            if step == 1 or step % settings.eval_every == 0 or step == settings.steps:
                elapsed = time.perf_counter() - started
                val_loss = evaluate_loss(model, val_tokens, device)
                record = {
                    "step": step, "train_batch_loss": float(loss.detach()),
                    "validation_loss": val_loss, "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm_before_clip": float(grad_norm), "elapsed_seconds": elapsed,
                }
                records.append(record)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)

        duration = time.perf_counter() - started
        train_loss_final = evaluate_loss(model, train_tokens, device)
        val_loss_final = evaluate_loss(model, val_tokens, device)
        report = {
            "model_config": config.to_dict(),
            "trainable_parameters": parameter_count(model),
            "training_bytes_including_boundary_tokens": int(train_tokens.numel()),
            "validation_bytes_including_boundary_tokens": int(val_tokens.numel()),
            "training_steps": settings.steps,
            "initial_training_loss": train_loss_start,
            "final_training_loss": train_loss_final,
            "initial_validation_loss": val_loss_start,
            "final_validation_loss": val_loss_final,
            "training_duration_seconds": duration,
            "hardware": str(device) + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
            "seed": settings.seed,
            "training_log": str(log_path),
        }
        save_checkpoint(checkpoint_path, model, config, settings.steps, optimizer, scheduler, report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Training report:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Checkpoint saved to {checkpoint_path}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
