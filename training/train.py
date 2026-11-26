"""Train NOVA from local, user-provided text files (no pretrained weights)."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from .dataset import TokenBatcher, load_datasets


def save_checkpoint(path: str | Path, model: NOVATransformer, config: ModelConfig, step: int, optimizer=None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": config.to_dict(), "step": step,
               "tokenizer": {"type": "utf8-byte", "vocab_size": ByteTokenizer.VOCAB_SIZE}}
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/training.txt"); parser.add_argument("--validation", default="data/validation.txt")
    parser.add_argument("--checkpoint", default="checkpoints/nova.pt"); parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4); parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128); parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1: parser.error("steps, batch-size, and eval-every must be positive")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(context_length=args.context, d_model=args.width, n_heads=args.heads, n_layers=args.layers)
    train_tokens, val_tokens = load_datasets(args.train, args.validation, ByteTokenizer())
    train_batcher, val_batcher = TokenBatcher(train_tokens, args.context, device), TokenBatcher(val_tokens, args.context, device)
    model = NOVATransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Cosine decay with a small floor avoids a zero learning rate at the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    log_path = Path(args.checkpoint).with_suffix(".jsonl"); log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train(); x, y = train_batcher.batch(args.batch_size)
        _, loss = model(x, y); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = val_batcher.batch(args.batch_size); _, val_loss = model(vx, vy)
            elapsed = time.time() - started
            record = {"step": step, "train_loss": float(loss), "validation_loss": float(val_loss), "lr": scheduler.get_last_lr()[0], "elapsed_seconds": elapsed}
            print(json.dumps(record), flush=True)
            with log_path.open("a", encoding="utf-8") as log: log.write(json.dumps(record) + "\n")
            save_checkpoint(args.checkpoint, model, config, step, optimizer)
    print(f"Training complete on {device}; checkpoint: {args.checkpoint}")

if __name__ == "__main__": main()
