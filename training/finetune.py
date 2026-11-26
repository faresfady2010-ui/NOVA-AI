"""Supervised fine-tuning from a NOVA checkpoint using local JSONL examples."""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from .instruction_dataset import InstructionBatcher, evaluate_instruction_loss, load_examples
from .train import save_checkpoint, parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/nova.pt", help="Previously trained NOVA base checkpoint")
    parser.add_argument("--output", default="checkpoints/nova-instruct.pt")
    parser.add_argument("--train", default="data/instructions_train.jsonl")
    parser.add_argument("--validation", default="data/instructions_validation.jsonl")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.lr <= 0:
        parser.error("steps, batch-size, eval-every and lr must be positive")
    try:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("tokenizer", {}).get("type") != "utf8-byte":
            raise ValueError("Base checkpoint has missing or unsupported tokenizer metadata")
        config = ModelConfig.from_dict(checkpoint["config"])
        model = NOVATransformer(config).to(device)
        model.load_state_dict(checkpoint["model"])
        tokenizer = ByteTokenizer()
        train_examples = load_examples(args.train)
        val_examples = load_examples(args.validation)
        train_batcher = InstructionBatcher(train_examples, tokenizer, config.context_length, device)
        val_batcher = InstructionBatcher(val_examples, tokenizer, config.context_length, device)
        initial_validation_loss = evaluate_instruction_loss(model, val_batcher)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
        log_path = Path(args.output).with_suffix(".jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        started = time.perf_counter()
        rng = random.Random(args.seed)
        last_train_loss = float("nan")
        for step in range(1, args.steps + 1):
            model.train()
            x, y = train_batcher.batch(args.batch_size, rng=rng)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            last_train_loss = float(loss.detach())
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                val_loss = evaluate_instruction_loss(model, val_batcher)
                record = {"step": step, "train_assistant_loss": last_train_loss,
                          "validation_assistant_loss": val_loss, "learning_rate": scheduler.get_last_lr()[0],
                          "gradient_norm_before_clip": float(grad_norm),
                          "elapsed_seconds": time.perf_counter() - started}
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
        duration = time.perf_counter() - started
        final_validation_loss = evaluate_instruction_loss(model, val_batcher)
        report = {
            "stage": "supervised_instruction_finetuning",
            "base_checkpoint": args.checkpoint,
            "model_config": config.to_dict(), "trainable_parameters": parameter_count(model),
            "training_examples": len(train_examples), "validation_examples": len(val_examples),
            "steps": args.steps, "last_train_assistant_loss": last_train_loss,
            "initial_validation_assistant_loss": initial_validation_loss,
            "final_validation_assistant_loss": final_validation_loss,
            "duration_seconds": duration, "device": str(device), "seed": args.seed,
            "validation_was_used_for_updates": False,
        }
        save_checkpoint(args.output, model, config, int(checkpoint.get("step", 0)) + args.steps,
                        optimizer, scheduler, report)
        report_path = Path(args.output).with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Fine-tuned checkpoint saved to {args.output}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Fine-tuning failed: {exc}\n")


if __name__ == "__main__":
    main()
