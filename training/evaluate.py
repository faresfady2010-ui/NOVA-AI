"""Evaluate a NOVA checkpoint on train/held-out text and a generation sample."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from .dataset import load_datasets
from .train import evaluate_loss, parameter_count
from inference.generate import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/nova.pt")
    parser.add_argument("--train", default="data/training.txt")
    parser.add_argument("--validation", default="data/validation.txt")
    parser.add_argument("--prompt", default="What is artificial intelligence?")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ModelConfig.from_dict(checkpoint["config"])
        tokenizer = ByteTokenizer()
        train_tokens, val_tokens = load_datasets(args.train, args.validation, tokenizer)
        model = NOVATransformer(config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        train_loss = evaluate_loss(model, train_tokens, device)
        val_loss = evaluate_loss(model, val_tokens, device)

        # Compare fixed-input logits from the in-memory model with a fresh
        # checkpoint reload. This checks serialization, not model quality.
        reloaded = NOVATransformer(config).to(device)
        reloaded.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
        reloaded.eval()
        fixed = val_tokens[:min(config.context_length, val_tokens.numel() - 1)].unsqueeze(0).to(device)
        with torch.no_grad():
            original_logits, _ = model(fixed)
            reloaded_logits, _ = reloaded(fixed)
        consistency_max_abs_error = float((original_logits - reloaded_logits).abs().max().item())

        rng = torch.Generator(device=device)
        rng.manual_seed(args.seed)
        sample = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                          temperature=0.8, top_k=40, device=device, generator=rng)
        result = {
            "checkpoint_step": int(checkpoint.get("step", -1)),
            "train_loss": train_loss,
            "training_perplexity": math.exp(min(train_loss, 80)),
            "validation_loss": val_loss,
            "validation_perplexity": math.exp(min(val_loss, 80)),
            "trainable_parameters": parameter_count(model),
            "checkpoint_consistency_max_abs_logit_error": consistency_max_abs_error,
            "generation_prompt": args.prompt,
            "generation_seed": args.seed,
            "generation_sample": sample,
            "quality_note": "Generation is a qualitative sample, not a benchmark or evidence of language understanding.",
        }
        output = Path(args.checkpoint).parent / "evaluation_report.json"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Evaluation report saved to {output}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
