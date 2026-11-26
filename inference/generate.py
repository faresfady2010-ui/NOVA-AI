"""Local text generation from a NOVA checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer


def load_model(checkpoint_path: str | Path, device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("tokenizer", {}).get("type") != "utf8-byte":
        raise ValueError("Checkpoint tokenizer metadata is missing or unsupported")
    config = ModelConfig.from_dict(checkpoint["config"])
    model = NOVATransformer(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ByteTokenizer(), config, device


def generate(model: NOVATransformer, tokenizer: ByteTokenizer, prompt: str, *,
             max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 40,
             device=None, generator: torch.Generator | None = None) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt) or [tokenizer.BOS]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = context[:, -model.config.context_length:]
            logits, _ = model(window)
            logits = logits[:, -1, :] / temperature
            logits[:, tokenizer.PAD] = float("-inf")
            logits[:, tokenizer.BOS] = float("-inf")
            logits[:, tokenizer.EOS] = float("-inf")
            if top_k > 0 and top_k < logits.size(-1):
                cutoff = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1, generator=generator)
            context = torch.cat((context, next_id), dim=1)
    return tokenizer.decode(context[0, len(ids):].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Text to continue")
    parser.add_argument("--checkpoint", default="checkpoints/nova.pt")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.temperature <= 0 or args.top_k < 0:
        parser.error("max-new-tokens/top-k must be non-negative and temperature must be positive")
    try:
        model, tokenizer, _, device = load_model(args.checkpoint)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        print(generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k, device=device,
                       generator=generator))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Generation failed: {exc}\n")


if __name__ == "__main__":
    main()
