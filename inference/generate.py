"""Local text generation using a NOVA checkpoint."""
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
    model = NOVATransformer(config).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    return model, ByteTokenizer(), config, device


def generate(model: NOVATransformer, tokenizer: ByteTokenizer, prompt: str, *, max_new_tokens: int = 120, temperature: float = 0.8, top_k: int = 40, device=None) -> str:
    if max_new_tokens < 0: raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0: raise ValueError("temperature must be greater than zero")
    if top_k < 0: raise ValueError("top_k must be non-negative")
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt) or [tokenizer.BOS]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = context[:, -model.config.context_length:]
            logits, _ = model(window); logits = logits[:, -1, :] / temperature
            # Control tokens are never emitted as user-facing generated content.
            logits[:, tokenizer.PAD] = float("-inf"); logits[:, tokenizer.BOS] = float("-inf"); logits[:, tokenizer.EOS] = float("-inf")
            if top_k > 0 and top_k < logits.size(-1):
                cutoff = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            context = torch.cat((context, next_id), dim=1)
    return tokenizer.decode(context[0, len(ids):].tolist())


def main():
    p=argparse.ArgumentParser(); p.add_argument("prompt"); p.add_argument("--checkpoint",default="checkpoints/nova.pt"); p.add_argument("--max-new-tokens",type=int,default=120); p.add_argument("--temperature",type=float,default=.8); p.add_argument("--top-k",type=int,default=40); a=p.parse_args()
    model, tokenizer, _, device = load_model(a.checkpoint); print(generate(model,tokenizer,a.prompt,max_new_tokens=a.max_new_tokens,temperature=a.temperature,top_k=a.top_k,device=device))

if __name__ == "__main__": main()
