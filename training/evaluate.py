"""Compute loss/perplexity for a trained checkpoint on held-out local text."""
import argparse, math
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from .dataset import TokenBatcher, load_datasets


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default="checkpoints/nova.pt"); p.add_argument("--validation", default="data/validation.txt"); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--batches", type=int, default=20)
    a = p.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(checkpoint["config"]); _, val = load_datasets("data/training.txt", a.validation, ByteTokenizer())
    batcher = TokenBatcher(val, config.context_length, device); model = NOVATransformer(config).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    losses=[]
    with torch.no_grad():
        for _ in range(a.batches):
            x,y=batcher.batch(a.batch_size); _,loss=model(x,y); losses.append(loss.item())
    mean=sum(losses)/len(losses)
    print(f"validation_loss={mean:.4f} perplexity={math.exp(min(mean, 80)):.2f}")

if __name__ == "__main__": main()
