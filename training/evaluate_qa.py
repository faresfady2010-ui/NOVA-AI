"""Compare a base and instruction-tuned checkpoint on held-out QA prompts.

This produces qualitative generations and a manual review template; it does not
pretend that relevance or instruction following can be scored reliably by loss.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from model import ByteTokenizer, ModelConfig, NOVATransformer
from inference.generate import load_model, generate
from .instruction_dataset import load_examples, USER_MARKER, ASSISTANT_MARKER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/nova.pt")
    parser.add_argument("--tuned", default="checkpoints/nova-instruct.pt")
    parser.add_argument("--evaluation", default="data/qa_evaluation.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/qa_comparison.json")
    args = parser.parse_args()
    try:
        examples = load_examples(args.evaluation)
        base, tokenizer, _, device = load_model(args.base)
        tuned, _, _, _ = load_model(args.tuned, device)
        results = []
        for index, item in enumerate(examples):
            # QA evaluation records contain one user prompt and a reference answer;
            # only the prompt is passed to either model.
            user_text = item["messages"][0]["content"]
            prompt = USER_MARKER + user_text.strip() + "\n" + ASSISTANT_MARKER
            outputs = {}
            for name, model in (("base", base), ("instruction_tuned", tuned)):
                rng = torch.Generator(device=device).manual_seed(args.seed + index)
                outputs[name] = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
                                         temperature=0.7, top_k=40, device=device, generator=rng,
                                         stop_sequence="\n<|end|>")
            results.append({"id": item.get("id", str(index)), "language": item.get("language", "unspecified"),
                            "category": item.get("category", "unspecified"), "prompt": user_text,
                            "reference_answer": item["messages"][1]["content"], **outputs})
        report = {
            "evaluation_examples": len(results), "examples": results,
            "automated_quality_scores": None,
            "manual_review_rubric": ["relevance", "instruction_fidelity", "factual correctness", "language quality", "English/Arabic quality"],
            "interpretation": "Held-out qualitative comparison only. Reference overlap and loss are not reliable measures of understanding; have human reviewers score outputs before making capability claims.",
            "validation_data_used_for_training": False,
        }
        output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"QA comparison saved to {output_path}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"QA evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
