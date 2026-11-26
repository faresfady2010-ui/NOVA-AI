# NOVA AI — Experimental Language Model

NOVA is an independently implemented PyTorch decoder-only Transformer with a UTF-8 byte tokenizer, local pretraining and instruction fine-tuning pipelines, evaluation utilities, and a local chat UI. It uses no external language-model API and no pretrained weights. Nothing in this project should be taken as evidence of general intelligence or reliable question answering.

## Install and test

Python 3.10+:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Stage 1: Train a base next-byte model

Run from the repository root. This is required before supervised fine-tuning; the project does not ship a trained checkpoint.

```bash
python -m training.train
```

It trains from random initialization on `data/training.txt` and evaluates on `data/validation.txt`. The default model and training settings live in `model/config.py`. Checkpoint output: `checkpoints/nova.pt`.

## Stage 2: Instruction fine-tuning

The JSONL conversation examples in `data/instructions_train.jsonl` and `data/instructions_validation.jsonl` are original, small demonstrations. Training labels apply only to assistant response bytes/end markers; user messages and role markers provide context but are masked with PyTorch's `ignore_index=-100`. Validation records are used only for loss evaluation, never optimizer updates.

```bash
python -m training.finetune --checkpoint checkpoints/nova.pt
```

The command loads the model configuration and weights from the project's own base checkpoint, continues training with AdamW, cosine learning-rate decay, gradient clipping, logs validation assistant-token loss, then writes `checkpoints/nova-instruct.pt` and a JSON report. Options include `--train`, `--validation`, `--output`, `--steps`, `--batch-size`, `--lr`, and `--seed`. The context size must match the base checkpoint. For longer examples, increase context when training the base model; examples are cropped from the left to the model window and must retain supervised answer tokens.

## Stage 3: QA evaluation and before/after comparison

`data/qa_evaluation.jsonl` is separate from both fine-tuning splits. It is not loaded by the fine-tuning process. After both checkpoints exist, run:

```bash
python -m training.evaluate_qa --base checkpoints/nova.pt --tuned checkpoints/nova-instruct.pt
```

The report `checkpoints/qa_comparison.json` includes matched base/tuned generations for each held-out prompt, English/Arabic labels, reference answers, and a manual review rubric for relevance, instruction fidelity, factual correctness, and language quality. Automatic free-form answer quality scoring is not implemented: token loss and reference overlap are not reliable measures of understanding. Human evaluation is required before making quality claims.

To inspect only held-out assistant-token loss during tuning, it is printed during `training.finetune`; evaluation data never contributes gradients.

## Ask NOVA

```bash
python -m inference.generate --prompt "<|user|>\nWhat is artificial intelligence?\n<|assistant|>\n" --checkpoint checkpoints/nova-instruct.pt --max-new-tokens 100 --temperature 0.7 --top-k 40 --seed 42
```

The generator supports top-k, temperature, output limit and seeded sampling. The conversation manager uses the same `<|user|>` / `<|assistant|>` markers, preserves only the latest in-window messages, and has no cross-request state; callers must pass the history belonging to one conversation only. The QA evaluator stops at the learned `\n<|end|>` boundary when generated.

## Format for adding examples

Each JSONL line is a separate conversation, with alternating user and assistant messages, beginning with user and ending with assistant:

```json
{"messages":[{"role":"user","content":"Explain evaporation."},{"role":"assistant","content":"Evaporation changes liquid into gas at a surface."}]}
```

Don't place validation/evaluation prompts in training. Keep examples concise relative to context size and use only data you have permission to use. Check exact-line/source overlap manually as well; the JSONL loader validates schema and role order, not semantic deduplication.

## Architecture and limitations

The model retains its original byte tokenizer, token and position embeddings, causal multi-head self-attention, residual pre-layer normalization, GELU feed-forward blocks, and next-token output head. Fine-tuning changes only the data format, target loss mask, optimizer run, and checkpoint metadata; it does not replace the architecture. UTF-8 bytes represent English and Arabic but use context inefficiently. The included datasets are tiny teaching examples and are not sufficient for broad knowledge, reliable instruction following, or robust Arabic performance. The software pipeline can be exercised on CPU, but training may be slow. The model can produce fluent-looking falsehoods, and uncertainty calibration, safety tuning, broad held-out benchmarks, and reliable automated QA scoring remain unfinished.

The base and fine-tuned checkpoints are generated artifacts and are not committed. Do not claim training or evaluation results until you run the commands locally and inspect their reports and generations.
