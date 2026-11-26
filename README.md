# NOVA AI

NOVA is an experimental language-model project implemented from scratch with PyTorch. It includes a UTF-8 byte tokenizer, a configurable decoder-only Transformer, local training/evaluation tools, checkpoint-based inference, a Flask API, and a responsive chat interface. It does **not** call commercial AI services and does not use pretrained model weights.

The included bilingual corpus is intentionally tiny and only suitable for a pipeline smoke test. Training this demo does not produce a capable general-purpose assistant. Generated text can be incoherent, wrong, or incomplete; output is not evidence of understanding.

## Install

Python 3.10+ is recommended. CPU works; CUDA-enabled PyTorch is selected automatically when available.

```bash
git clone https://github.com/faresfady2010-ui/NOVA-AI.git
cd NOVA-AI
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build suitable for your CUDA version from the official PyTorch installation selector, then install the remaining requirements.

## Data

The project reads UTF-8 text from `data/training.txt` and `data/validation.txt`. Add only data you have permission to use. It does not download corpora. The loader rejects empty files and identical non-empty lines across splits; for serious experiments also ensure that documents and sources do not overlap. The bundled examples are small, original demonstrations, not meaningful training material.

## Train a model

Run from the repository root:

```bash
python -m training.train
```

This starts from random initialization using defaults in `model/config.py` and saves `checkpoints/nova.pt`, `checkpoints/training.jsonl`, and `checkpoints/training_report.json`. Example overrides:

```bash
python -m training.train --steps 2000 --batch-size 8 --context 128 \
  --layers 4 --heads 4 --width 128 --lr 0.0003 --eval-every 100 \
  --train data/training.txt --validation data/validation.txt
```

The training report records model configuration, trainable parameter count, tokenized dataset sizes, steps, initial/final train and validation loss, duration, seed, and hardware. Results should be interpreted honestly: low training loss on tiny data may mean memorization, not generalization. Runtime depends on hardware; CPU execution may be slow.

## Evaluate

```bash
python -m training.evaluate
```

Evaluation prints and writes `checkpoints/evaluation_report.json` with deterministic train/validation next-byte loss and perplexity, a sampled continuation, checkpoint reload logit consistency, and the checkpoint step. A text sample is a qualitative inspection only; it is not a language-capability benchmark. Checkpoint consistency measures serialization fidelity, not quality.

## Generate

```bash
python -m inference.generate --prompt "What is artificial intelligence?"
```

Options include `--checkpoint`, `--max-new-tokens`, `--temperature`, `--top-k`, and `--seed`. The seed controls sampling reproducibly on a given compatible device/runtime. Context is cropped to the model's configured context length. Incomplete UTF-8 at the end of a generation is decoded with replacement characters.

## Web interface

After training a checkpoint:

```bash
python -m backend.server
```

Open <http://127.0.0.1:5000>. Choose another checkpoint with the `NOVA_CHECKPOINT` environment variable. Do not expose the development server publicly without production hosting, authentication, rate limiting, and security controls.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover tokenizer round trips, model shapes and loss, causal masking, optimizer parameter updates, checkpoint round-trip, deterministic seeded generation, and basic backend validation.

## Architecture

- UTF-8 byte IDs 0–255 encode arbitrary Unicode; PAD/BOS/EOS use 256–258.
- Learned token and position embeddings feed a stack of pre-normalized Transformer blocks.
- Each block has multi-head causal self-attention, residual connections, and a 4× GELU feed-forward network.
- The output projection predicts next-byte logits; input/output token weights are tied.
- Training uses cross-entropy, AdamW, cosine learning-rate decay, and gradient clipping.

The byte representation is easy to understand and language-independent but is inefficient compared with a subword tokenizer. Default width/layers/context are small, configurable experimental settings rather than a claim of useful capability.

## Limitations and future work

NOVA has no pretrained knowledge, instruction tuning, retrieval, tool use, calibrated uncertainty, or robust multilingual benchmark. Its conversational wrapper supplies history but no hardcoded answers. The tiny bundled corpus is insufficient for general language competence. Future work should prioritize licensed and deduplicated corpora, better tokenization, rigorous evaluation, training resume/mixed precision, and data-quality checks before scaling.
