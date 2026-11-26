# NOVA AI

NOVA is an experimental language-model project implemented from scratch with PyTorch. It includes a UTF-8 byte tokenizer, a configurable decoder-only Transformer, local training/evaluation tools, checkpoint-based inference, a Flask API, and a responsive chat interface. It does **not** call commercial AI services and does not use pretrained model weights.

The included example corpus is tiny and intended only to verify that the pipeline runs. It cannot provide general-purpose language ability. A functioning chat UI is not evidence of intelligence; outputs can be incoherent, wrong, or incomplete.

## Architecture

- **Tokenizer:** each UTF-8 byte is one token (IDs 0–255); PAD/BOS/EOS occupy 256–258. It supports English, Arabic, and arbitrary Unicode without a learned vocabulary. Invalid partial UTF-8 on decode is replaced safely.
- **Embeddings:** token embeddings represent byte identity; learned positional embeddings represent sequence location.
- **Transformer blocks:** pre-layer-normalized multi-head causal self-attention plus a 4× GELU feed-forward network, each with residual connections and dropout. Attention masks all future positions.
- **Language-model head:** projects final normalized hidden states to vocabulary logits. Input/output token weights are tied.
- **Objective:** next-token cross-entropy, optimized using AdamW, cosine learning-rate decay, and gradient clipping.

The default model is small (4 layers, 4 heads, width 128, context 128 in the training command) and configurable. Byte-level modeling is easy to inspect, but substantially less token-efficient than subword tokenization.

## Requirements and installation

Python 3.10+ is recommended. CPU works; CUDA-enabled PyTorch is used automatically when available. Training speed and memory needs depend on sequence length, batch size, and model size.

```bash
git clone https://github.com/faresfady2010-ui/NOVA-AI.git
cd NOVA-AI
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build suitable for your CUDA version using the official PyTorch installation selector, then install the remaining requirements.

## Prepare datasets

Edit `data/training.txt` and `data/validation.txt`, or pass other UTF-8 text file paths with `--train` and `--validation`. Only use text you have the right to use. No dataset is downloaded automatically. The loader normalizes newlines, rejects empty files, adds boundary control tokens, and refuses identical non-empty lines across the two files. Keep documents/sources separated across splits to avoid less-obvious leakage too. The default sample files are deliberately small and for smoke testing only.

## Train

From the repository root:

```bash
python -m training.train --steps 500 --batch-size 16 --context 128
```

A checkpoint is written to `checkpoints/nova.pt`; periodic JSON-lines metrics are written beside it. To change architecture or paths:

```bash
python -m training.train --train data/training.txt --validation data/validation.txt \
  --checkpoint checkpoints/nova.pt --steps 2000 --batch-size 8 \
  --context 256 --layers 4 --heads 4 --width 128 --lr 0.0003 --eval-every 100
```

Choose a batch/context/model size that fits your hardware. Training begins with random initialization. It is normal for the tiny demo corpus to overfit and for generated text to be poor.

## Evaluate

```bash
python -m training.evaluate --checkpoint checkpoints/nova.pt --validation data/validation.txt --batches 20
```

The script reports held-out byte-token cross-entropy and perplexity. These metrics alone do not measure conversational usefulness or factual accuracy.

## Generate text

```bash
python -m inference.generate "Explain how plants use light." --checkpoint checkpoints/nova.pt \
  --max-new-tokens 120 --temperature 0.8 --top-k 40
```

Generation is autoregressive and uses only the checkpoint's model configuration and byte tokenizer. Context is cropped to the configured window. Temperature must be positive; top-k is optional.

## Launch the website

Train a checkpoint first, then run:

```bash
python -m backend.server
```

Visit <http://127.0.0.1:5000>. To select another checkpoint, set `NOVA_CHECKPOINT`. API endpoints: `GET /api/health` and `POST /api/chat` with a `messages` array of `{role, content}` objects plus optional `temperature` and `max_new_tokens`. The chat UI stores conversation history in the browser's local storage. Do not expose the development server publicly without adding appropriate production hosting, authentication, rate limiting, and security controls.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover byte round trips (including Arabic and Unicode), special tokens, model dimensions/loss, causal masking, optimizer updates, checkpoint round-trip, generation, and basic API validation/health.

## Limitations

- The bundled corpus is tiny and does not create useful general knowledge.
- Byte-level tokens consume context inefficiently; long contexts and training are expensive.
- No instruction-tuning, preference optimization, retrieval, tool use, safety fine-tuning, or distributed training is included.
- The conversational layer supplies formatted history; it does not add knowledge beyond learned weights.
- Confidence and factuality are not calibrated. NOVA cannot reliably identify when it lacks knowledge.
- Arabic is representable, but quality depends entirely on training data and evaluation.

## Roadmap

1. Collect and document permissioned, deduplicated multilingual corpora and robust evaluations.
2. Add tokenizer abstraction and compare efficient learned/subword tokenization.
3. Improve training resume support, checkpoint retention, mixed precision, and profiling.
4. Add reproducible benchmark suites, data-quality tools, and calibrated uncertainty behavior.
5. Explore scaling, instruction tuning, retrieval, and deployment hardening only after the small baseline is validated.

## License

MIT for project code. Dataset rights remain with their respective contributors; use only appropriately licensed data.
