# NanoTransformer — Scientific Documentation

## Scope & purpose
NanoTransformer is a compact, causal Transformer language model designed for next‑token prediction with a fixed trigram context (`context_len = 3`). The repository provides a full pipeline: corpus download, BPE tokenizer training, dataset construction, model training, evaluation, a trigram baseline, and latency benchmarking. Packaging metadata is included for PyPI‑style distribution with CLI entry points.

---

## Repository layout (key paths)
- Core package: `/Users/abhyudaymishra/NanoTransformer/nanotransformer/`
  - Model: `nanotransformer/model.py`
  - Dataset: `nanotransformer/dataset.py`
  - Training: `nanotransformer/train.py`
  - Evaluation: `nanotransformer/eval.py`
  - Baseline: `nanotransformer/baseline_trigram.py`
  - Latency: `nanotransformer/measure_latency.py`
  - Utilities: `nanotransformer/utils.py`
- Scripts: `/Users/abhyudaymishra/NanoTransformer/scripts/`
  - `download_corpus.py`, `train_tokenizer.py`, `prepare_data.py`, `install_deps.py`
- Config: `/Users/abhyudaymishra/NanoTransformer/configs/base.json`
- Packaging: `/Users/abhyudaymishra/NanoTransformer/pyproject.toml`
- License: `/Users/abhyudaymishra/NanoTransformer/LICENSE`
- Root wrappers: `/Users/abhyudaymishra/NanoTransformer/train.py`, `eval.py`, `baseline_trigram.py`, `measure_latency.py`

---

## Problem formulation
Given tokenized text, for each position *t*, the model predicts the next token from the previous three tokens:

```
x_{t-3}, x_{t-2}, x_{t-1}  ->  x_t
```

Loss is cross‑entropy over the next‑token prediction:

```
L = - (1/N) * sum_t log Pθ(x_t | x_{t-3}, x_{t-2}, x_{t-1})
```

Perplexity:

```
PPL = exp(L)
```

---

## Model architecture
Defined in `/Users/abhyudaymishra/NanoTransformer/nanotransformer/model.py`.

- Token embedding: `V x d`
- Positional embedding: `T x d` where `T = 3`
- `L` Transformer blocks, each with:
  - LayerNorm
  - Causal multi‑head self‑attention (upper‑triangular mask)
  - Residual + dropout
  - LayerNorm
  - 2‑layer FFN with GELU (width `ff_mult * d`)
  - Residual + dropout
- Output: only final position in context window
- LM head: linear projection to vocab, weight‑tied to token embedding

### Causality constraint
Upper‑triangular attention mask ensures no access to future positions:

```
mask_{ij} = 1 if j > i
```

---

## Parameterization (base config)
From `/Users/abhyudaymishra/NanoTransformer/configs/base.json`:

- `V = 10000`, `d = 128`, `L = 4`, `ff_mult = 4`, `h = 4`, `T = 3`

Approximate parameter count (tied embeddings):

```
Params ≈ V*d + L*(4 + 2*ff_mult)*d^2
       ≈ 10000*128 + 4*12*128^2
       ≈ 2.07M
```

---

## Data & tokenization pipeline
Scripts in `/Users/abhyudaymishra/NanoTransformer/scripts/`:

1) **Corpus download** (`download_corpus.py`)
   - Fetches Wikitext‑103 raw from Hugging Face mirror
   - Extracts ~50MB corpus to `data/corpus.txt`

2) **Tokenizer training** (`train_tokenizer.py`)
   - ByteLevel BPE with special tokens: `<pad>`, `<s>`, `</s>`, `<unk>`
   - Saves `artifacts/tokenizer.json`

3) **Tokenization + split** (`prepare_data.py`)
   - Encodes corpus into token IDs
   - Saves:
     - `artifacts/data_ids.pt` (full tensor)
     - `artifacts/splits.pt` (train/val/test splits)

---

## Dataset construction
`TrigramDataset` in `/Users/abhyudaymishra/NanoTransformer/nanotransformer/dataset.py`:

- Input `x`: three preceding tokens
- Target `y`: next token

---

## Training procedure
`nanotransformer/train.py`:

- Optimizer: AdamW
- Scheduler: linear warmup → cosine decay
- Loss: CrossEntropy
- Gradient clipping: `grad_clip`
- Logging: `--log-interval`
- Early cap: `--max-steps`
- Best checkpoint: `checkpoints/best.pt`
- Metrics: `artifacts/metrics.json`

---

## Evaluation
`nanotransformer/eval.py`:

- Computes val/test loss + perplexity
- Writes `artifacts/eval_metrics.json`

---

## Baseline (trigram LM)
`nanotransformer/baseline_trigram.py`:

Add‑k smoothing:

```
P(w | c) = (count(c,w) + k) / (count(c) + k*V)
```

Outputs val/test cross‑entropy + perplexity to `artifacts/baseline_metrics.json`.

---

## Latency measurement
`nanotransformer/measure_latency.py`:

- Uses `torch.inference_mode()`
- Random contexts (shape `1 x 3`)
- Reports mean/p50/p95/p99 ms
- Writes `artifacts/latency.json`

---

## Configuration (base)
`/Users/abhyudaymishra/NanoTransformer/configs/base.json`:

- `vocab_size`: 10000
- `d_model`: 128
- `n_layers`: 4
- `n_heads`: 4
- `ff_mult`: 4
- `context_len`: 3
- `dropout`: 0.0
- `batch_size`: 256
- `lr`: 3e-4
- `warmup_steps`: 1000
- `weight_decay`: 0.01
- `num_workers`: 0

---

## Packaging & CLI
`pyproject.toml` defines:

- PyPI name: **nanotransformer**
- CLI entry points:
  - `nanotransformer-train`
  - `nanotransformer-eval`
  - `nanotransformer-baseline`
  - `nanotransformer-latency`

---

## Reproducibility
- Seed control in `nanotransformer/utils.py`
- Config‑driven runs
- Tokenizer JSON is stored with artifacts

---

## Limitations & future extensions
- No unit tests yet
- Fixed context length = 3 limits capacity
- No quantized/mixed‑precision inference path
- Minimal data cleaning beyond raw ingestion
