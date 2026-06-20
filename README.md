# LLMs from Scratch

GPT-2 inference engine from scratch. Zero external dependencies.

Implements GPT-2 Small (124M) and GPT-2 Medium (355M) from scratch — loading real pre-trained weights from HuggingFace — then running autoregressive generation entirely in Go. No CGo, no CUDA, no external libraries.

---

## Features

- **Three profiles**: `debug` (mock 2-layer), `small` (GPT-2 Small 124M), `medium` (GPT-2 Medium 355M)
- **KV-cached autoregressive generation** — prefill prompt once, decode one token at a time
- **Full sampling pipeline**: RepPen → Temperature → TopP → Softmax → Sample (or argmax at T=0)
- **Weight tying**: output projection shares token embedding matrix (`logits = h · WTEᵀ`)
- **Bidirectional strict mode**: validates every tensor shape, flags missing and extra keys
- **GPT-2 BPE tokenizer**: real vocabulary + merge rules loaded from `assets/tokenizer/`
- **Deterministic mode**: `-seed` flag for reproducible outputs
- **EOS detection**: auto-stops at token 50256

---

## Model Profiles

| Profile | Config | Vocab | Embed | Layers | Heads | Params | Tokenizer | Weights File |
|---------|--------|-------|-------|--------|-------|--------|-----------|-------------|
| `debug` | inline in `main.go` | 1,000 | 32 | 2 | 4 | ~1M (untrained) | Mock (6 tokens) | None |
| `small` | `config.DefaultConfig` | 50,257 | 768 | 12 | 12 | 124,439,808 | BPE from `assets/tokenizer/` | `gpt2_124M.bin` (196 tensors) |
| `medium` | `config.GPT2Medium` | 50,257 | 1,024 | 24 | 16 | 354,823,168 | BPE from `assets/tokenizer/` | `gpt2_medium.bin` (388 tensors) |

---

## Quick Start

### debug (no weights needed, instant)

```
go run ./cmd/main/... -profile=debug -prompt="Hello" -max_tokens=10
```

### small (GPT-2 Small 124M)

```
# Step 1: Convert weights from HuggingFace
pip install torch transformers
python3 scripts/convert_gpt2_weights.py --model gpt2

# Step 2: Greedy generation (deterministic)
go run ./cmd/main/... -profile=small -weights=gpt2_124M.bin -format=bin \
  -prompt="Hello" -temperature=0 -strict

# Step 3: Sampling
go run ./cmd/main/... -profile=small -weights=gpt2_124M.bin -format=bin \
  -prompt="The meaning of life is" -temperature=0.8 -top_p=0.9 -seed=42
```

### medium (GPT-2 Medium 355M)

```
# Convert weights
python3 scripts/convert_gpt2_weights.py --model gpt2-medium

# Run with sampling
go run ./cmd/main/... -profile=medium -weights=gpt2_medium.bin -format=bin \
  -prompt="Once upon a time" -temperature=0.8 -top_p=0.9 -seed=7
```

---

## CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-profile` | string | `debug` | `debug` \| `small` \| `medium` |
| `-weights` | string | `""` | Path to weights file (skip for debug) |
| `-format` | string | `json` | `json` or `bin` |
| `-prompt` | string | `"The"` | Input text |
| `-max_tokens` | int | 30 | Tokens to generate |
| `-temperature` | float | 1.0 | 0 = greedy argmax |
| `-top_p` | float | 1.0 | Nucleus threshold (1.0 = off) |
| `-repetition_penalty` | float | 1.0 | >1.0 penalizes repeated tokens |
| `-seed` | int | 0 | Random seed (0 = time-based) |
| `-strict` | bool | false | Fail on missing/extra weights |

---

## Architecture

### Data Flow

```
prompt → Tokenizer(Encode) → tokenIDs → Embeddings(token + position)
  → [TransformerBlock × N] → LayerNorm → OutputProj
  → logits → Sampling pipeline → nextTokenID → append → loop → Tokenizer(Decode) → text
```

### Engine Diagram

```
┌──────────────────────────────────────────────────────────┐
│                      inference.Generate                   │
│  prefill: forwardWithCache(ids, nil) → logits + KVCache │
│  decode loop:                                            │
│    extract last-token logits → RepPen → Temp → TopP      │
│    → Softmax → Sample → append → forwardWithCache(token) │
└──────┬───────────────────────────────────────────────────┘
       │ ForwardWithCache(tokenIDs, cache)
       ▼
┌──────────────────────────────────────────────────────────┐
│                        GPTModel                           │
│  Embeddings(ids, startPos) →                              │
│    [block.ForwardWithCache × N layers] → LayerNorm →      │
│    OutputProj(weight = TokenEmbedding) → logits           │
│  KVCache{Keys[], Values[], SeqLen}                        │
└──┬────────┬────────┬───────────────────────────┬─────────┘
   │        │        │                           │
   ▼        ▼        ▼                           ▼
┌──────┐ ┌─────────┐ ┌─────────┐           ┌─────────────┐
│Embed │ │Block[0] │ │Block[1] │ ...       │ Final LN    │
│dings │ │         │ │         │           │ + OutProj   │
└──┬───┘ └──┬──────┘ └──┬──────┘           └─────────────┘
   │        │           │
   ▼        ▼           ▼
┌──────────┐  ┌──────────────────────┐
│TokenEmb  │  │ Pre-LN Transformer   │
│(V,1,Emb) │  │ Block:                │
│PosEmb    │  │  x+Attn(LN(x))       │
│(ctx,1,Em)│  │  x+FFN(LN(x))        │
└──────────┘  │                      │
              │ SelfAttention        │
              │  Wq,Wk,Wv,Wo(linear)│
              │  d_k = d_model       │
              │                      │
              │ FeedForward          │
              │  Linear→GELU→Linear  │
              └──────────────────────┘
```

### Sampling Pipeline Order

```
logits → ApplyRepetitionPenalty → ApplyTemperature
  → ApplyTopP(nucleus) → Softmax → Sample(CDF + random draw) → tokenID
```

When `temperature=0`, Sampling is skipped entirely: `argmax(logits)`.

### KV-Cached Inference

1. **Prefill**: process the full prompt through `ForwardWithCache(ids, nil)`. All layers compute and store K/V tensors. Returns logits + `KVCache{Keys, Values, SeqLen=len(ids)}`.
2. **Decode**: for each new token, call `ForwardWithCache([tokenID], prevCache)`. Each layer concatenates past K/V with new K/V via `ConcatSeq`, computes attention over the full sequence, and returns updated K/V. Only the last-token logit is sampled.

---

## Package Breakdown

| Package | Files | What it does | Key types |
|---------|-------|-------------|-----------|
| `tensor` | `tensor.go` | 3D tensor `[batch, seq, embed]`, flat `[]float64` | `Tensor` |
| `math` | `math.go` | Softmax, GELU (tanh approx), LayerNorm, Dropout | |
| `linear` | `linear.go` | `y = x·Wᵀ + b` | `Linear{Weight, Bias}` |
| `attention` | `attention.go` | QKV projections, scaled dot-product, KV cache | `SelfAttention` |
| `model` | `model.go`, `block.go`, `embeddings.go`, `ffn.go` | Full transformer stack | `GPTModel`, `TransformerBlock`, `FeedForward`, `Embeddings`, `KVCache` |
| `config` | `config.go` | Model hyperparameters + presets | `Config` |
| `inference` | `inference.go`, `sampling.go` | Generation loop + sampling | |
| `tokenizer` | `tokenizer.go` | BPE tokenizer (mock + GPT-2 file) | `Tokenizer` |
| `weights` | `weights.go` | JSON/binary weight load/save with strict validation | |

---

## Weight Format

Binary format (preferred — smaller and faster):

```
┌──────────┬─────────────────────┐
│ Header   │ Magic:   0x4C4C4D00 │
│          │ Version: 1          │
│          │ Tensors: N          │
├──────────┼─────────────────────┤
│ Tensor 1 │ Key length + key    │
│          │ Dim count + dims    │
│          │ float32 data []     │
├──────────┼─────────────────────┤
│ Tensor N │ ...                 │
└──────────┴─────────────────────┘
```

JSON format is also supported but ~3x larger.

All weights use `float32`. The output projection weight is excluded from the file (it's weight-tied to the token embedding in Go). Bias tensors are included for all QKV and FFN layers (GPT-2 uses biases everywhere).

---

## Testing

```
go test -count=1 ./...
```

57 tests covering all packages: tensor operations, math functions, linear layers, attention, transformer blocks, full model forward pass, tokenizer (mock + GPT-2 BPE), weight save/load round-trips, sampling functions.

---

## Known Limitations

- **GPT-2 architecture only** — no RoPE, SiLU, RMSNorm, or GQA (the Llama-family building blocks used by every modern model)
- **`SelfAttention.d_k = d_model`** — a simplifying approximation; attention probabilities are broader than proper multi-head `d_model / n_heads`
- **Inference only** — no training loop. Dropout is wired but always disabled at runtime
- **CPU only** — no GPU acceleration, no quantization
- **2019 base model ceiling** — GPT-2 Small (124M) with greedy argmax degenerates into repetition loops (expected model behavior, not a bug)
- **Sampling performance**: top-p sorts 50K vocabulary every token; Medium (355M) is ~3x slower per token than Small (124M)

---

## References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. 2019
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — weight source for the Python converter
