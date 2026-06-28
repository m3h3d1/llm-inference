# LLMs from Scratch

Inference engine for GPT-2 and Llama-family models, written in Go. Zero external dependencies.

Supports GPT-2 Small (124M), GPT-2 Medium (355M), and SmolLM2-135M/360M/1.7B via GGUF — loading real pre-trained weights and running autoregressive generation entirely in Go. No CGo, no CUDA, no external libraries.

---

## Features

- **Two architectures**: GPT-2 (GELU, LayerNorm, learned position embeddings, full MHA) and Llama (RoPE, RMSNorm, SiLU/SwiGLU, GQA)
- **GGUF weight loading**: parses GGUF metadata, loads F16/F32/Q8_0 tensors, dequantizes on load
- **KV-cached autoregressive generation** — prefill prompt once, decode one token at a time
- **Full sampling pipeline**: RepPen → Temperature → TopP → Softmax → Sample (or argmax at T=0)
- **Weight tying**: output projection shares token embedding matrix
- **Bidirectional strict mode**: validates every tensor shape, flags missing and extra keys
- **GPT-2 BPE tokenizer**: real vocabulary + merge rules; also loaded from GGUF
- **Goroutine parallelism**: linear layers, attention heads, output logits parallelized at 500K-op threshold
- **Row-slice optimization**: direct `[]float64` access eliminates At/Set function call overhead (6-9× speedup)
- **Aggregated profiling**: `LLM_BENCH_DEBUG=1` prints timing counters for hot paths
- **Deterministic mode**: `-seed` flag for reproducible outputs
- **EOS detection**: auto-stops generation
- **Benchmark infrastructure**: `go test -tags=bench` measures prefill/decode throughput
- **29 concept documents** in `concepts/`: architecture, tensor ops, numerics, performance, testing, etc.

---

## Model Profiles

### GPT-2 (JSON/binary weights via Python conversion)

| Profile | Vocab | Embed | Layers | Heads | Params | Tokenizer | Weights |
|---------|-------|-------|--------|-------|--------|-----------|---------|
| `debug` | 1,000 | 32 | 2 | 4 | ~1M | Mock (6 tokens) | None |
| `small` | 50,257 | 768 | 12 | 12 | 124M | BPE from files | `gpt2_124M.bin` |
| `medium` | 50,257 | 1,024 | 24 | 16 | 355M | BPE from files | `gpt2_medium.bin` |

### Llama (GGUF, no Python conversion needed)

| Profile | Vocab | Embed | Layers | Heads | KV Heads | Params | GGUF |
|---------|-------|-------|--------|-------|----------|--------|------|
| `smollm2_135m` | 49,152 | 576 | 30 | 9 | 3 | 135M | `SmolLM2-135M-Q8_0.gguf` |
| `smollm2_360m` | 49,152 | 960 | 32 | 15 | 5 | 360M | `SmolLM2-360M-Q8_0.gguf` |
| `smollm2_1_7b` | 49,152 | 2,048 | 24 | 32 | 32 | 1.7B | `SmolLM2-1.7B-Q8_0.gguf` |

---

## Quick Start

### debug (no weights needed, instant)

```
go run ./cmd/main/... -profile=debug -prompt="Hello" -max_tokens=10
```

### SmolLM2-135M via GGUF

```
# Download from HuggingFace: https://huggingface.co/smollm2/SmolLM2-135M-GGUF
go run ./cmd/main/... -profile=smollm2_135m -gguf=SmolLM2-135M-Q8_0.gguf \
  -prompt="Hello" -temperature=0
```

### GPT-2 Small 124M

```
# Step 1: Convert weights from HuggingFace
pip install torch transformers
python3 scripts/convert_gpt2_weights.py --model gpt2

# Step 2: Greedy generation
go run ./cmd/main/... -profile=small -weights=gpt2_124M.bin -format=bin \
  -prompt="Hello" -temperature=0 -strict

# Step 3: Sampling
go run ./cmd/main/... -profile=small -weights=gpt2_124M.bin -format=bin \
  -prompt="The meaning of life is" -temperature=0.8 -top_p=0.9 -seed=42
```

### GPT-2 Medium 355M

```
python3 scripts/convert_gpt2_weights.py --model gpt2-medium
go run ./cmd/main/... -profile=medium -weights=gpt2_medium.bin -format=bin \
  -prompt="Once upon a time" -temperature=0.8 -top_p=0.9 -seed=7
```

---

## CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-profile` | string | `debug` | `debug` \| `small` \| `medium` \| `smollm2_135m` \| `smollm2_360m` \| `smollm2_1_7b` |
| `-weights` | string | `""` | Path to GPT-2 weights file (skip for debug, not needed with `-gguf`) |
| `-format` | string | `json` | `json` or `bin` (GPT-2 only) |
| `-gguf` | string | `""` | Path to GGUF weights file (Llama only) |
| `-prompt` | string | `"The"` | Input text |
| `-max_tokens` | int | 30 | Tokens to generate |
| `-temperature` | float | 1.0 | 0 = greedy argmax |
| `-top_p` | float | 1.0 | Nucleus threshold (1.0 = off) |
| `-repetition_penalty` | float | 1.0 | >1.0 penalizes repeated tokens |
| `-seed` | int | 0 | Random seed (0 = time-based) |
| `-strict` | bool | false | Fail on missing/extra weights (GPT-2 only) |

---

## Architecture

### Data Flow

```
prompt → Tokenizer(Encode) → tokenIDs → Embed → [Block × N] → FinalNorm
  → OutputProj → logits → Sampling → nextTokenID → append → loop
  → Tokenizer(Decode) → text
```

### Engine Diagram

```
┌──────────────────────────────────────────────────────────┐
│                      inference.Generate                  │
│  prefill: forwardWithCache(ids, nil) → logits + KVCache  │
│  decode loop:                                            │
│    extract last-token logits → RepPen → Temp → TopP      │
│    → Softmax → Sample → append → forwardWithCache(token) │
└──────┬───────────────────────────────────────────────────┘
       │ ForwardWithCache(tokenIDs, cache)
       ▼
┌──────────────────────────────────────────────────────────┐
│              Model (GPT2.Model or Llama.Model)           │
│  Embed(ids, startPos) → [block.ForwardWithCache × N]     │
│    → FinalNorm → OutputProj(weight = TokenEmbedding)     │
│  KVCache{Keys[], Values[], SeqLen}                       │
└──┬────────┬────────┬───────────────────────────┬─────────┘
   │        │        │                           │
   ▼        ▼        ▼                           ▼
┌──────┐ ┌─────────┐ ┌─────────┐           ┌─────────────┐
│Embed │ │Block[0] │ │Block[1] │ ...       │ FinalNorm   │
│dings │ │         │ │         │           │ + OutProj   │
└──┬───┘ └──┬──────┘ └──┬──────┘           └─────────────┘
   │        │           │
   ▼        ▼           ▼
┌──────────┐  ┌───────────────────────┐
│ GPT-2:   │  │ GPT-2 Block:          │
│ TokenEmb │  │  x + Attn(LN(x))      │
│ + PosEmb │  │  x + FFN(GELU(LN(x))) │
│          │  │ SelfAttention: Q,K,V  │
│ Llama:   │  │   full MHA, all bias  │
│ TokenEmb │  │ FeedForward:          │
│ (no pos) │  │   Linear → GELU → Lin │
└──────────┘  └───────────────────────┘
```

### Llama Block Detail

```
┌────────────────────────────────┐
│ Llama Block:                   │
│  x + Attn(RMSNorm(x))          │
│  x + FFN(RMSNorm(x))           │
│                                │
│ Attention: GQA                 │
│  Wq (nHeads × headDim)         │
│  Wk (nKVHeads × headDim)       │
│  Wv (nKVHeads × headDim)       │
│  Wo (nHeads × headDim)         │
│  RoPE applied to Q,K (headDim) │
│  headDim = 64                  │
│                                │
│ SwiGLU FFN:                    │
│  Gate → SiLU → × Up → Down    │
│  (no bias anywhere)            │
│                                │
│ RMSNorm: no shift (β=0)        │
└────────────────────────────────┘
```

### Sampling Pipeline Order

```
logits → ApplyRepetitionPenalty → ApplyTemperature
  → ApplyTopP(nucleus) → Softmax → Sample(CDF + random draw) → tokenID
```

When `temperature=0`, Sampling is skipped entirely: `argmax(logits)`.

### KV-Cached Inference

1. **Prefill**: process the full prompt through `ForwardWithCache(ids, nil)`. All layers compute and store K/V tensors. Causal mask prevents attending to future tokens. Returns logits + `KVCache{Keys, Values, SeqLen=len(ids)}`.
2. **Decode**: for each new token, call `ForwardWithCache([tokenID], prevCache)`. Each layer concatenates past K/V with new K/V via `ConcatSeq`, computes attention over the full sequence (no causal mask needed — past is always before present by cache construction), and returns updated K/V. Only the last-token logit is sampled.

---

## Package Breakdown

| Package | Files | What it does | Key types |
|---------|-------|-------------|-----------|
| `tensor` | `tensor.go` | 3D tensor `[batch, seq, embed]`, flat `[]float64` | `Tensor` |
| `math` | `math.go` | Softmax, GELU, LayerNorm, RMSNorm, Dropout, causal mask | |
| `linear` | `linear.go` | `y = x·Wᵀ + b`, Row-slice + parallel paths | `Linear{Weight, Bias}` |
| `model/gpt2` | `model.go`, `block.go`, `attention.go`, `embeddings.go`, `ffn.go` | GPT-2 full stack | `Model`, `TransformerBlock`, `SelfAttention`, `FeedForward`, `Embeddings` |
| `model/llama` | `model.go`, `block.go`, `attention.go`, `rmsnorm.go`, `rope.go`, `activations.go` | Llama full stack | `Model`, `LlamaBlock`, `LlamaAttention`, `RopeTables` |
| `model` | `common.go` | Shared types | `KVCache` |
| `config` | `config.go` | Hyperparameters + presets | `Config` |
| `inference` | `inference.go`, `sampling.go` | Generation loop + sampling | `Model` interface |
| `tokenizer` | `tokenizer.go`, `gguf.go` | BPE (mock + file + GGUF) | `Tokenizer` |
| `weights` | `weights.go`, `common.go`, `gguf.go` | JSON/binary + GGUF load | `WeightData`, `LoadConfigFromGGUF` |
| `gguf` | `reader.go`, `types.go` | GGUF binary parser | `Reader`, `TensorInfo`, `Value` |
| `internal/benchprof` | `prof.go` | Aggregated timing counters | `RecordLinearForward`, etc. |

---

## Weight Format

### GGUF (Llama models)

Self-contained binary format with embedded tokenizer. No Python conversion needed.

```
┌──────────┬─────────────────────────────────┐
│ Header   │ Magic: 0x46554747 ("GGUF")      │
│          │ Version: 3                       │
│          │ TensorCount: N                   │
├──────────┼─────────────────────────────────┤
│ Metadata │ Typed KV pairs (strings, ints,   │
│ KV       │ floats, arrays) describing the   │
│          │ model architecture & tokenizer   │
├──────────┼─────────────────────────────────┤
│ Tensor   │ Name, dims, type (F32/F16/Q8_0), │
│ Index    │ offset → tensor data region      │
├──────────┼─────────────────────────────────┤
│ Tensor   │ Aligned blocks of numeric data   │
│ Data     │ Q8_0: F16 scale + 32 × int8     │
└──────────┴─────────────────────────────────┘
```

Types loaded: strings, int32, int64, float32, float64, bool, arrays. Q8_0 dequantized on load (block of 32 int8 values + F16 scale → `[]float64`). F16 → float64 via IEEE 754 binary16 arithmetic conversion.

### Custom binary (GPT-2)

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

JSON format also available for GPT-2 but ~3x larger. The output projection weight is excluded (weight-tied to token embedding in Go).

---

## Testing & Benchmarks

```
go test -count=1 ./...
```

Tests cover all 12 packages: tensor ops, math functions, linear layers, GPT-2/Llama blocks, full model forward pass, BPE tokenizer (mock + GGUF), GGUF parser (synthetic), weight load round-trips, sampling, and generation.

### Benchmarks

```
go test -tags=bench -run TestSpeed -v ./inference/
```

Measures prefill and decode throughput for SmolLM2-135M Q8_0. Current results (BENCHMARK3.md): prefill-128t ~18s (7.1 tok/s), decode-1t ~0.58s (1.7 tok/s) on i7-1185G7.

---

## Known Limitations

- **float64 only** — all tensor operations use `float64` to avoid precision loss in tail computations; float32 conversion regressed performance due to At/Set call overhead, not arithmetic width
- **CPU only** — no GPU acceleration
- **Inference only** — no training loop. Dropout is wired but always disabled at runtime
- **No PGO, tiling, or cache-friendly matmul** — evaluated but deferred; Row-slice optimization provides ~10× speedup without algorithmic changes
- **Sampling performance**: top-p sorts 50K vocabulary every token

---

## References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. 2019
- [SmolLM2](https://huggingface.co/blog/smollm2) — Allal et al. 2024
- [RoPE](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding (Su et al. 2021)
- [GQA](https://arxiv.org/abs/2305.13245) — Grouped Query Attention (Ainslie et al. 2023)
- [GGUF Spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) — GGUF binary format
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — GPT-2 weight source (Python converter)
- [LLM from Scratch](https://sebastianraschka.com/llms-from-scratch) - LLM from Scratch book by Sebastian Raschka
