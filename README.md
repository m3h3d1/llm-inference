# LLMs from Scratch

Inference engine for GPT-2 and Llama-family models in Go. Zero external dependencies.

Supports GPT-2 Small (124M), GPT-2 Medium (355M) and SmolLM2-135M/360M/1.7B via GGUF.
No CGo, no CUDA, no Python conversion needed for Llama models.

---

## Quick Start

### Debug (no weights, instant)

```
go run ./cmd/main/... -profile=debug -prompt="Hello" -max_tokens=10
```

### SmolLM2-135M via GGUF

```bash
# Greedy
go run ./cmd/main/... -profile=smollm2_135m -gguf=SmolLM2-135M-Q8_0.gguf \
  -prompt="Hello" -temperature=0

# Instruct model with ChatML
go run ./cmd/main/... -profile=smollm2_135m -gguf=SmolLM2-135M-Q8_0.gguf \
  -chat -prompt="What is gravity?" -temperature=0.2 -max_tokens=100

# Interactive multi-turn chat
go run ./cmd/main/... -profile=smollm2_135m -gguf=SmolLM2-135M-Q8_0.gguf \
  -chat -interactive --temperature=0.2 --max_tokens=100
```

### GPT-2 Small 124M

```bash
# Convert weights from HuggingFace
pip install torch transformers
python3 scripts/convert_gpt2_weights.py --model gpt2

# Greedy
go run ./cmd/main/... -profile=small -weights=gpt2_124M.bin -format=bin \
  -prompt="Hello" -temperature=0

# Creative
go run ./cmd/main/... -profile=small -weights=gpt2_124M.bin -format=bin \
  -prompt="The meaning of life is" -temperature=0.8 -top_p=0.9 -seed=42
```

### GPT-2 Medium 355M

```bash
python3 scripts/convert_gpt2_weights.py --model gpt2-medium
go run ./cmd/main/... -profile=medium -weights=gpt2_medium.bin -format=bin \
  -prompt="Once upon a time" -temperature=0.8 -top_p=0.9 -seed=7
```

---

## Features

**Architectures**
- GPT-2 (GELU, LayerNorm, learned position embeddings, full MHA)
- Llama (RoPE, RMSNorm, SiLU/SwiGLU, GQA, shared KV cache)

**Weight loading**
- GGUF binary parser — reads F32/F16/Q8_0 tensors, dequantizes on load, extracts tokenizer
- Custom binary and JSON formats for GPT-2 weights

**Performance**
- Row-slice direct access (6-9× speedup over At/Set in hot loops)
- Goroutine parallelism on large matmuls and attention heads (500K-op threshold)
- KV-cached autoregressive generation (prefill once, decode one token at a time)

**Developer experience**
- Pure Go — no external dependencies
- Deterministic mode (`-seed` flag)
- EOS detection, weight tying, shape validation
- 84% test coverage across 12 packages
- Profiling via `LLM_BENCH_DEBUG=1`

---

## CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-profile` | string | `debug` | `debug` \| `small` \| `medium` \| `smollm2_135m` \| `smollm2_360m` \| `smollm2_1_7b` |
| `-weights` | string | `""` | GPT-2 weights file path |
| `-format` | string | `json` | `json` or `bin` (GPT-2) |
| `-gguf` | string | `""` | GGUF file path (overrides profile/weights/format) |
| `-prompt` | string | `"The"` | Input text |
| `-max_tokens` | int | 30 | Tokens to generate |
| `-temperature` | float | 1.0 | 0 = greedy argmax |
| `-top_p` | float | 1.0 | Nucleus threshold (1.0 = off) |
| `-repetition_penalty` | float | 1.0 | >1.0 penalizes repeated tokens (useful for GPT-2) |
| `-seed` | int | 0 | Random seed (0 = time-based) |
| `-chat` | bool | false | Wrap prompt in ChatML template (for instruct models) |
| `-interactive` | bool | false | Interactive multi-turn chat (requires --chat) |
| `-strict` | bool | false | Fail on missing/extra weights (GPT-2 only) |

---

## Architecture

```
prompt → Encode → Embed → [Block × N] → FinalNorm → OutputProj
  → logits → [RepPen] → Temp → TopP → Softmax → Sample → next token

Prefill: ForwardWithCache(ids, nil) → KVCache{Keys, Values}
Decode:  ForwardWithCache([token], cache) → updated KVCache
```

Two model implementations behind a shared interface:

| Component | GPT-2 | Llama |
|-----------|-------|-------|
| Norm | LayerNorm (shift + scale) | RMSNorm (scale only) |
| Activation | GELU | SiLU / SwiGLU |
| Position | Learned embeddings | RoPE (Rotary Position Embedding) |
| Attention | Full MHA (n heads × d_k) | GQA (n query / n kv heads) |
| FFN | Linear → GELU → Linear | Gate(SiLU) × Up → Down |
| Bias | Yes | No |

---

## Project Status

**Works:**
- GPT-2 Small (124M), GPT-2 Medium (355M)
- SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B (via GGUF, Q8_0)
- Greedy and nucleus sampling with temperature, top-p, seed, repetition penalty
- KV-cached streaming generation
- ChatML prompt formatting and interactive multi-turn chat

---

## References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [SmolLM2](https://huggingface.co/blog/smollm2)
- [RoPE](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding
- [GQA](https://arxiv.org/abs/2305.13245) — Grouped Query Attention
- [GGUF Spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [LLM from Scratch](https://sebastianraschka.com/llms-from-scratch) — book by Sebastian Raschka
