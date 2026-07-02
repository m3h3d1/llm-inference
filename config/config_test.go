package config

import "testing"

func TestGPT2_124M(t *testing.T) {
	if GPT2_124M.EmbDim%GPT2_124M.NHeads != 0 {
		t.Errorf("GPT2_124M: EmbDim (%d) not divisible by NHeads (%d)", GPT2_124M.EmbDim, GPT2_124M.NHeads)
	}
	if GPT2_124M.VocabSize <= 0 {
		t.Errorf("GPT2_124M: VocabSize (%d) must be positive", GPT2_124M.VocabSize)
	}
	if GPT2_124M.NLayers <= 0 {
		t.Errorf("GPT2_124M: NLayers (%d) must be positive", GPT2_124M.NLayers)
	}
}

func TestGPT2_355M(t *testing.T) {
	if GPT2_355M.EmbDim%GPT2_355M.NHeads != 0 {
		t.Errorf("GPT2_355M: EmbDim (%d) not divisible by NHeads (%d)", GPT2_355M.EmbDim, GPT2_355M.NHeads)
	}
	if GPT2_355M.VocabSize <= 0 {
		t.Errorf("GPT2_355M: VocabSize (%d) must be positive", GPT2_355M.VocabSize)
	}
}

func TestSmolLM2_135M(t *testing.T) {
	if SmolLM2_135M.EmbDim%SmolLM2_135M.NHeads != 0 {
		t.Errorf("SmolLM2_135M: EmbDim (%d) not divisible by NHeads (%d)", SmolLM2_135M.EmbDim, SmolLM2_135M.NHeads)
	}
	headDim := SmolLM2_135M.EmbDim / SmolLM2_135M.NHeads
	if SmolLM2_135M.RopeDim > headDim {
		t.Errorf("SmolLM2_135M: RopeDim (%d) exceeds headDim (%d)", SmolLM2_135M.RopeDim, headDim)
	}
	if SmolLM2_135M.VocabSize <= 0 {
		t.Errorf("SmolLM2_135M: VocabSize (%d) must be positive", SmolLM2_135M.VocabSize)
	}
	if SmolLM2_135M.NKVHeads <= 0 || SmolLM2_135M.NKVHeads > SmolLM2_135M.NHeads {
		t.Errorf("SmolLM2_135M: NKVHeads (%d) out of range", SmolLM2_135M.NKVHeads)
	}
}

func TestSmolLM2_360M(t *testing.T) {
	if SmolLM2_360M.EmbDim%SmolLM2_360M.NHeads != 0 {
		t.Errorf("SmolLM2_360M: EmbDim (%d) not divisible by NHeads (%d)", SmolLM2_360M.EmbDim, SmolLM2_360M.NHeads)
	}
	headDim := SmolLM2_360M.EmbDim / SmolLM2_360M.NHeads
	if SmolLM2_360M.RopeDim > headDim {
		t.Errorf("SmolLM2_360M: RopeDim (%d) exceeds headDim (%d)", SmolLM2_360M.RopeDim, headDim)
	}
	if SmolLM2_360M.NKVHeads <= 0 || SmolLM2_360M.NKVHeads > SmolLM2_360M.NHeads {
		t.Errorf("SmolLM2_360M: NKVHeads (%d) out of range", SmolLM2_360M.NKVHeads)
	}
}

func TestSmolLM2_1_7B(t *testing.T) {
	if SmolLM2_1_7B.EmbDim%SmolLM2_1_7B.NHeads != 0 {
		t.Errorf("SmolLM2_1_7B: EmbDim (%d) not divisible by NHeads (%d)", SmolLM2_1_7B.EmbDim, SmolLM2_1_7B.NHeads)
	}
	headDim := SmolLM2_1_7B.EmbDim / SmolLM2_1_7B.NHeads
	if SmolLM2_1_7B.RopeDim > headDim {
		t.Errorf("SmolLM2_1_7B: RopeDim (%d) exceeds headDim (%d)", SmolLM2_1_7B.RopeDim, headDim)
	}
	if SmolLM2_1_7B.NKVHeads <= 0 || SmolLM2_1_7B.NKVHeads > SmolLM2_1_7B.NHeads {
		t.Errorf("SmolLM2_1_7B: NKVHeads (%d) out of range", SmolLM2_1_7B.NKVHeads)
	}
}
