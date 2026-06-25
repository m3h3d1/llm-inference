package config

import "testing"

func TestDefaultConfig(t *testing.T) {
	if DefaultConfig.EmbDim%DefaultConfig.NHeads != 0 {
		t.Errorf("Default: EmbDim (%d) not divisible by NHeads (%d)", DefaultConfig.EmbDim, DefaultConfig.NHeads)
	}
	if DefaultConfig.VocabSize <= 0 {
		t.Errorf("Default: VocabSize (%d) must be positive", DefaultConfig.VocabSize)
	}
	if DefaultConfig.NLayers <= 0 {
		t.Errorf("Default: NLayers (%d) must be positive", DefaultConfig.NLayers)
	}
}

func TestGPT2Medium(t *testing.T) {
	if GPT2Medium.EmbDim%GPT2Medium.NHeads != 0 {
		t.Errorf("GPT2Medium: EmbDim (%d) not divisible by NHeads (%d)", GPT2Medium.EmbDim, GPT2Medium.NHeads)
	}
	if GPT2Medium.VocabSize <= 0 {
		t.Errorf("GPT2Medium: VocabSize (%d) must be positive", GPT2Medium.VocabSize)
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
