package llama

import (
	"math"
	"testing"

	"github.com/llm/config"
	"github.com/llm/tensor"
)

func testLlamaConfig() config.Config {
	return config.Config{
		VocabSize:         100,
		ContextLen:        16,
		EmbDim:            8,
		NHeads:            4,
		NKVHeads:          2,
		NLayers:           2,
		DFF:               16,
		DropRate:          0.0,
		RopeDim:           4,
		RopeTheta:         10000.0,
		RmsNormEps:        1e-5,
		QKVBias:           false,
		Temperature:       1.0,
		TopP:              1.0,
		RepetitionPenalty: 1.0,
		Seed:              0,
	}
}

func TestNewModel(t *testing.T) {
	cfg := testLlamaConfig()
	m := NewModel(cfg)
	if m == nil {
		t.Fatal("NewModel returned nil")
	}
	if len(m.Blocks) != cfg.NLayers {
		t.Errorf("expected %d blocks, got %d", cfg.NLayers, len(m.Blocks))
	}
	if m.OutputWeight != m.TokenEmbedding {
		t.Error("OutputWeight should be same pointer as TokenEmbedding (weight tying)")
	}
}

func TestLlamaForward(t *testing.T) {
	cfg := testLlamaConfig()
	m := NewModel(cfg)

	tokenIDs := []int{1, 2, 3}
	logits := m.Forward(tokenIDs)
	if logits == nil {
		t.Fatal("Forward returned nil")
	}
	dims := logits.Dimensions()
	if dims[0] != 1 || dims[1] != 3 || dims[2] != cfg.VocabSize {
		t.Errorf("logits shape: expected [1 3 %d], got %v", cfg.VocabSize, dims)
	}

	for b := 0; b < dims[0]; b++ {
		for s := 0; s < dims[1]; s++ {
			var sum float64
			for v := 0; v < dims[2]; v++ {
				sum += logits.At(b, s, v)
			}
			if math.IsNaN(sum) {
				t.Errorf("NaN in logits at [%d,%d]", b, s)
			}
		}
	}
}

func TestLlamaForwardWithCache(t *testing.T) {
	cfg := testLlamaConfig()
	m := NewModel(cfg)

	prefillIDs := []int{1, 2, 3}
	logits1, cache := m.ForwardWithCache(prefillIDs, nil)
	if logits1 == nil || cache == nil {
		t.Fatal("ForwardWithCache prefill returned nil")
	}
	if cache.SeqLen != 3 {
		t.Errorf("expected SeqLen 3, got %d", cache.SeqLen)
	}

	decodeIDs := []int{4}
	logits2, cache2 := m.ForwardWithCache(decodeIDs, cache)
	if logits2 == nil || cache2 == nil {
		t.Fatal("ForwardWithCache decode returned nil")
	}
	if cache2.SeqLen != 4 {
		t.Errorf("expected SeqLen 4, got %d", cache2.SeqLen)
	}

	dims1 := logits1.Dimensions()
	dims2 := logits2.Dimensions()
	if dims2[1] != 1 {
		t.Errorf("decode logits seq dim: expected 1, got %d", dims2[1])
	}
	if dims2[2] != dims1[2] {
		t.Errorf("vocab size mismatch: prefill %d, decode %d", dims1[2], dims2[2])
	}

	for b := 0; b < dims2[0]; b++ {
		for s := 0; s < dims2[1]; s++ {
			for v := 0; v < dims2[2]; v++ {
				if math.IsNaN(logits2.At(b, s, v)) {
					t.Errorf("NaN in decode logits at [%d,%d,%d]", b, s, v)
				}
			}
		}
	}
}

func TestLlamaParameters(t *testing.T) {
	cfg := testLlamaConfig()
	m := NewModel(cfg)
	params := m.Parameters()

	if len(params) == 0 {
		t.Fatal("Parameters returned empty map")
	}

	if _, ok := params["TokenEmbedding"]; !ok {
		t.Error("missing TokenEmbedding")
	}

	for i := 0; i < cfg.NLayers; i++ {
		prefix := "Blocks." + itoa(i) + "."
		expected := []string{
			prefix + "AttentionNorm.Weight",
			prefix + "FFNNorm.Weight",
			prefix + "Attention.Wq.Weight",
			prefix + "Attention.Wk.Weight",
			prefix + "Attention.Wv.Weight",
			prefix + "Attention.Wo.Weight",
			prefix + "FFN.Gate.Weight",
			prefix + "FFN.Up.Weight",
			prefix + "FFN.Down.Weight",
		}
		for _, key := range expected {
			if _, ok := params[key]; !ok {
				t.Errorf("missing %s", key)
			}
		}
	}
}

func TestLlamaSetParameter(t *testing.T) {
	cfg := testLlamaConfig()
	m := NewModel(cfg)

	// Case 1: Set token_embd.weight, verify change
	newEmb := tensor.NewTensor(cfg.VocabSize, 1, cfg.EmbDim)
	newEmb.Set(0, 0, 0, 42.0)
	m.SetParameter("token_embd.weight", newEmb)
	if m.Parameters()["TokenEmbedding"].At(0, 0, 0) != 42.0 {
		t.Error("SetParameter token_embd.weight did not update TokenEmbedding")
	}
	if m.OutputWeight != m.TokenEmbedding {
		t.Error("OutputWeight should still be TokenEmbedding after SetParameter")
	}

	// Case 2: Set block parameter
	hd := cfg.EmbDim / cfg.NHeads
	qWeight := tensor.NewTensor(1, cfg.NHeads*hd, cfg.EmbDim)
	qWeight.Set(0, 0, 0, 99.0)
	m.SetParameter("blk.0.attn_q.weight", qWeight)
	if m.Parameters()["Blocks.0.Attention.Wq.Weight"].At(0, 0, 0) != 99.0 {
		t.Error("SetParameter blk.0.attn_q.weight did not update")
	}

	// Case 3: Unknown component should be silent no-op
	pre := m.Parameters()["Blocks.0.Attention.Wq.Weight"].At(0, 0, 0)
	m.SetParameter("blk.0.nonexistent", qWeight)
	if m.Parameters()["Blocks.0.Attention.Wq.Weight"].At(0, 0, 0) != pre {
		t.Error("Unknown component should not change parameters")
	}

	// Case 4: Unknown layer index should be silent no-op
	pre4 := m.Parameters()["Blocks.0.Attention.Wq.Weight"].At(0, 0, 0)
	m.SetParameter("blk.99.attn_q.weight", qWeight)
	if m.Parameters()["Blocks.0.Attention.Wq.Weight"].At(0, 0, 0) != pre4 {
		t.Error("Unknown layer should not change parameters")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [8]byte
	n := i
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[pos:])
}
