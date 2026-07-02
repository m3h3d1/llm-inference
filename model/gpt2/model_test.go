package gpt2

import (
	"math"
	"testing"

	"github.com/llm/config"
	"github.com/llm/tokenizer"
	"github.com/llm/tensor"
)

func TestEmbeddings(t *testing.T) {
	vocabSize := 100
	contextLen := 10
	embDim := 8

	emb := NewEmbeddings(vocabSize, contextLen, embDim)

	tokenIDs := []int{1, 2, 3}
	result := emb.Forward(tokenIDs, 0)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 3 || dims[2] != embDim {
		t.Errorf("Expected shape (1, 3, %d), got %v", embDim, dims)
	}
	for _, v := range result.Data {
		if math.IsNaN(v) {
			t.Error("NaN in embeddings output")
			break
		}
	}
}

func TestFFN(t *testing.T) {
	dModel := 8
	dFF := 16
	ffn := NewFeedForward(dModel, dFF)

	input := tensor.NewTensor(1, 2, dModel)
	result := ffn.Forward(input)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 2 || dims[2] != dModel {
		t.Errorf("Expected shape (1, 2, %d), got %v", dModel, dims)
	}
	for _, v := range result.Data {
		if math.IsNaN(v) {
			t.Error("NaN in FFN output")
			break
		}
	}
}

func TestTransformerBlock(t *testing.T) {
	dModel := 8
	block := NewTransformerBlock(dModel, 2, 0.0)

	input := tensor.NewTensor(1, 2, dModel)
	result := block.Forward(input, nil)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 2 || dims[2] != dModel {
		t.Errorf("Expected shape (1, 2, %d), got %v", dModel, dims)
	}
	for _, v := range result.Data {
		if math.IsNaN(v) {
			t.Error("NaN in transformer block output")
			break
		}
	}
}

func TestGPTModel(t *testing.T) {
	cfg := config.GPT2_124M
	cfg.NLayers = 2
	cfg.NHeads = 4
	cfg.EmbDim = 16
	cfg.VocabSize = 100

	model := NewModel(cfg)

	tokenIDs := []int{1, 2, 3, 4, 5}
	logits := model.Forward(tokenIDs)

	if logits == nil {
		t.Fatal("Logits should not be nil")
	}

	dims := logits.Dimensions()
	if dims[0] != 1 || dims[1] != 5 || dims[2] != cfg.VocabSize {
		t.Errorf("Expected shape (1, 5, %d), got %v", cfg.VocabSize, dims)
	}
	for _, v := range logits.Data {
		if math.IsNaN(v) {
			t.Error("NaN in model forward logits")
			break
		}
	}
}

func TestForwardWithCache(t *testing.T) {
	cfg := config.GPT2_124M
	cfg.NLayers = 2
	cfg.NHeads = 4
	cfg.EmbDim = 16
	cfg.VocabSize = 100

	model := NewModel(cfg)

	// Prefill: 3 tokens
	tokenIDs := []int{1, 2, 3}
	logits1, cache := model.ForwardWithCache(tokenIDs, nil)
	if logits1 == nil || cache == nil {
		t.Fatal("ForwardWithCache (prefill) returned nil")
	}
	if cache.SeqLen != 3 {
		t.Errorf("Expected cache.SeqLen=3, got %d", cache.SeqLen)
	}
	if len(cache.Keys) != cfg.NLayers || len(cache.Values) != cfg.NLayers {
		t.Errorf("Expected %d cache entries, got Keys=%d Values=%d", cfg.NLayers, len(cache.Keys), len(cache.Values))
	}
	dims := logits1.Dimensions()
	if dims[0] != 1 || dims[1] != 3 || dims[2] != cfg.VocabSize {
		t.Errorf("Prefill logits shape (1,3,%d), got %v", cfg.VocabSize, dims)
	}

	// Decode: 1 new token with cache
	tokenIDs2 := []int{4}
	logits2, cache2 := model.ForwardWithCache(tokenIDs2, cache)
	if logits2 == nil || cache2 == nil {
		t.Fatal("ForwardWithCache (decode) returned nil")
	}
	if cache2.SeqLen != 4 {
		t.Errorf("Expected cache.SeqLen=4, got %d", cache2.SeqLen)
	}
	dims2 := logits2.Dimensions()
	if dims2[0] != 1 || dims2[1] != 1 || dims2[2] != cfg.VocabSize {
		t.Errorf("Decode logits shape (1,1,%d), got %v", cfg.VocabSize, dims2)
	}

	// Forward without cache for the full 4-token sequence should match
	// the prefill logits for the first 3 positions
	fullLogits := model.Forward([]int{1, 2, 3, 4})
	for s := 0; s < 3; s++ {
		for v := 0; v < cfg.VocabSize; v++ {
			if logits1.At(0, s, v) != fullLogits.At(0, s, v) {
				t.Errorf("Prefill logit mismatch at seq=%d, vocab=%d: %f vs %f", s, v, logits1.At(0, s, v), fullLogits.At(0, s, v))
				return
			}
		}
	}
}

func TestIntegrationWithTokenizer(t *testing.T) {
	cfg := config.GPT2_124M
	cfg.NLayers = 2
	cfg.NHeads = 4
	cfg.EmbDim = 16

	tok := tokenizer.NewMock()
	model := NewModel(cfg)

	prompt := "hello"
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		t.Fatal("Tokenizer returned empty IDs")
	}

	logits := model.Forward(ids)
	if logits == nil {
		t.Fatal("Logits should not be nil after forward pass")
	}

	dims := logits.Dimensions()
	expectedSeq := len(ids)
	if dims[1] != expectedSeq || dims[2] != cfg.VocabSize {
		t.Errorf("Expected shape (1, %d, %d), got %v", expectedSeq, cfg.VocabSize, dims)
	}
	for _, v := range logits.Data {
		if math.IsNaN(v) {
			t.Error("NaN in integration test logits")
			break
		}
	}
}
