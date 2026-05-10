package model

import (
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
	result := emb.Forward(tokenIDs)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 3 || dims[2] != embDim {
		t.Errorf("Expected shape (1, 3, %d), got %v", embDim, dims)
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
}

func TestTransformerBlock(t *testing.T) {
	dModel := 8
	block := NewTransformerBlock(dModel)

	input := tensor.NewTensor(1, 2, dModel)
	result := block.Forward(input, nil)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 2 || dims[2] != dModel {
		t.Errorf("Expected shape (1, 2, %d), got %v", dModel, dims)
	}
}

func TestGPTModel(t *testing.T) {
	cfg := config.DefaultConfig
	// Use smaller config for testing
	cfg.NLayers = 2
	cfg.EmbDim = 16
	cfg.VocabSize = 100

	model := NewGPTModel(cfg)

	tokenIDs := []int{1, 2, 3, 4, 5}
	logits := model.Forward(tokenIDs)

	if logits == nil {
		t.Fatal("Logits should not be nil")
	}

	dims := logits.Dimensions()
	if dims[0] != 1 || dims[1] != 5 || dims[2] != cfg.VocabSize {
		t.Errorf("Expected shape (1, 5, %d), got %v", cfg.VocabSize, dims)
	}
}

func TestIntegrationWithTokenizer(t *testing.T) {
	// Only run if we have assets (skip if running in isolation)
	cfg := config.DefaultConfig
	cfg.NLayers = 2
	cfg.EmbDim = 16

	tok := tokenizer.NewMock()
	model := NewGPTModel(cfg)

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
}
