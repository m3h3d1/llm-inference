package inference

import (
	"testing"

	"github.com/llm/config"
	"github.com/llm/model"
	"github.com/llm/tokenizer"
)

func TestGenerate(t *testing.T) {
	cfg := config.DefaultConfig
	cfg.NLayers = 2
	cfg.EmbDim = 16
	cfg.VocabSize = 100

	// Use a simple mock for the test
	m := model.NewGPTModel(cfg)
	tok := tokenizer.NewMock()

	prompt := "h" // "h" maps to ID 0 in our mock
	generated := Generate(cfg, m, tok, prompt, 5)

	// Verify it's not empty
	if len(generated) == 0 {
		t.Fatal("Generated string is empty")
	}

	t.Logf("Generated: %s", generated)
}