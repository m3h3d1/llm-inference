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
	cfg.Seed = 42

	m := model.NewGPTModel(cfg)
	tok := tokenizer.NewMock()

	prompt := "h"
	generated := Generate(cfg, m, tok, prompt, 5)

	if len(generated) == 0 {
		t.Fatal("Generated string is empty")
	}

	t.Logf("Generated: %s", generated)
}
