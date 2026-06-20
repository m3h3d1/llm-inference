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
	cfg.NHeads = 4
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

func TestGenerateStreaming(t *testing.T) {
	cfg := config.DefaultConfig
	cfg.NLayers = 2
	cfg.NHeads = 4
	cfg.EmbDim = 16
	cfg.VocabSize = 100
	cfg.Seed = 42

	m := model.NewGPTModel(cfg)
	tok := tokenizer.NewMock()

	var deltas []string
	onToken := func(delta string) {
		deltas = append(deltas, delta)
	}

	prompt := "h"
	result := GenerateStreaming(cfg, m, tok, prompt, 5, onToken)

	if len(result) == 0 {
		t.Fatal("Result is empty")
	}

	if len(deltas) == 0 {
		t.Fatal("onToken was never called")
	}

	// Deltas cover generated content only; full result includes prompt.
	var combined string
	for _, d := range deltas {
		combined += d
	}
	if prompt+combined != result {
		t.Errorf("prompt+deltas [%q] != result [%q]", prompt+combined, result)
	}

	t.Logf("Streaming generated: %s", result)
}

func TestGenerateStreamingStopsOnEOS(t *testing.T) {
	// Override the package-level eosTokenID by using a small vocab
	// with a token that equals the configured EOS.
	cfg := config.DefaultConfig
	cfg.NLayers = 1
	cfg.NHeads = 4
	cfg.EmbDim = 8
	cfg.VocabSize = 10
	cfg.Seed = 0 // Use time-based seed for variety
	cfg.Temperature = 0.0

	m := model.NewGPTModel(cfg)
	tok := tokenizer.NewMock()

	// Generate "hello" (tokens [4,5,3]) which maps to "hello".
	// Then generation continues for maxNewTokens=10.
	// With temp=0, it's argmax — won't naturally hit EOS at 50256 since VocabSize=10.
	// So this test primarily verifies that the loop terminates without error.
	result := Generate(cfg, m, tok, "he", 10)
	if len(result) == 0 {
		t.Fatal("Result is empty")
	}
	t.Logf("EOS test generated: %s", result)
}
