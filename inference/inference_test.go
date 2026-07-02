package inference

import (
	"testing"

	"github.com/llm/config"
	"github.com/llm/model"
	gpt2 "github.com/llm/model/gpt2"
	"github.com/llm/tensor"
	"github.com/llm/tokenizer"
)

type mockFixedModel struct {
	vocabSize int
	// logitFn returns the logit value for vocabulary index v.
	// Called once per generation step for the last token position.
	logitFn func(v int) float64
}

func (m *mockFixedModel) Forward(tokenIDs []int) *tensor.Tensor {
	seq := len(tokenIDs)
	logits := tensor.NewTensor(1, seq, m.vocabSize)
	for v := 0; v < m.vocabSize; v++ {
		logits.Set(0, seq-1, v, m.logitFn(v))
	}
	return logits
}

func (m *mockFixedModel) ForwardWithCache(tokenIDs []int, pastCache *model.KVCache) (*tensor.Tensor, *model.KVCache) {
	startPos := 0
	if pastCache != nil {
		startPos = pastCache.SeqLen
	}
	seq := len(tokenIDs)
	logits := tensor.NewTensor(1, seq, m.vocabSize)
	for v := 0; v < m.vocabSize; v++ {
		logits.Set(0, seq-1, v, m.logitFn(v))
	}
	cache := &model.KVCache{SeqLen: startPos + seq}
	return logits, cache
}

func TestArgmax(t *testing.T) {
	if got := Argmax([]float64{1, 5, 3}); got != 1 {
		t.Errorf("basic: expected 1, got %d", got)
	}
	if got := Argmax([]float64{5, 5, 3}); got != 0 {
		t.Errorf("ties: expected 0 (first), got %d", got)
	}
	if got := Argmax([]float64{42}); got != 0 {
		t.Errorf("single element: expected 0, got %d", got)
	}
	if got := Argmax([]float64{-10, -5, -1, -3}); got != 2 {
		t.Errorf("negatives: expected 2, got %d", got)
	}
}

func TestGenerate(t *testing.T) {
	tok := tokenizer.NewMock()
	m := &mockFixedModel{
		vocabSize: 10,
		logitFn:   func(v int) float64 { return float64(101 - v) },
	}
	cfg := config.GPT2_124M
	cfg.VocabSize = 10
	cfg.Temperature = 0.0

	result := Generate(cfg, m, tok, "he", 2)
	if result != "hehh" {
		t.Errorf("got %q, want %q", result, "hehh")
	}
}

func TestGenerateStreaming(t *testing.T) {
	cfg := config.GPT2_124M
	cfg.NLayers = 2
	cfg.NHeads = 4
	cfg.EmbDim = 16
	cfg.VocabSize = 100
	cfg.Seed = 42

	m := gpt2.NewModel(cfg)
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

func TestGenerateBasic(t *testing.T) {
	cfg := config.GPT2_124M
	cfg.NLayers = 1
	cfg.NHeads = 4
	cfg.EmbDim = 8
	cfg.VocabSize = 10
	cfg.Temperature = 0.0

	m := gpt2.NewModel(cfg)
	tok := tokenizer.NewMock()

	result := Generate(cfg, m, tok, "he", 5)
	if len(result) == 0 {
		t.Fatal("Result is empty")
	}
	// With Temperature=0 (argmax), result includes prompt plus generated tokens
	if len(result) <= 2 {
		t.Errorf("expected more than prompt length, got %q", result)
	}
}

func TestGenerateRepetitionPenalty(t *testing.T) {
	tok := tokenizer.NewMock()
	// token 0="h" (100), token 1="e" (99).
	// Prompt "he" → [4]. With maxNewTokens=2:
	//   Step 1: argmax picks token 0 (100). ids=[4,0].
	//   Step 2: RepetitionPenalty(2.0): logit[0]=100→50, logit[4]=0→0.
	//           Now token 1=99 > token 0=50. argmax picks 1. ids=[4,0,1].
	// Result: Decode([4,0,1]) = "he" + "h" + "e" = "hehe".
	m := &mockFixedModel{
		vocabSize: 10,
		logitFn:   func(v int) float64 { return float64(100 - v) },
	}
	cfg := config.GPT2_124M
	cfg.VocabSize = 10
	cfg.Temperature = 0.0
	cfg.RepetitionPenalty = 2.0

	result := Generate(cfg, m, tok, "he", 2)
	if result != "hehe" {
		t.Errorf("with penalty: got %q, want %q", result, "hehe")
	}

	// Without penalty, token 0 keeps winning: ids=[4,0,0] → "hehh".
	cfg.RepetitionPenalty = 1.0
	result2 := Generate(cfg, m, tok, "he", 2)
	if result2 != "hehh" {
		t.Errorf("without penalty: got %q, want %q", result2, "hehh")
	}
}

func TestGenerateTemperature(t *testing.T) {
	tok := tokenizer.NewMock()
	// Two almost-tied logits: token 0=101, token 1=100.
	// With Temperature=0.001 (near-zero but >0, exercises sampling path):
	//   After scaling: [101000, 100000]. Softmax extremely peaked at token 0.
	//   Sample picks token 0 deterministically.
	// With Temperature=0 (argmax path): also picks token 0.
	// The key is this exercises the temperature>0 code path with deterministic result.
	m := &mockFixedModel{
		vocabSize: 10,
		logitFn:   func(v int) float64 { return float64(101 - v) },
	}
	cfg := config.GPT2_124M
	cfg.VocabSize = 10
	cfg.Seed = 42
	cfg.Temperature = 0.001

	result := Generate(cfg, m, tok, "he", 2)
	if result != "hehh" {
		t.Errorf("temperature=0.001: got %q, want %q", result, "hehh")
	}

	// Higher temperature changes distribution and may produce different result.
	cfg.Temperature = 10.0
	result2 := Generate(cfg, m, tok, "he", 2)
	if result2 == "hehh" {
		t.Logf("temperature=10: got %q (same as argmax — possible but unlikely)", result2)
	} else {
		// Different result confirms sampling path was taken.
		t.Logf("temperature=10: got %q (different from argmax path)", result2)
	}
	if len(result2) <= 2 {
		t.Errorf("temperature=10 result too short: %q", result2)
	}
}

func TestGenerateEOSWithMock(t *testing.T) {
	m := &mockFixedModel{
		vocabSize: 10,
		logitFn:   func(v int) float64 { return float64(v) },
	}
	// Override so token 5 is argmax
	m.logitFn = func(v int) float64 {
		if v == 5 {
			return 1000.0
		}
		return float64(v)
	}

	tok := tokenizer.NewMock()

	cfg := config.GPT2_124M
	cfg.VocabSize = 10
	cfg.EOSTokenID = 5
	cfg.Temperature = 0.0

	// Prompt "he" encodes to token [4]. Mock always returns token 5 as argmax.
	// EOSTokenID=5 fires before appending, so result = prompt only = "he".
	result := Generate(cfg, m, tok, "he", 10)
	if result != "he" {
		t.Errorf("EOS should stop immediately: got %q, want %q", result, "he")
	}
}
