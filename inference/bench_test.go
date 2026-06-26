//go:build bench

package inference

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm/gguf"
	"github.com/llm/model/llama"
	"github.com/llm/tokenizer"
	"github.com/llm/weights"
)

func TestSpeed(t *testing.T) {
	matches, err := filepath.Glob("../models/*.gguf")
	if err != nil || len(matches) == 0 {
		t.Fatal("No GGUF models found in ../models/")
	}

	for _, path := range matches {
		t.Run(filepath.Base(path), func(t *testing.T) {
			speedTest(t, path)
		})
	}
}

func speedTest(t *testing.T, path string) {
	t0 := time.Now()

	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	cfg, err := weights.LoadConfigFromGGUF(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	cfg.Temperature = 0

	m := llama.NewModel(cfg)
	if err := weights.LoadWeightsFromGGUF(m, f); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}

	tok, err := tokenizer.NewFromGGUF(f)
	if err != nil {
		t.Fatalf("Tokenizer: %v", err)
	}

	loadTime := time.Since(t0)

	shortIDs := tok.Encode("Hello world")
	ntokens := len(shortIDs)

	longIDs := make([]int, 0, 128)
	for len(longIDs) < 128 {
		longIDs = append(longIDs, shortIDs...)
	}
	longIDs = longIDs[:128]

	nextID := shortIDs[len(shortIDs)-1]

	type run struct {
		name  string
		iters int
		fn    func(int)
	}
	runs := []run{
		{"prefill-4t", 3, func(i int) {
			_, cache := m.ForwardWithCache(shortIDs, nil)
			_ = cache
		}},
		{"prefill-128t", 1, func(i int) {
			_, cache := m.ForwardWithCache(longIDs, nil)
			_ = cache
		}},
		{"decode-1t", 5, func(i int) {
			_, cache := m.ForwardWithCache(shortIDs, nil)
			_, cache = m.ForwardWithCache([]int{nextID}, cache)
			_ = cache
		}},
	}

	t.Logf("Model loaded in %v", loadTime)
	t.Logf("Config: layers=%d emb=%d heads=%d kv=%d dff=%d",
		cfg.NLayers, cfg.EmbDim, cfg.NHeads, cfg.NKVHeads, cfg.DFF)
	t.Logf("Prompt tokens: %d (short), 128 (long)", ntokens)

	// Warmup with short prompt only (long warmup is too expensive)
	m.ForwardWithCache(shortIDs, nil)

	for _, prefix := range runs {
		start := time.Now()
		for i := 0; i < prefix.iters; i++ {
			prefix.fn(i)
		}
		elapsed := time.Since(start)
		avgNs := float64(elapsed.Nanoseconds()) / float64(prefix.iters)
		var metric string
		switch prefix.name {
		case "prefill-4t":
			metric = fmt.Sprintf("%.1f tok/s", float64(ntokens)/elapsed.Seconds()*float64(prefix.iters))
		case "prefill-128t":
			metric = fmt.Sprintf("%.1f tok/s", 128.0/elapsed.Seconds()*float64(prefix.iters))
		case "decode-1t":
			metric = fmt.Sprintf("%.1f tok/s", 1.0/(avgNs/1e9))
		}
		t.Logf("%s: %d iters, avg %.0f ms/op, %s",
			prefix.name, prefix.iters, avgNs/1e6, metric)
	}

	logits, _ := m.ForwardWithCache(shortIDs, nil)
	lastPos := len(shortIDs) - 1
	var top5IDs [5]int
	var top5Vals [5]float64
	for v := 0; v < cfg.VocabSize; v++ {
		val := logits.At(0, lastPos, v)
		for j := 0; j < 5; j++ {
			if val > top5Vals[j] {
				for k := 4; k > j; k-- {
					top5Vals[k] = top5Vals[k-1]
					top5IDs[k] = top5IDs[k-1]
				}
				top5Vals[j] = val
				top5IDs[j] = v
				break
			}
		}
	}
	decoded := tok.Decode(top5IDs[:])
	t.Logf("Top-5 tokens: IDs=%v logits=%v decoded=%q", top5IDs, roundFloats(top5Vals[:]), decoded)

	var sum float64
	for v := 0; v < cfg.VocabSize; v++ {
		sum += math.Exp(logits.At(0, lastPos, v) - top5Vals[0])
	}
	entropy := 0.0
	for v := 0; v < cfg.VocabSize; v++ {
		p := math.Exp(logits.At(0, lastPos, v)-top5Vals[0]) / sum
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	_ = entropy
	t.Logf("Model size: %d MB", fileSize(path)/1024/1024)
}

func roundFloats(vals []float64) []float64 {
	out := make([]float64, len(vals))
	for i, v := range vals {
		out[i] = math.Round(v*100) / 100
	}
	return out
}

func fileSize(path string) int64 {
	fi, _ := os.Stat(path)
	return fi.Size()
}
