//go:build bench

package inference

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm/gguf"
	"github.com/llm/internal/benchprof"
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
	debug := benchprof.Enabled()

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

	t.Logf("Model loaded in %v", loadTime)
	t.Logf("Config: layers=%d emb=%d heads=%d kv=%d dff=%d",
		cfg.NLayers, cfg.EmbDim, cfg.NHeads, cfg.NKVHeads, cfg.DFF)
	t.Logf("Prompt tokens: %d (short), 128 (long)", ntokens)

	// Warmup with short prompt only (long warmup is too expensive)
	m.ForwardWithCache(shortIDs, nil)

	if debug {
		benchprof.ResetAll()
	}

	type prefillRun struct {
		name     string
		ids      []int
		iters    int
		tokPerOp int
	}
	for _, pr := range []prefillRun{
		{"prefill-4t", shortIDs, 3, ntokens},
		{"prefill-128t", longIDs, 1, 128},
	} {
		if debug {
			benchprof.ResetAll()
		}
		start := time.Now()
		for i := 0; i < pr.iters; i++ {
			m.ForwardWithCache(pr.ids, nil)
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(pr.iters) / 1e6
		tokPerSec := float64(pr.tokPerOp) / elapsed.Seconds() * float64(pr.iters)
		t.Logf("%s: %d iters, avg %.0f ms/op, %.1f tok/s", pr.name, pr.iters, avgMs, tokPerSec)
		logProfile(t, debug)
	}

	// decode-1t: prefill shortIDs, then decode one token with cache
	if debug {
		benchprof.ResetAll()
	}
	var prefillTotal, decodeTotal time.Duration
	start := time.Now()
	for i := 0; i < 5; i++ {
		tp := time.Now()
		_, cache := m.ForwardWithCache(shortIDs, nil)
		prefillTotal += time.Since(tp)

		td := time.Now()
		_, cache = m.ForwardWithCache([]int{nextID}, cache)
		decodeTotal += time.Since(td)
		_ = cache
	}
	elapsed := time.Since(start)
	avgMs := float64(elapsed.Nanoseconds()) / 5 / 1e6
	t.Logf("decode-1t: 5 iters, avg %.0f ms/op, %.1f tok/s", avgMs, 1.0/(avgMs/1e3))
	t.Logf("  prefill: avg %v  decode: avg %v", prefillTotal/5, decodeTotal/5)
	logProfile(t, debug)

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
	t.Logf("Model size: %d MB", fileSize(path)/1024/1024)
}

func logProfile(t *testing.T, debug bool) {
	if !debug {
		return
	}
	s := benchprof.Summary()
	if s != "" {
		t.Logf("profile:\n%s", s)
	}
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
