package weights

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/llm/config"
	"github.com/llm/model"
	"github.com/llm/tensor"
)

func testConfig() config.Config {
	return config.Config{
		VocabSize:  100,
		ContextLen: 10,
		EmbDim:     8,
		NHeads:     2,
		NLayers:    2,
		DropRate:   0.0,
		QKVBias:    false,
	}
}

func paramsEqual(t *testing.T, got, want map[string]*tensor.Tensor) bool {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("param count mismatch: %d vs %d", len(got), len(want))
		return false
	}
	for key, gt := range got {
		wt, ok := want[key]
		if !ok {
			t.Errorf("missing key %q in want", key)
			return false
		}
		gd, wd := gt.Dimensions(), wt.Dimensions()
		if gd != wd {
			t.Errorf("shape mismatch for %q: %v vs %v", key, gd, wd)
			return false
		}
		for b := 0; b < gd[0]; b++ {
			for s := 0; s < gd[1]; s++ {
				for e := 0; e < gd[2]; e++ {
					gv, wv := gt.At(b, s, e), wt.At(b, s, e)
					if gv != wv {
						t.Errorf("%s[%d,%d,%d]: %f vs %f", key, b, s, e, gv, wv)
						return false
					}
				}
			}
		}
	}
	return true
}

func paramsClose(t *testing.T, got, want map[string]*tensor.Tensor, eps float64) bool {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("param count mismatch: %d vs %d", len(got), len(want))
		return false
	}
	for key, gt := range got {
		wt, ok := want[key]
		if !ok {
			t.Errorf("missing key %q in want", key)
			return false
		}
		gd, wd := gt.Dimensions(), wt.Dimensions()
		if gd != wd {
			t.Errorf("shape mismatch for %q: %v vs %v", key, gd, wd)
			return false
		}
		for b := 0; b < gd[0]; b++ {
			for s := 0; s < gd[1]; s++ {
				for e := 0; e < gd[2]; e++ {
					diff := math.Abs(gt.At(b, s, e) - wt.At(b, s, e))
					if diff > eps {
						t.Errorf("%s[%d,%d,%d]: %f vs %f (diff %f)", key, b, s, e, gt.At(b, s, e), wt.At(b, s, e), diff)
						return false
					}
				}
			}
		}
	}
	return true
}

func TestSaveLoadJSON(t *testing.T) {
	cfg := testConfig()
	original := model.NewGPTModel(cfg)

	path := filepath.Join(t.TempDir(), "weights.json")
	if err := SaveWeightsJSON(original, path); err != nil {
		t.Fatalf("SaveWeightsJSON: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat temp file: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved JSON weights file is empty")
	}

	loaded := model.NewGPTModel(cfg)
	if err := LoadWeightsJSON(loaded, path, true); err != nil {
		t.Fatalf("LoadWeightsJSON: %v", err)
	}

	if !paramsEqual(t, loaded.Parameters(), original.Parameters()) {
		t.Fatal("JSON round-trip: params differ")
	}
}

func TestSaveLoadBinary(t *testing.T) {
	cfg := testConfig()
	original := model.NewGPTModel(cfg)

	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := SaveWeightsBinary(original, path); err != nil {
		t.Fatalf("SaveWeightsBinary: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat temp file: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved binary weights file is empty")
	}

	loaded := model.NewGPTModel(cfg)
	if err := LoadWeightsBinary(loaded, path, true); err != nil {
		t.Fatalf("LoadWeightsBinary: %v", err)
	}

	if !paramsClose(t, loaded.Parameters(), original.Parameters(), 1e-6) {
		t.Fatal("Binary round-trip: params differ beyond float32 precision")
	}
}

func TestJSONStrictMissingKey(t *testing.T) {
	cfgSmall := testConfig()
	cfgSmall.NLayers = 1

	cfgLarge := testConfig()

	small := model.NewGPTModel(cfgSmall)
	path := filepath.Join(t.TempDir(), "weights_partial.json")
	if err := SaveWeightsJSON(small, path); err != nil {
		t.Fatalf("SaveWeightsJSON: %v", err)
	}

	large := model.NewGPTModel(cfgLarge)
	err := LoadWeightsJSON(large, path, true)
	if err == nil {
		t.Fatal("expected error for missing keys in strict mode")
	}
}

func TestJSONNonStrictMissingKey(t *testing.T) {
	cfgSmall := testConfig()
	cfgSmall.NLayers = 1

	cfgLarge := testConfig()

	small := model.NewGPTModel(cfgSmall)
	path := filepath.Join(t.TempDir(), "weights_partial.json")
	if err := SaveWeightsJSON(small, path); err != nil {
		t.Fatalf("SaveWeightsJSON: %v", err)
	}

	large := model.NewGPTModel(cfgLarge)
	if err := LoadWeightsJSON(large, path, false); err != nil {
		t.Fatalf("non-strict load should not error: %v", err)
	}
}

func TestBinaryInvalidMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	if err := os.WriteFile(path, []byte{0x00, 0x00, 0x00, 0x00}, 0644); err != nil {
		t.Fatalf("write bad file: %v", err)
	}

	cfg := testConfig()
	m := model.NewGPTModel(cfg)
	err := LoadWeightsBinary(m, path, true)
	if err == nil {
		t.Fatal("expected error for invalid magic number")
	}
}

func TestBinaryInvalidVersion(t *testing.T) {
	cfg := testConfig()
	original := model.NewGPTModel(cfg)

	path := filepath.Join(t.TempDir(), "bad_version.bin")
	if err := SaveWeightsBinary(original, path); err != nil {
		t.Fatalf("SaveWeightsBinary: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	data[4] = 0xFF
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("write corrupted file: %v", err)
	}

	m := model.NewGPTModel(cfg)
	err = LoadWeightsBinary(m, path, true)
	if err == nil {
		t.Fatal("expected error for invalid version")
	}
}

func TestJSONStrictExtraKey(t *testing.T) {
	cfgSmall := testConfig()
	cfgSmall.NLayers = 1

	cfgLarge := testConfig()

	large := model.NewGPTModel(cfgLarge)
	path := filepath.Join(t.TempDir(), "weights_extra.json")
	if err := SaveWeightsJSON(large, path); err != nil {
		t.Fatalf("SaveWeightsJSON: %v", err)
	}

	small := model.NewGPTModel(cfgSmall)
	err := LoadWeightsJSON(small, path, true)
	if err == nil {
		t.Fatal("expected error for extra keys in strict mode")
	}
}

func TestBinaryStrictMissingKey(t *testing.T) {
	cfgSmall := testConfig()
	cfgSmall.NLayers = 1

	cfgLarge := testConfig()

	small := model.NewGPTModel(cfgSmall)
	path := filepath.Join(t.TempDir(), "weights_partial.bin")
	if err := SaveWeightsBinary(small, path); err != nil {
		t.Fatalf("SaveWeightsBinary: %v", err)
	}

	large := model.NewGPTModel(cfgLarge)
	err := LoadWeightsBinary(large, path, true)
	if err == nil {
		t.Fatal("expected error for missing keys in strict mode")
	}
}

func TestBinaryStrictExtraKey(t *testing.T) {
	cfgSmall := testConfig()
	cfgSmall.NLayers = 1

	cfgLarge := testConfig()

	large := model.NewGPTModel(cfgLarge)
	path := filepath.Join(t.TempDir(), "weights_extra.bin")
	if err := SaveWeightsBinary(large, path); err != nil {
		t.Fatalf("SaveWeightsBinary: %v", err)
	}

	small := model.NewGPTModel(cfgSmall)
	err := LoadWeightsBinary(small, path, true)
	if err == nil {
		t.Fatal("expected error for extra keys in strict mode")
	}
}
