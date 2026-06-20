package attention

import (
	"math"
	"testing"

	llmmath "github.com/llm/math"
	"github.com/llm/tensor"
)

// TestSelfAttentionV2 verifies attention with learnable projections
func TestSelfAttentionV2(t *testing.T) {
	inFeatures := 4
	sa := NewSelfAttention(inFeatures, 2, 0.0)

	input := tensor.NewTensor(1, 2, inFeatures)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 1, 0, 0.5)

	result := sa.Forward(input, nil)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 2 || dims[2] != inFeatures {
		t.Errorf("Expected shape (1, 2, %d), got %v", inFeatures, dims)
	}
}

// TestCausalAttention verifies that tokens cannot attend to future tokens
func TestCausalAttention(t *testing.T) {
	inFeatures := 4
	sa := NewSelfAttention(inFeatures, 2, 0.0)

	input := tensor.NewTensor(1, 3, inFeatures) // 3 tokens
	for i := 0; i < 3; i++ {
		input.Set(0, i, 0, 1.0)
	}

	mask := llmmath.CreateCausalMask(3)

	result := sa.Forward(input, mask)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[1] != 3 {
		t.Errorf("Expected seq=3, got %d", dims[1])
	}
}

// TestExtractHead verifies that extractHead slices the correct portion of the tensor.
func TestExtractHead(t *testing.T) {
	dModel := 4
	nHeads := 2
	dK := dModel / nHeads // 2
	sa := NewSelfAttention(dModel, nHeads, 0.0)

	// Input: (1, 2, 4) with sequential values
	x := tensor.NewTensor(1, 2, dModel)
	x.Set(0, 0, 0, 0); x.Set(0, 0, 1, 1); x.Set(0, 0, 2, 2); x.Set(0, 0, 3, 3)
	x.Set(0, 1, 0, 4); x.Set(0, 1, 1, 5); x.Set(0, 1, 2, 6); x.Set(0, 1, 3, 7)

	// Head 0: columns 0-1
	h0 := sa.extractHead(x, 0)
	if h0.Dimensions()[1] != 2 || h0.Dimensions()[2] != dK {
		t.Errorf("Head 0 shape: (1,2,%d), got %v", dK, h0.Dimensions())
	}
	if h0.At(0, 0, 0) != 0 || h0.At(0, 0, 1) != 1 {
		t.Errorf("Head 0 seq0 wrong: got %v", []float64{h0.At(0, 0, 0), h0.At(0, 0, 1)})
	}
	if h0.At(0, 1, 0) != 4 || h0.At(0, 1, 1) != 5 {
		t.Errorf("Head 0 seq1 wrong: got %v", []float64{h0.At(0, 1, 0), h0.At(0, 1, 1)})
	}

	// Head 1: columns 2-3
	h1 := sa.extractHead(x, 2)
	if h1.At(0, 0, 0) != 2 || h1.At(0, 0, 1) != 3 {
		t.Errorf("Head 1 seq0 wrong: got %v", []float64{h1.At(0, 0, 0), h1.At(0, 0, 1)})
	}
	if h1.At(0, 1, 0) != 6 || h1.At(0, 1, 1) != 7 {
		t.Errorf("Head 1 seq1 wrong: got %v", []float64{h1.At(0, 1, 0), h1.At(0, 1, 1)})
	}
}

// TestSelfAttentionParameters verifies the structure of returned parameters.
func TestSelfAttentionParameters(t *testing.T) {
	// With bias
	sa := NewSelfAttention(4, 2, 0.0)
	params := sa.Parameters()
	expectedKeys := []string{
		"Wq.Weight", "Wq.Bias",
		"Wk.Weight", "Wk.Bias",
		"Wv.Weight", "Wv.Bias",
		"Wo.Weight", "Wo.Bias",
	}
	if len(params) != len(expectedKeys) {
		t.Errorf("Expected %d params with bias, got %d", len(expectedKeys), len(params))
	}
	for _, k := range expectedKeys {
		if _, ok := params[k]; !ok {
			t.Errorf("Missing param %q", k)
		}
	}

	// Without bias
	sa2 := NewSelfAttention(4, 2, 0.0)
	sa2.Wq.HasBias = false
	sa2.Wk.HasBias = false
	sa2.Wv.HasBias = false
	sa2.Wo.HasBias = false
	sa2.Wq.Bias = nil
	sa2.Wk.Bias = nil
	sa2.Wv.Bias = nil
	sa2.Wo.Bias = nil

	params2 := sa2.Parameters()
	if len(params2) != 4 {
		t.Errorf("Expected 4 params without bias, got %d", len(params2))
	}
	for k := range params2 {
		if k != "Wq.Weight" && k != "Wk.Weight" && k != "Wv.Weight" && k != "Wo.Weight" {
			t.Errorf("Unexpected param %q without bias", k)
		}
	}
}

// TestForwardWithCache verifies KV cache prefill-then-decode produces correct shapes.
func TestForwardWithCache(t *testing.T) {
	dModel := 4
	nHeads := 2
	sa := NewSelfAttention(dModel, nHeads, 0.0)

	// Prefill: 3 tokens with causal mask
	xPrefill := tensor.NewTensor(1, 3, dModel)
	xPrefill.Set(0, 0, 0, 1.0)
	xPrefill.Set(0, 1, 0, 0.5)
	xPrefill.Set(0, 2, 0, 0.2)
	mask := llmmath.CreateCausalMask(3)

	out, k, v := sa.ForwardWithCache(xPrefill, mask, nil, nil)
	if out == nil || k == nil || v == nil {
		t.Fatal("ForwardWithCache (prefill) returned nil")
	}
	outDims := out.Dimensions()
	if outDims[0] != 1 || outDims[1] != 3 || outDims[2] != dModel {
		t.Errorf("Prefill output shape (1,3,%d), got %v", dModel, outDims)
	}
	if k.Dimensions()[1] != 3 || v.Dimensions()[1] != 3 {
		t.Errorf("Prefill K,V seq should be 3, got K=%v V=%v", k.Dimensions(), v.Dimensions())
	}
	if k.Dimensions()[2] != dModel {
		t.Errorf("Prefill K embed should be %d, got %v", dModel, k.Dimensions())
	}

	// Decode: 1 new token with past cache, no mask
	xDecode := tensor.NewTensor(1, 1, dModel)
	xDecode.Set(0, 0, 0, 0.1)
	out2, k2, v2 := sa.ForwardWithCache(xDecode, nil, k, v)
	if out2 == nil || k2 == nil || v2 == nil {
		t.Fatal("ForwardWithCache (decode) returned nil")
	}
	out2Dims := out2.Dimensions()
	if out2Dims[0] != 1 || out2Dims[1] != 1 || out2Dims[2] != dModel {
		t.Errorf("Decode output shape (1,1,%d), got %v", dModel, out2Dims)
	}
	if k2.Dimensions()[1] != 4 || v2.Dimensions()[1] != 4 {
		t.Errorf("Decode K,V seq should be 4, got K=%v V=%v", k2.Dimensions(), v2.Dimensions())
	}

	// Output should not contain NaN or Inf
	for _, v := range out2.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Decode output contains NaN/Inf: %f", v)
		}
	}
}


