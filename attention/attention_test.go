package attention

import (
	"testing"

	llmmath "github.com/llm/math"
	"github.com/llm/tensor"
)

// TestSelfAttentionV1 verifies basic dot-product attention (Q=K=V=X)
func TestSelfAttentionV1(t *testing.T) {
	input := tensor.NewTensor(1, 2, 2)
	input.Set(0, 0, 0, 1.0); input.Set(0, 0, 1, 0.0)
	input.Set(0, 1, 0, 0.0); input.Set(0, 1, 1, 1.0)

	result := SimpleAttention(input)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	if result.At(0, 0, 0) < 0.6 || result.At(0, 1, 1) < 0.6 {
		t.Errorf("SelfAttentionV1 failed to maintain identity for identity input, got %v", result.Data)
	}
}

// TestSelfAttentionV2 verifies attention with learnable projections
func TestSelfAttentionV2(t *testing.T) {
	inFeatures := 4
	sa := NewSelfAttention(inFeatures, 0.0)

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
	sa := NewSelfAttention(inFeatures, 0.0)

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

// TestMultiHeadAttention verifies MHA output shape and basic functionality
func TestMultiHeadAttention(t *testing.T) {
	dModel := 8
	nHeads := 2
	mha := NewMultiHeadAttention(dModel, nHeads)

	input := tensor.NewTensor(1, 2, dModel)
	for i := 0; i < dModel; i++ {
		input.Set(0, 0, i, float64(i))
		input.Set(0, 1, i, float64(i)*0.5)
	}

	result := mha.Forward(input, nil)

	if result == nil {
		t.Fatal("Result should not be nil")
	}

	dims := result.Dimensions()
	if dims[0] != 1 || dims[1] != 2 || dims[2] != dModel {
		t.Errorf("Expected shape (1, 2, %d), got %v", dModel, dims)
	}
}
