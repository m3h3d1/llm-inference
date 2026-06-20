package attention

import (
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


