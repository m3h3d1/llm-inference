package math

import (
	"math"
	"testing"

	"github.com/llm/tensor"
)

// TestSoftmax verifies Softmax computation against expected values
// Input: [1.0, 2.0, 3.0, 4.0]
// Expected: [exp(1)/sum, exp(2)/sum, exp(3)/sum, exp(4)/sum]
func TestSoftmax(t *testing.T) {
	input := tensor.NewTensor(1, 1, 4)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)
	input.Set(0, 0, 2, 3.0)
	input.Set(0, 0, 3, 4.0)

	result := Softmax(input, -1)

	exp0 := math.Exp(1.0)
	exp1 := math.Exp(2.0)
	exp2 := math.Exp(3.0)
	exp3 := math.Exp(4.0)
	sum := exp0 + exp1 + exp2 + exp3

	tolerance := 1e-6
	if math.Abs(result.At(0, 0, 0)-exp0/sum) > tolerance {
		t.Errorf("Expected %f, got %f", exp0/sum, result.At(0, 0, 0))
	}
	if math.Abs(result.At(0, 0, 1)-exp1/sum) > tolerance {
		t.Errorf("Expected %f, got %f", exp1/sum, result.At(0, 0, 1))
	}
	if math.Abs(result.At(0, 0, 2)-exp2/sum) > tolerance {
		t.Errorf("Expected %f, got %f", exp2/sum, result.At(0, 0, 2))
	}
	if math.Abs(result.At(0, 0, 3)-exp3/sum) > tolerance {
		t.Errorf("Expected %f, got %f", exp3/sum, result.At(0, 0, 3))
	}
}

// TestSoftmaxSumsToOne verifies property: softmax output sums to 1.0
func TestSoftmaxSumsToOne(t *testing.T) {
	input := tensor.NewTensor(1, 1, 3)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)
	input.Set(0, 0, 2, 3.0)

	result := Softmax(input, -1)

	sum := result.At(0, 0, 0) + result.At(0, 0, 1) + result.At(0, 0, 2)
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Expected sum = 1.0, got %f", sum)
	}
}

// TestSoftmaxWithNegatives verifies Softmax with negative input values
func TestSoftmaxWithNegatives(t *testing.T) {
	input := tensor.NewTensor(1, 1, 3)
	input.Set(0, 0, 0, -1.0)
	input.Set(0, 0, 1, 0.0)
	input.Set(0, 0, 2, 1.0)

	result := Softmax(input, -1)

	sum := result.At(0, 0, 0) + result.At(0, 0, 1) + result.At(0, 0, 2)
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Expected sum = 1.0, got %f", sum)
	}
}

// TestSoftmaxWithZeros verifies Softmax with all zeros input
func TestSoftmaxWithZeros(t *testing.T) {
	input := tensor.NewTensor(1, 1, 3)
	input.Set(0, 0, 0, 0.0)
	input.Set(0, 0, 1, 0.0)
	input.Set(0, 0, 2, 0.0)

	result := Softmax(input, -1)

	sum := result.At(0, 0, 0) + result.At(0, 0, 1) + result.At(0, 0, 2)
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Expected sum = 1.0, got %f", sum)
	}
}

// TestGELU verifies GELU computation against expected formula
// Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func TestGELU(t *testing.T) {
	input := tensor.NewTensor(1, 1, 1)
	input.Set(0, 0, 0, 1.0)

	result := GELU(input)

	sqrt2OverPi := math.Sqrt(2.0 / math.Pi)
	expected := 0.5 * 1.0 * (1 + math.Tanh(sqrt2OverPi*(1.0+0.044715*1.0)))

	tolerance := 1e-6
	if math.Abs(result.At(0, 0, 0)-expected) > tolerance {
		t.Errorf("Expected %f, got %f", expected, result.At(0, 0, 0))
	}
}

// TestGELUZero verifies GELU(0) = 0 property
func TestGELUZero(t *testing.T) {
	input := tensor.NewTensor(1, 1, 1)
	input.Set(0, 0, 0, 0.0)

	result := GELU(input)

	if math.Abs(result.At(0, 0, 0)) > 1e-6 {
		t.Errorf("Expected GELU(0) = 0, got %f", result.At(0, 0, 0))
	}
}

// TestGELUPositive verifies GELU(x) > 0 for x > 0
func TestGELUPositive(t *testing.T) {
	input := tensor.NewTensor(1, 1, 1)
	input.Set(0, 0, 0, 2.0)

	result := GELU(input)

	if result.At(0, 0, 0) <= 0 {
		t.Errorf("Expected GELU(x) > 0 for x > 0, got %f", result.At(0, 0, 0))
	}
}

// TestGELUNegative verifies GELU(x) < 0 for x < 0
func TestGELUNegative(t *testing.T) {
	input := tensor.NewTensor(1, 1, 1)
	input.Set(0, 0, 0, -2.0)

	result := GELU(input)

	if result.At(0, 0, 0) >= 0 {
		t.Errorf("Expected GELU(x) < 0 for x < 0, got %f", result.At(0, 0, 0))
	}
}

// TestGELUShape verifies GELU preserves input shape
func TestGELUShape(t *testing.T) {
	input := tensor.NewTensor(2, 3, 4)

	result := GELU(input)

	if result.Dimensions() != input.Dimensions() {
		t.Errorf("Expected %v, got %v", input.Dimensions(), result.Dimensions())
	}
}

// TestLayerNormBasic verifies LayerNorm with provided gamma and beta
func TestLayerNormBasic(t *testing.T) {
	// Input: (1, 1, 4) with values [1, 2, 3, 4]
	input := tensor.NewTensor(1, 1, 4)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)
	input.Set(0, 0, 2, 3.0)
	input.Set(0, 0, 3, 4.0)

	// gamma = ones, beta = zeros
	gamma := tensor.NewTensor(1, 1, 4)
	beta := tensor.NewTensor(1, 1, 4)
	for i := 0; i < 4; i++ {
		gamma.Set(0, 0, i, 1.0)
		beta.Set(0, 0, i, 0.0)
	}

	result := LayerNorm(input, gamma, beta, 1e-5)

	// After normalization: should have mean ≈ 0, var ≈ 1
	// But with gamma=1, beta=0: output = (x - mean) / sqrt(var + eps)
	mean := (1.0 + 2.0 + 3.0 + 4.0) / 4.0
	expectedMean := 0.0
	expectedVar := ((1-mean)*(1-mean) + (2-mean)*(2-mean) + (3-mean)*(3-mean) + (4-mean)*(4-mean)) / 4.0

	tolerance := 1e-5
	avgDiff := result.At(0, 0, 0) + result.At(0, 0, 1) + result.At(0, 0, 2) + result.At(0, 0, 3)
	avgDiff = avgDiff / 4.0
	if math.Abs(avgDiff-expectedMean) > tolerance {
		t.Errorf("Expected mean ≈ 0, got %f", avgDiff)
	}
	_ = expectedVar
}

// TestLayerNormDefaultGammaBeta verifies LayerNorm with nil gamma/beta (defaults to 1.0 and 0.0)
func TestLayerNormDefaultGammaBeta(t *testing.T) {
	input := tensor.NewTensor(1, 1, 3)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)
	input.Set(0, 0, 2, 3.0)

	result := LayerNorm(input, nil, nil, 1e-5)

	// Output shape should match input
	if result.Dimensions() != input.Dimensions() {
		t.Errorf("Expected shape %v, got %v", input.Dimensions(), result.Dimensions())
	}
}

// TestLayerNormShape verifies LayerNorm preserves (batch, seq, embed) shape
func TestLayerNormShape(t *testing.T) {
	input := tensor.NewTensor(2, 3, 4)
	gamma := tensor.NewTensor(1, 1, 4)
	beta := tensor.NewTensor(1, 1, 4)

	result := LayerNorm(input, gamma, beta, 1e-5)

	if result.Dimensions() != input.Dimensions() {
		t.Errorf("Expected %v, got %v", input.Dimensions(), result.Dimensions())
	}
}

// TestCausalMask verifies causal mask structure
// Upper triangle (j > i) should be -inf
// Lower triangle (j <= i) should be 0
func TestCausalMask(t *testing.T) {
	mask := CreateCausalMask(3)

	// Verify shape: (1, 3, 3)
	dims := mask.Dimensions()
	if dims[0] != 1 || dims[1] != 3 || dims[2] != 3 {
		t.Errorf("Expected shape (1, 3, 3), got %v", dims)
	}

	// Verify upper triangle is -inf (j > i)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if j > i {
				if !math.IsInf(mask.At(0, i, j), -1) {
					t.Errorf("Expected -inf at [%d][%d], got %f", i, j, mask.At(0, i, j))
				}
			} else {
				if mask.At(0, i, j) != 0.0 {
					t.Errorf("Expected 0 at [%d][%d], got %f", i, j, mask.At(0, i, j))
				}
			}
		}
	}
}

// TestCausalMaskSize4 verifies causal mask with different size
func TestCausalMaskSize4(t *testing.T) {
	mask := CreateCausalMask(4)

	dims := mask.Dimensions()
	if dims[1] != 4 || dims[2] != 4 {
		t.Errorf("Expected (1, 4, 4), got %v", dims)
	}

	// Verify upper triangle is -inf
	if !math.IsInf(mask.At(0, 2, 3), -1) {
		t.Errorf("Expected -inf at [2][3], got %f", mask.At(0, 2, 3))
	}
	if mask.At(0, 0, 0) != 0.0 {
		t.Errorf("Expected 0 at [0][0], got %f", mask.At(0, 0, 0))
	}
}