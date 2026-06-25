package linear

import (
	"math"
	"testing"

	"github.com/llm/tensor"
)

// TestLinearForwardShape verifies Linear produces correct output shape with no NaN
// Input: (batch=1, seq=2, inFeatures=3)
// Expected: (batch=1, seq=2, outFeatures=4)
func TestLinearForwardShape(t *testing.T) {
	inFeatures := 3
	outFeatures := 4

	linear := NewLinear(inFeatures, outFeatures, false)

	input := tensor.NewTensor(1, 2, inFeatures)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)
	input.Set(0, 0, 2, 3.0)
	input.Set(0, 1, 0, 4.0)
	input.Set(0, 1, 1, 5.0)
	input.Set(0, 1, 2, 6.0)

	result := linear.Forward(input)

	dims := result.Dimensions()
	if dims[0] != 1 {
		t.Errorf("Expected batch=1, got %d", dims[0])
	}
	if dims[1] != 2 {
		t.Errorf("Expected seq=2, got %d", dims[1])
	}
	if dims[2] != outFeatures {
		t.Errorf("Expected outFeatures=%d, got %d", outFeatures, dims[2])
	}
	for _, v := range result.Data {
		if math.IsNaN(v) {
			t.Error("NaN in linear forward output")
			break
		}
	}
}

// TestLinearForwardWithBias verifies bias is added when enabled
// Input: (1,1,2) = [[1, 2]]
// Weights: (1, 2, 2)  [0, 0]
//                     [0, 0]
// Bias: (1, 1, 2) = [1, 1] (set to ones)
func TestLinearForwardWithBias(t *testing.T) {
	inFeatures := 2
	outFeatures := 2

	linear := NewLinear(inFeatures, outFeatures, true)

	input := tensor.NewTensor(1, 1, inFeatures)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)

	result := linear.Forward(input)

	dims := result.Dimensions()
	if dims[2] != outFeatures {
		t.Errorf("Expected outFeatures=%d, got %d", outFeatures, dims[2])
	}

	// Weights initialized as W[o][i] = float64(o)*0.01, Bias = 1.0
	// Input [1.0, 2.0]:
	//   output[0] = 0.00*1 + 0.00*2 + 1.0 = 1.0
	//   output[1] = 0.01*1 + 0.01*2 + 1.0 = 1.03
	got0 := result.At(0, 0, 0)
	got1 := result.At(0, 0, 1)
	if got0 != 1.0 || got1 != 1.03 {
		t.Errorf("Expected [1.0, 1.03], got [%f, %f]", got0, got1)
	}
}

// TestLinearParameters verifies Parameters returns weight and bias
func TestLinearParameters(t *testing.T) {
	linear := NewLinear(3, 4, true)
	params := linear.Parameters()

	if len(params) != 2 {
		t.Errorf("Expected 2 params (weight + bias), got %d", len(params))
	}

	weight, ok := params["Weight"]
	if !ok {
		t.Errorf("Expected Weight param")
	}
	weightDims := weight.Dimensions()
	if weightDims[1] != 4 || weightDims[2] != 3 {
		t.Errorf("Expected weight shape (1, 4, 3), got %v", weightDims)
	}

	bias, ok := params["Bias"]
	if !ok {
		t.Errorf("Expected Bias param")
	}
	biasDims := bias.Dimensions()
	if biasDims[2] != 4 {
		t.Errorf("Expected bias shape (1, 1, 4), got %v", biasDims)
	}
}

// TestLinearNoBias verifies works without bias
func TestLinearNoBias(t *testing.T) {
	linear := NewLinear(2, 3, false)
	params := linear.Parameters()

	if len(params) != 1 {
		t.Errorf("Expected 1 param (weight only), got %d", len(params))
	}
	_, ok := params["Weight"]
	if !ok {
		t.Errorf("Expected Weight param")
	}
}

// TestLinearBatchGreaterThan1 verifies with batch > 1
func TestLinearBatchGreaterThan1(t *testing.T) {
	inFeatures := 2
	outFeatures := 3

	linear := NewLinear(inFeatures, outFeatures, true)

	input := tensor.NewTensor(2, 1, inFeatures)
	input.Set(0, 0, 0, 1.0)
	input.Set(0, 0, 1, 2.0)
	input.Set(1, 0, 0, 3.0)
	input.Set(1, 0, 1, 4.0)

	result := linear.Forward(input)

	dims := result.Dimensions()
	if dims[0] != 2 {
		t.Errorf("Expected batch=2, got %d", dims[0])
	}
	if dims[2] != outFeatures {
		t.Errorf("Expected outFeatures=%d, got %d", outFeatures, dims[2])
	}
	for _, v := range result.Data {
		if math.IsNaN(v) {
			t.Error("NaN in linear batch forward output")
			break
		}
	}
}

// TestLinearSeqGreaterThan1 verifies with seq > 1
func TestLinearSeqGreaterThan1(t *testing.T) {
	inFeatures := 2
	outFeatures := 3

	linear := NewLinear(inFeatures, outFeatures, false)

	input := tensor.NewTensor(1, 3, inFeatures)
	result := linear.Forward(input)

	dims := result.Dimensions()
	if dims[1] != 3 {
		t.Errorf("Expected seq=3, got %d", dims[1])
	}
	for _, v := range result.Data {
		if math.IsNaN(v) {
			t.Error("NaN in linear seq forward output")
			break
		}
	}
}