package tensor

import (
	"testing"
)

// TestNewTensor verifies that NewTensor creates a 3D tensor with correct shape [batch, seq, embed]
// and allocates the proper amount of data (batch * seq * embed).
// Input:  batch=2, seq=3, embed=4
// Expected: total data = 2 * 3 * 4 = 24 elements
func TestNewTensor(t *testing.T) {
	shape := [3]int{2, 3, 4}
	expectedSize := 24

	tensor := NewTensor(shape[0], shape[1], shape[2])

	dims := tensor.Dimensions()
	if dims[0] != shape[0] {
		t.Errorf("Expected batch size %d, got %d", shape[0], dims[0])
	}
	if dims[1] != shape[1] {
		t.Errorf("Expected seq size %d, got %d", shape[1], dims[1])
	}
	if dims[2] != shape[2] {
		t.Errorf("Expected embed size %d, got %d", shape[2], dims[2])
	}
	if len(tensor.Data) != expectedSize {
		t.Errorf("Expected data size %d, got %d", expectedSize, len(tensor.Data))
	}
}

// TestTensorShape verifies that Dimensions() returns the correct shape [batch, seq, embed].
func TestTensorShape(t *testing.T) {
	shape := [3]int{1, 2, 3}
	tensor := NewTensor(shape[0], shape[1], shape[2])

	got := tensor.Dimensions()
	expected := shape

	if got[0] != expected[0] || got[1] != expected[1] || got[2] != expected[2] {
		t.Errorf("Expected shape %v, got %v", expected, got)
	}
}

// TestTensorAt verifies that At() getter and Set() setter work correctly.
// Input tensor at (0,0,0)=1.5, (0,1,0)=2.5
// Expected: At(0,0,0)=1.5, At(0,1,0)=2.5
func TestTensorAt(t *testing.T) {
	shape := [3]int{1, 2, 1}
	tensor := NewTensor(shape[0], shape[1], shape[2])
	tensor.Set(0, 0, 0, 1.5)
	tensor.Set(0, 1, 0, 2.5)

	val0 := tensor.At(0, 0, 0)
	if val0 != 1.5 {
		t.Errorf("Expected 1.5 at (0,0,0), got %f", val0)
	}

	val1 := tensor.At(0, 1, 0)
	if val1 != 2.5 {
		t.Errorf("Expected 2.5 at (0,1,0), got %f", val1)
	}
}

// TestMatMul verifies batch matrix multiplication:
// A (batch=1, seq1=2, embed=3):
//   [1, 2, 3]
//   [4, 5, 6]
// B (batch=1, embed=3, seq2=2):
//   [7,  8]
//   [9, 10]
//   [11,12]
// Expected C (batch=1, seq1=2, seq2=2):
//   [58,  64]   (1*7+2*9+3*11, 1*8+2*10+3*12)
//   [139, 154]  (4*7+5*9+6*11, 4*8+5*10+6*12)
func TestMatMul(t *testing.T) {
	batch := 1
	seq1 := 2
	seq2 := 2
	embed := 3

	A := NewTensor(batch, seq1, embed)
	A.Set(0, 0, 0, 1); A.Set(0, 0, 1, 2); A.Set(0, 0, 2, 3)
	A.Set(0, 1, 0, 4); A.Set(0, 1, 1, 5); A.Set(0, 1, 2, 6)

	B := NewTensor(batch, embed, seq2)
	B.Set(0, 0, 0, 7); B.Set(0, 0, 1, 8)
	B.Set(0, 1, 0, 9); B.Set(0, 1, 1, 10)
	B.Set(0, 2, 0, 11); B.Set(0, 2, 1, 12)

	C := A.MatMul(B)

	if C.At(0, 0, 0) != 58 {
		t.Errorf("Expected C[0,0,0]=58, got %f", C.At(0, 0, 0))
	}
	if C.At(0, 0, 1) != 64 {
		t.Errorf("Expected C[0,0,1]=64, got %f", C.At(0, 0, 1))
	}
	if C.At(0, 1, 0) != 139 {
		t.Errorf("Expected C[0,1,0]=139, got %f", C.At(0, 1, 0))
	}
	if C.At(0, 1, 1) != 154 {
		t.Errorf("Expected C[0,1,1]=154, got %f", C.At(0, 1, 1))
	}
}

// TestTranspose verifies swapping last two dimensions:
// Input A (batch=1, seq=2, embed=3):
//   [1, 2, 3]
//   [4, 5, 6]
// Expected (batch=1, embed=3, seq=2):
//   [1, 4]
//   [2, 5]
//   [3, 6]
func TestTranspose(t *testing.T) {
	batch, seq, embed := 1, 2, 3
	A := NewTensor(batch, seq, embed)
	A.Set(0, 0, 0, 1); A.Set(0, 0, 1, 2); A.Set(0, 0, 2, 3)
	A.Set(0, 1, 0, 4); A.Set(0, 1, 1, 5); A.Set(0, 1, 2, 6)

	AT := A.Transpose()

	if AT.Dimensions()[1] != embed || AT.Dimensions()[2] != seq {
		t.Errorf("Expected shape (1, %d, %d), got %v", embed, seq, AT.Dimensions())
	}

	if AT.At(0, 0, 1) != 4 {
		t.Errorf("Expected AT[0,0,1]=4, got %f", AT.At(0, 0, 1))
	}
	if AT.At(0, 2, 1) != 6 {
		t.Errorf("Expected AT[0,2,1]=6, got %f", AT.At(0, 2, 1))
	}
}

// TestReshape verifies reshape works with valid shapes and returns nil for invalid shapes.
// Valid reshape: (1,2,3) with 6 elements -> (1,3,2) with 6 elements
// Invalid reshape: (1,2,3) with 6 elements -> (1,1,10) with 10 elements (fails)
func TestReshape(t *testing.T) {
	A := NewTensor(1, 2, 3)
	B := A.Reshape(1, 3, 2)

	if B == nil || B.Dimensions()[1] != 3 || B.Dimensions()[2] != 2 {
		t.Errorf("Reshape failed to change dimensions correctly")
	}

	C := A.Reshape(1, 1, 10)
	if C != nil {
		t.Error("Expected nil for invalid reshape")
	}
}

// TestZerosOnes verifies factory functions create tensors with correct initial values.
// Zeros: all elements should be 0.0
// Ones: all elements should be 1.0
func TestZerosOnes(t *testing.T) {
	Z := Zeros(1, 2, 2)
	for _, v := range Z.Data {
		if v != 0.0 {
			t.Errorf("Expected zero, got %f", v)
		}
	}

	O := Ones(1, 2, 2)
	for _, v := range O.Data {
		if v != 1.0 {
			t.Errorf("Expected one, got %f", v)
		}
	}
}

// TestFlattenAndIdx verifies flat indexing methods work correctly.
// Input tensor (1,2,2):
//   [10, 20]
//   [30, 40]
// Flattened: [10, 20, 30, 40]
// AtIdx(2) should return 30
func TestFlattenAndIdx(t *testing.T) {
	A := NewTensor(1, 2, 2)
	A.Set(0, 0, 0, 10)
	A.Set(0, 0, 1, 20)
	A.Set(0, 1, 0, 30)
	A.Set(0, 1, 1, 40)

	flat := A.Flatten()
	if len(flat) != 4 || flat[2] != 30 {
		t.Errorf("Flatten failed, got %v", flat)
	}

	if A.AtIdx(2) != 30 {
		t.Errorf("AtIdx failed, expected 30, got %f", A.AtIdx(2))
	}
}

// TestConcatSeq verifies concatenation along the seq dimension
func TestConcatSeq(t *testing.T) {
	// Case 1: Basic concat
	A := NewTensor(1, 2, 3)
	A.Set(0, 0, 0, 1); A.Set(0, 0, 1, 2); A.Set(0, 0, 2, 3)
	A.Set(0, 1, 0, 4); A.Set(0, 1, 1, 5); A.Set(0, 1, 2, 6)
	B := NewTensor(1, 2, 3)
	B.Set(0, 0, 0, 7); B.Set(0, 0, 1, 8); B.Set(0, 0, 2, 9)
	B.Set(0, 1, 0, 10); B.Set(0, 1, 1, 11); B.Set(0, 1, 2, 12)

	C := ConcatSeq([]*Tensor{A, B})
	if C == nil {
		t.Fatal("ConcatSeq returned nil")
	}
	dims := C.Dimensions()
	if dims[0] != 1 || dims[1] != 4 || dims[2] != 3 {
		t.Errorf("Expected shape (1,4,3), got %v", dims)
	}
	if C.At(0, 0, 0) != 1 || C.At(0, 3, 2) != 12 {
		t.Errorf("ConcatSeq values wrong: got %v", C.Data)
	}

	// Case 2: Shape mismatch returns nil
	C2 := ConcatSeq([]*Tensor{NewTensor(1, 2, 3), NewTensor(2, 2, 3)})
	if C2 != nil {
		t.Error("Expected nil for batch mismatch")
	}

	C3 := ConcatSeq([]*Tensor{NewTensor(1, 2, 3), NewTensor(1, 2, 4)})
	if C3 != nil {
		t.Error("Expected nil for embed mismatch")
	}

	// Case 3: Empty input returns nil
	if ConcatSeq(nil) != nil {
		t.Error("Expected nil for nil input")
	}
}

// TestScale verifies scalar multiplication
func TestScale(t *testing.T) {
	A := NewTensor(1, 1, 3)
	A.Set(0, 0, 0, 1.0)
	A.Set(0, 0, 1, 2.0)
	A.Set(0, 0, 2, 3.0)

	scaled := A.Scale(0.5)

	if scaled.At(0, 0, 0) != 0.5 || scaled.At(0, 0, 1) != 1.0 || scaled.At(0, 0, 2) != 1.5 {
		t.Errorf("Scale failed, got %v", scaled.Data)
	}
}

// TestAdd verifies element-wise addition with broadcasting
func TestAdd(t *testing.T) {
	// Case 1: Identical shapes
	A := NewTensor(1, 1, 2)
	A.Set(0, 0, 0, 1.0)
	A.Set(0, 0, 1, 2.0)
	B := NewTensor(1, 1, 2)
	B.Set(0, 0, 0, 3.0)
	B.Set(0, 0, 1, 4.0)
	
	sum := A.Add(B)
	if sum.At(0, 0, 0) != 4.0 || sum.At(0, 0, 1) != 6.0 {
		t.Errorf("Simple Add failed, got %v", sum.Data)
	}

	// Case 2: Batch broadcasting (B, S, E) + (1, S, E)
	A2 := NewTensor(2, 1, 2)
	A2.Set(0, 0, 0, 1.0); A2.Set(0, 0, 1, 1.0)
	A2.Set(1, 0, 0, 2.0); A2.Set(1, 0, 1, 2.0)
	B2 := NewTensor(1, 1, 2)
	B2.Set(0, 0, 0, 10.0); B2.Set(0, 0, 1, 20.0)

	sum2 := A2.Add(B2)
	if sum2.At(0, 0, 0) != 11.0 || sum2.At(1, 0, 0) != 12.0 {
		t.Errorf("Batch broadcasting Add failed, got %v", sum2.Data)
	}

	// Case 3: Invalid shape
	C := NewTensor(1, 1, 3)
	res := A.Add(C)
	if res != nil {
		t.Error("Expected nil for invalid shape addition")
	}
}
