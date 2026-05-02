package tensor

import (
	"testing"
)

func TestNewTensor(t *testing.T) {
	// Test creating a new 3D tensor
	shape := [3]int{2, 3, 4} // batch=2, seq=3, embed=4
	expectedSize := 24 // 2 * 3 * 4

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

func TestTensorShape(t *testing.T) {
	// Test getting the shape of a tensor
	shape := [3]int{1, 2, 3}
	tensor := NewTensor(shape[0], shape[1], shape[2])

	got := tensor.Dimensions()
	expected := shape

	if got[0] != expected[0] || got[1] != expected[1] || got[2] != expected[2] {
		t.Errorf("Expected shape %v, got %v", expected, got)
	}
}

func TestTensorAt(t *testing.T) {
	// Test getting element at specific position
	shape := [3]int{1, 2, 1}
	tensor := NewTensor(shape[0], shape[1], shape[2])
	tensor.Set(0, 0, 0, 1.5)
	tensor.Set(0, 1, 0, 2.5)

	// Test getting values
	val0 := tensor.At(0, 0, 0)
	if val0 != 1.5 {
		t.Errorf("Expected 1.5 at (0,0,0), got %f", val0)
	}

	val1 := tensor.At(0, 1, 0)
	if val1 != 2.5 {
		t.Errorf("Expected 2.5 at (0,1,0), got %f", val1)
	}
}