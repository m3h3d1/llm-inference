package tensor

// Tensor represents a 3D tensor [batch, seq, embed]
type Tensor struct {
	Data  []float64
	Shape [3]int // [batch, seq, embed]
}


// NewTensor creates a new 3D tensor with given dimensions
func NewTensor(batch, seq, embed int) *Tensor {
	size := batch * seq * embed
	return &Tensor{
		Data:  make([]float64, size),
		Shape: [3]int{batch, seq, embed},
	}
}

// Dimensions returns the shape of the tensor
func (t *Tensor) Dimensions() [3]int {
	return t.Shape
}

// Batch returns the batch size
func (t *Tensor) Batch() int {
	return t.Dimensions()[0]
}

// Seq returns the sequence length
func (t *Tensor) Seq() int {
	return t.Dimensions()[1]
}

// Embed returns the embedding dimension
func (t *Tensor) Embed() int {
	return t.Dimensions()[2]
}

// At returns the value at the specified position
func (t *Tensor) At(b, s, e int) float64 {
	dims := t.Dimensions()
	idx := b*dims[1]*dims[2] + s*dims[2] + e
	return t.Data[idx]
}

// Set sets the value at the specified position
func (t *Tensor) Set(b, s, e int, value float64) {
	dims := t.Dimensions()
	idx := b*dims[1]*dims[2] + s*dims[2] + e
	t.Data[idx] = value
}

// Flatten returns a flattened 1D slice
func (t *Tensor) Flatten() []float64 {
	return t.Data
}

// Reshape creates a new tensor with the given shape
func (t *Tensor) Reshape(batch, seq, embed int) *Tensor {
	newSize := batch * seq * embed
	if newSize != len(t.Data) {
		return nil // Cannot reshape to different total size
	}
	return &Tensor{
		Data:  t.Data,
		Shape: [3]int{batch, seq, embed},
	}
}

// AtIdx returns the value at flattened index
func (t *Tensor) AtIdx(idx int) float64 {
	return t.Data[idx]
}

// SetIdx sets the value at flattened index
func (t *Tensor) SetIdx(idx int, value float64) {
	t.Data[idx] = value
}

// Zeros creates a tensor filled with zeros
func Zeros(batch, seq, embed int) *Tensor {
	return NewTensor(batch, seq, embed)
}

// Ones creates a tensor filled with ones
func Ones(batch, seq, embed int) *Tensor {
	t := NewTensor(batch, seq, embed)
	for i := range t.Data {
		t.Data[i] = 1.0
	}
	return t
}

// Scale multiplies all elements by a scalar factor
func (t *Tensor) Scale(factor float64) *Tensor {
	result := NewTensor(t.Batch(), t.Seq(), t.Embed())
	for i := range t.Data {
		result.Data[i] = t.Data[i] * factor
	}
	return result
}

// ConcatSeq concatenates tensors along the seq dimension (axis 1).
// All tensors must share the same batch and embed dimensions.
func ConcatSeq(tensors []*Tensor) *Tensor {
	if len(tensors) == 0 {
		return nil
	}

	dims := tensors[0].Dimensions()
	batch, _, embed := dims[0], dims[1], dims[2]

	totalSeq := 0
	for _, t := range tensors {
		tDims := t.Dimensions()
		if tDims[0] != batch || tDims[2] != embed {
			return nil
		}
		totalSeq += tDims[1]
	}

	result := NewTensor(batch, totalSeq, embed)

	for b := 0; b < batch; b++ {
		currentSeq := 0
		for _, t := range tensors {
			tSeq := t.Dimensions()[1]
			for s := 0; s < tSeq; s++ {
				for e := 0; e < embed; e++ {
					result.Set(b, currentSeq+s, e, t.At(b, s, e))
				}
			}
			currentSeq += tSeq
		}
	}
	return result
}

// Add performs element-wise addition with broadcasting
func (t *Tensor) Add(other *Tensor) *Tensor {
	dims1 := t.Dimensions()
	dims2 := other.Dimensions()

	// Determine result shape and broadcasting
	resShape := [3]int{}
	for i := 0; i < 3; i++ {
		if dims1[i] == dims2[i] {
			resShape[i] = dims1[i]
		} else if dims1[i] == 1 {
			resShape[i] = dims2[i]
		} else if dims2[i] == 1 {
			resShape[i] = dims1[i]
		} else {
			return nil // Incompatible shapes
		}
	}

	result := NewTensor(resShape[0], resShape[1], resShape[2])

	for b := 0; b < resShape[0]; b++ {
		for s := 0; s < resShape[1]; s++ {
			for e := 0; e < resShape[2]; e++ {
				// Map indices back to source tensors (broadcasting)
				b1, s1, e1 := b, s, e
				if dims1[0] == 1 { b1 = 0 }
				if dims1[1] == 1 { s1 = 0 }
				if dims1[2] == 1 { e1 = 0 }

				b2, s2, e2 := b, s, e
				if dims2[0] == 1 { b2 = 0 }
				if dims2[1] == 1 { s2 = 0 }
				if dims2[2] == 1 { e2 = 0 }

				result.Set(b, s, e, t.At(b1, s1, e1)+other.At(b2, s2, e2))
			}
		}
	}
	return result
}

func Concat(tensors []*Tensor) *Tensor {
	if len(tensors) == 0 {
		return nil
	}
	
	dims := tensors[0].Dimensions()
	batch, seq, _ := dims[0], dims[1], dims[2]
	
	totalEmbed := 0
	for _, t := range tensors {
		tDims := t.Dimensions()
		if tDims[0] != batch || tDims[1] != seq {
			return nil // Incompatible batch or seq
		}
		totalEmbed += tDims[2]
	}
	
	result := NewTensor(batch, seq, totalEmbed)
	
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			currentEmbed := 0
			for _, t := range tensors {
				tEmbed := t.Dimensions()[2]
				for e := 0; e < tEmbed; e++ {
					result.Set(b, s, currentEmbed+e, t.At(b, s, e))
				}
				currentEmbed += tEmbed
			}
		}
	}
	return result
}

// MatMul performs matrix multiplication: this (batch, seq1, embed) × other (batch, embed, seq2) = (batch, seq1, seq2)
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	dims1 := t.Dimensions()
	dims2 := other.Dimensions()
	
	batch := dims1[0]
	seq1 := dims1[1]
	embed := dims1[2]
	seq2 := dims2[2]
	
	result := NewTensor(batch, seq1, seq2)
	
	for b := 0; b < batch; b++ {
		for i := 0; i < seq1; i++ {
			for j := 0; j < seq2; j++ {
				var sum float64
				for e := 0; e < embed; e++ {
					sum += t.At(b, i, e) * other.At(b, e, j)
				}
				result.Set(b, i, j, sum)
			}
		}
	}
	
	return result
}

// Transpose swaps the last two dimensions (seq, embed) -> (embed, seq)
func (t *Tensor) Transpose() *Tensor {
	dims := t.Dimensions()
	batch := dims[0]
	seq := dims[1]
	embed := dims[2]
	
	result := NewTensor(batch, embed, seq)
	
	for b := 0; b < batch; b++ {
		for i := 0; i < seq; i++ {
			for j := 0; j < embed; j++ {
				result.Set(b, j, i, t.At(b, i, j))
			}
		}
	}
	
	return result
}