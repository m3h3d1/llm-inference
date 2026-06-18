package math

import (
	"math"
	"math/rand"

	"github.com/llm/tensor"
)

// Softmax computes softmax along the specified dimension
// Formula: softmax(x_i) = e^x_i / sum(e^x_j)
func Softmax(t *tensor.Tensor, dim int) *tensor.Tensor {
	dims := t.Dimensions()
	result := tensor.NewTensor(dims[0], dims[1], dims[2])

	// For dim=-1, apply softmax across last dimension
	if dim == -1 || dim == 2 {
		// Apply softmax across embedding dimension
		batchSize := dims[0]
		seqLen := dims[1]
		embedDim := dims[2]

		for b := 0; b < batchSize; b++ {
			for s := 0; s < seqLen; s++ {
				// Find max for numerical stability
				maxVal := t.At(b, s, 0)
				for e := 1; e < embedDim; e++ {
					val := t.At(b, s, e)
					if val > maxVal {
						maxVal = val
					}
				}

				// Compute exp sum
				var expSum float64
				for e := 0; e < embedDim; e++ {
					val := t.At(b, s, e)
					expSum += math.Exp(val - maxVal)
				}

				// Normalize
				for e := 0; e < embedDim; e++ {
					val := t.At(b, s, e)
					exp := math.Exp(val - maxVal)
					result.Set(b, s, e, exp/expSum)
				}
			}
		}
	}

	return result
}

// GELU computes the Gaussian Error Linear Unit activation
// Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GELU(t *tensor.Tensor) *tensor.Tensor {
	dims := t.Dimensions()
	result := tensor.NewTensor(dims[0], dims[1], dims[2])

	sqrt2OverPi := math.Sqrt(2.0 / math.Pi)
	coef := 0.044715

	batchSize := dims[0]
	seqLen := dims[1]
	embedDim := dims[2]

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for e := 0; e < embedDim; e++ {
				x := t.At(b, s, e)
				gelu := 0.5 * x * (1 + math.Tanh(sqrt2OverPi*(x+coef*x*x*x)))
				result.Set(b, s, e, gelu)
			}
		}
	}

	return result
}

// LayerNorm applies Layer Normalization
// Formula: (x - mean) / sqrt(var + eps) * gamma + beta
func LayerNorm(t *tensor.Tensor, gamma, beta *tensor.Tensor, eps float64) *tensor.Tensor {
	dims := t.Dimensions()
	result := tensor.NewTensor(dims[0], dims[1], dims[2])

	batchSize := dims[0]
	seqLen := dims[1]
	embedDim := dims[2]

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			// Compute mean
			var sum float64
			for e := 0; e < embedDim; e++ {
				sum += t.At(b, s, e)
			}
			mean := sum / float64(embedDim)

			// Compute variance
			var varSum float64
			for e := 0; e < embedDim; e++ {
				diff := t.At(b, s, e) - mean
				varSum += diff * diff
			}
			variance := varSum / float64(embedDim)

			// Normalize
			for e := 0; e < embedDim; e++ {
				x := t.At(b, s, e)
				norm := (x - mean) / math.Sqrt(variance+eps)
				
				var g, bet float64
				if gamma != nil {
					g = gamma.At(0, 0, e)
				} else {
					g = 1.0
				}
				if beta != nil {
					bet = beta.At(0, 0, e)
				}
				
				result.Set(b, s, e, norm*g+bet)
			}
		}
	}

	return result
}

// Dropout applies inverted dropout. When training=false or rate=0, returns input unchanged.
// When training=true, zeroes out a fraction of elements and scales the rest by 1/(1-rate).
func Dropout(x *tensor.Tensor, rate float64, training bool) *tensor.Tensor {
	if rate == 0 || !training {
		return x
	}
	scale := 1.0 / (1.0 - rate)
	result := tensor.NewTensor(x.Batch(), x.Seq(), x.Embed())
	for i, v := range x.Data {
		if rand.Float64() < rate {
			result.Data[i] = 0
		} else {
			result.Data[i] = v * scale
		}
	}
	return result
}

// CreateCausalMask creates a causal mask for masked attention
// Upper triangle (where j > i) is set to -inf
// Returns tensor of shape (1, size, size)
func CreateCausalMask(size int) *tensor.Tensor {
	result := tensor.NewTensor(1, size, size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if j > i {
				result.Set(0, i, j, -math.Inf(1))
			} else {
				result.Set(0, i, j, 0.0)
			}
		}
	}
	return result
}