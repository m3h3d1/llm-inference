package model

import (
	"math"

	"github.com/llm/tensor"
)

type RMSNorm struct {
	Weight *tensor.Tensor
	Eps    float64
}

func NewRMSNorm(size int, eps float64) *RMSNorm {
	w := tensor.Ones(1, 1, size)
	return &RMSNorm{Weight: w, Eps: eps}
}

func (r *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	dims := x.Dimensions()
	result := tensor.NewTensor(dims[0], dims[1], dims[2])
	for b := 0; b < dims[0]; b++ {
		for s := 0; s < dims[1]; s++ {
			var sumSq float64
			for e := 0; e < dims[2]; e++ {
				v := x.At(b, s, e)
				sumSq += v * v
			}
			rms := math.Sqrt(sumSq/float64(dims[2]) + r.Eps)
			for e := 0; e < dims[2]; e++ {
				result.Set(b, s, e, x.At(b, s, e)/rms*r.Weight.At(0, 0, e))
			}
		}
	}
	return result
}
