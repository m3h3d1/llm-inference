package model

import (
	"math"

	"github.com/llm/tensor"
)

func SiLU(x *tensor.Tensor) *tensor.Tensor {
	dims := x.Dimensions()
	result := tensor.NewTensor(dims[0], dims[1], dims[2])
	for i, v := range x.Data {
		result.Data[i] = v / (1 + math.Exp(-v))
	}
	return result
}
