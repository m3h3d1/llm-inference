package linear

import (
	"github.com/llm/tensor"
)

type Linear struct {
	InFeatures  int
	OutFeatures int
	HasBias     bool
	Weight     *tensor.Tensor
	Bias       *tensor.Tensor
}

func NewLinear(inFeatures, outFeatures int, hasBias bool) *Linear {
	l := &Linear{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		HasBias:     hasBias,
	}

	l.Weight = tensor.NewTensor(1, outFeatures, inFeatures)
	for i := 0; i < outFeatures; i++ {
		for j := 0; j < inFeatures; j++ {
			l.Weight.Set(0, i, j, float64(i)*0.01)
		}
	}

	if hasBias {
		l.Bias = tensor.NewTensor(1, 1, outFeatures)
		for i := 0; i < outFeatures; i++ {
			l.Bias.Set(0, 0, i, 1.0)
		}
	}

	return l
}

func (l *Linear) Forward(input *tensor.Tensor) *tensor.Tensor {
	dims := input.Dimensions()
	batch := dims[0]
	seq := dims[1]

	result := tensor.NewTensor(batch, seq, l.OutFeatures)

	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for o := 0; o < l.OutFeatures; o++ {
				var sum float64
				for i := 0; i < l.InFeatures; i++ {
					sum += input.At(b, s, i) * l.Weight.At(0, o, i)
				}
				if l.HasBias {
					sum += l.Bias.At(0, 0, o)
				}
				result.Set(b, s, o, sum)
			}
		}
	}

	return result
}

func (l *Linear) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{l.Weight}
	if l.HasBias {
		params = append(params, l.Bias)
	}
	return params
}