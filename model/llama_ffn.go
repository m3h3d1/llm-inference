package model

import (
	"github.com/llm/linear"
	"github.com/llm/tensor"
)

type SwiGLUFFN struct {
	Gate *linear.Linear
	Up   *linear.Linear
	Down *linear.Linear
}

func NewSwiGLUFFN(embDim, dff int) *SwiGLUFFN {
	return &SwiGLUFFN{
		Gate: linear.NewLinear(embDim, dff, false),
		Up:   linear.NewLinear(embDim, dff, false),
		Down: linear.NewLinear(dff, embDim, false),
	}
}

func (ff *SwiGLUFFN) Forward(x *tensor.Tensor) *tensor.Tensor {
	gate := SiLU(ff.Gate.Forward(x))
	up := ff.Up.Forward(x)
	return ff.Down.Forward(gate.Mul(up))
}

func (ff *SwiGLUFFN) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	for k, v := range ff.Gate.Parameters() {
		params["Gate."+k] = v
	}
	for k, v := range ff.Up.Parameters() {
		params["Up."+k] = v
	}
	for k, v := range ff.Down.Parameters() {
		params["Down."+k] = v
	}
	return params
}
