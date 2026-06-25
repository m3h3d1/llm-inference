package gpt2

import (
	"github.com/llm/linear"
	llmmath "github.com/llm/math"
	"github.com/llm/tensor"
)

type FeedForward struct {
	L1 *linear.Linear
	L2 *linear.Linear
}

func NewFeedForward(dModel, dFF int) *FeedForward {
	return &FeedForward{
		L1: linear.NewLinear(dModel, dFF, true),
		L2: linear.NewLinear(dFF, dModel, true),
	}
}

func (ff *FeedForward) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Linear 1 -> GELU -> Linear 2
	out1 := ff.L1.Forward(x)
	act := llmmath.GELU(out1)
	out2 := ff.L2.Forward(act)
	
	return out2
}

func (ff *FeedForward) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	for k, v := range ff.L1.Parameters() {
		params["Linear1."+k] = v
	}
	for k, v := range ff.L2.Parameters() {
		params["Linear2."+k] = v
	}
	return params
}
