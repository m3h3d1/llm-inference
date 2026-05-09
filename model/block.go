package model

import (
	"github.com/llm/attention"
	"github.com/llm/math"
	"github.com/llm/tensor"
)

type TransformerBlock struct {
	Attention *attention.SelfAttention
	FFN       *FeedForward
	LN1       *tensor.Tensor // We'll use a simplified LayerNorm state or just call math.LayerNorm
	LN2       *tensor.Tensor
	dModel    int
}

func NewTransformerBlock(dModel int) *TransformerBlock {
	return &TransformerBlock{
		Attention: attention.NewSelfAttention(dModel),
		FFN:       NewFeedForward(dModel, dModel*4),
		dModel:    dModel,
	}
}

func (tb *TransformerBlock) Forward(x *tensor.Tensor, mask *tensor.Tensor) *tensor.Tensor {
	// Pre-LN architecture: x = x + Attention(LN(x))
	
	// 1. Attention sub-layer
	norm1 := math.LayerNorm(x, nil, nil, 1e-5)
	attnOut := tb.Attention.Forward(norm1, mask)
	x = x.Add(attnOut) // Residual connection

	// 2. MLP sub-layer
	norm2 := math.LayerNorm(x, nil, nil, 1e-5)
	mlpOut := tb.FFN.Forward(norm2)
	x = x.Add(mlpOut) // Residual connection

	return x
}
