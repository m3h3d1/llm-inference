package model

import (
	"github.com/llm/attention"
	"github.com/llm/math"
	"github.com/llm/tensor"
)

type TransformerBlock struct {
	Attention *attention.SelfAttention
	FFN       *FeedForward
	LN1Gamma  *tensor.Tensor
	LN1Beta   *tensor.Tensor
	LN2Gamma  *tensor.Tensor
	LN2Beta   *tensor.Tensor
	DropRate  float64
}

func NewTransformerBlock(dModel int, nHeads int, dropRate float64) *TransformerBlock {
	lN1Gamma := tensor.NewTensor(1, 1, dModel)
	lN1Beta := tensor.NewTensor(1, 1, dModel)
	lN2Gamma := tensor.NewTensor(1, 1, dModel)
	lN2Beta := tensor.NewTensor(1, 1, dModel)

	for i := 0; i < dModel; i++ {
		lN1Gamma.Set(0, 0, i, 1.0)
		lN2Gamma.Set(0, 0, i, 1.0)
	}

	return &TransformerBlock{
		Attention: attention.NewSelfAttention(dModel, nHeads, dropRate),
		FFN:       NewFeedForward(dModel, dModel*4),
		LN1Gamma:  lN1Gamma,
		LN1Beta:   lN1Beta,
		LN2Gamma:  lN2Gamma,
		LN2Beta:   lN2Beta,
		DropRate:  dropRate,
	}
}

func (tb *TransformerBlock) Forward(x *tensor.Tensor, mask *tensor.Tensor) *tensor.Tensor {
	// Pre-LN architecture: x = x + Attention(LN(x))

	// 1. Attention sub-layer
	norm1 := math.LayerNorm(x, tb.LN1Gamma, tb.LN1Beta, 1e-5)
	attnOut := tb.Attention.Forward(norm1, mask)
	attnOut = math.Dropout(attnOut, tb.DropRate, false)
	x = x.Add(attnOut) // Residual connection

	// 2. MLP sub-layer
	norm2 := math.LayerNorm(x, tb.LN2Gamma, tb.LN2Beta, 1e-5)
	mlpOut := tb.FFN.Forward(norm2)
	mlpOut = math.Dropout(mlpOut, tb.DropRate, false)
	x = x.Add(mlpOut) // Residual connection

	return x
}

func (tb *TransformerBlock) ForwardWithCache(x *tensor.Tensor, mask *tensor.Tensor, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	// Pre-LN architecture with KV cache

	// 1. Attention sub-layer
	norm1 := math.LayerNorm(x, tb.LN1Gamma, tb.LN1Beta, 1e-5)
	attnOut, newK, newV := tb.Attention.ForwardWithCache(norm1, mask, pastK, pastV)
	attnOut = math.Dropout(attnOut, tb.DropRate, false)
	x = x.Add(attnOut) // Residual connection

	// 2. MLP sub-layer
	norm2 := math.LayerNorm(x, tb.LN2Gamma, tb.LN2Beta, 1e-5)
	mlpOut := tb.FFN.Forward(norm2)
	mlpOut = math.Dropout(mlpOut, tb.DropRate, false)
	x = x.Add(mlpOut) // Residual connection

	return x, newK, newV
}

func (tb *TransformerBlock) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	
	params["LN1.Gamma"] = tb.LN1Gamma
	params["LN1.Beta"] = tb.LN1Beta
	params["LN2.Gamma"] = tb.LN2Gamma
	params["LN2.Beta"] = tb.LN2Beta
	
	for k, v := range tb.Attention.Parameters() {
		params["Attention."+k] = v
	}
	for k, v := range tb.FFN.Parameters() {
		params["FFN."+k] = v
	}
	
	return params
}
