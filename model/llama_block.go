package model

import (
	"github.com/llm/tensor"
)

type LlamaBlock struct {
	AttentionNorm *RMSNorm
	Attention     *LlamaAttention
	FFNNorm       *RMSNorm
	FFN           *SwiGLUFFN
}

func NewLlamaBlock(embDim, nHeads, nKVHeads, headDim, ropeDim, dff int, tables *RopeTables, rmsEps float64) *LlamaBlock {
	return &LlamaBlock{
		AttentionNorm: NewRMSNorm(embDim, rmsEps),
		Attention:     NewLlamaAttention(embDim, nHeads, nKVHeads, headDim, ropeDim, tables),
		FFNNorm:       NewRMSNorm(embDim, rmsEps),
		FFN:           NewSwiGLUFFN(embDim, dff),
	}
}

func (lb *LlamaBlock) Forward(x *tensor.Tensor, mask *tensor.Tensor, startPos int) *tensor.Tensor {
	residual := x
	h := lb.AttentionNorm.Forward(x)
	h = lb.Attention.Forward(h, mask, startPos)
	h = h.Add(residual)

	residual = h
	h = lb.FFNNorm.Forward(h)
	h = lb.FFN.Forward(h)
	h = h.Add(residual)
	return h
}

func (lb *LlamaBlock) ForwardWithCache(x *tensor.Tensor, mask *tensor.Tensor, pastK, pastV *tensor.Tensor, startPos int) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	residual := x
	h := lb.AttentionNorm.Forward(x)
	h, newK, newV := lb.Attention.ForwardWithCache(h, mask, pastK, pastV, startPos)
	h = h.Add(residual)

	residual = h
	h = lb.FFNNorm.Forward(h)
	h = lb.FFN.Forward(h)
	h = h.Add(residual)
	return h, newK, newV
}

func (lb *LlamaBlock) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	params["AttentionNorm.Weight"] = lb.AttentionNorm.Weight
	params["FFNNorm.Weight"] = lb.FFNNorm.Weight
	for k, v := range lb.Attention.Parameters() {
		params["Attention."+k] = v
	}
	for k, v := range lb.FFN.Parameters() {
		params["FFN."+k] = v
	}
	return params
}
