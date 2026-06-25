package model

import "github.com/llm/tensor"

// KVCache holds the cached Key and Value tensors for each transformer layer
// and tracks the total number of tokens processed (for position offset).
type KVCache struct {
	Keys   []*tensor.Tensor
	Values []*tensor.Tensor
	SeqLen int
}
