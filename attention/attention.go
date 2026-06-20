package attention

import (
	"math"

	"github.com/llm/linear"
	llmmath "github.com/llm/math"
	"github.com/llm/tensor"
)

// SelfAttention handles the core attention logic
type SelfAttention struct {
	Wq, Wk, Wv *linear.Linear
	Wo         *linear.Linear
	d_k        int
	DropRate   float64
}

func NewSelfAttention(d_model int, dropRate float64) *SelfAttention {
	return &SelfAttention{
		Wq:       linear.NewLinear(d_model, d_model, true),
		Wk:       linear.NewLinear(d_model, d_model, true),
		Wv:       linear.NewLinear(d_model, d_model, true),
		Wo:       linear.NewLinear(d_model, d_model, true),
		d_k:      d_model,
		DropRate: dropRate,
	}
}

func (sa *SelfAttention) Forward(x *tensor.Tensor, mask *tensor.Tensor) *tensor.Tensor {
	// 1. Project Q, K, V
	q := sa.Wq.Forward(x) // (batch, seq, embed)
	k := sa.Wk.Forward(x) // (batch, seq, embed)
	v := sa.Wv.Forward(x) // (batch, seq, embed)

	// 2. Compute scores: Q * K^T
	// Q: (batch, seq, embed), K^T: (batch, embed, seq)
	scores := q.MatMul(k.Transpose()) // (batch, seq, seq)

	// 3. Scale scores by 1/sqrt(dk)
	scaledScores := scores.Scale(1.0 / math.Sqrt(float64(sa.d_k)))

	// 4. Apply Causal Mask if provided
	if mask != nil {
		scaledScores = scaledScores.Add(mask)
	}

	// 5. Softmax to get attention weights
	weights := llmmath.Softmax(scaledScores, -1) // (batch, seq, seq)

	// 6. Apply dropout to attention weights (training only)
	weights = llmmath.Dropout(weights, sa.DropRate, false)

	// 7. Multiply weights by V
	// weights: (batch, seq, seq), V: (batch, seq, embed)
	result := weights.MatMul(v) // (batch, seq, embed)

	// 8. Output projection
	return sa.Wo.Forward(result)
}

// ForwardWithCache computes attention with optional past K,V cache.
// During decode (pastK != nil), no causal mask is needed since all
// cached positions precede the current token by construction.
// Returns (output, newK, newV) where newK/V are the combined past + current.
func (sa *SelfAttention) ForwardWithCache(x *tensor.Tensor, mask *tensor.Tensor, pastK, pastV *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	q := sa.Wq.Forward(x)
	k := sa.Wk.Forward(x)
	v := sa.Wv.Forward(x)

	if pastK != nil {
		k = tensor.ConcatSeq([]*tensor.Tensor{pastK, k})
		v = tensor.ConcatSeq([]*tensor.Tensor{pastV, v})
	}

	scores := q.MatMul(k.Transpose())
	scaledScores := scores.Scale(1.0 / math.Sqrt(float64(sa.d_k)))

	if mask != nil {
		scaledScores = scaledScores.Add(mask)
	}

	weights := llmmath.Softmax(scaledScores, -1)
	weights = llmmath.Dropout(weights, sa.DropRate, false)
	result := weights.MatMul(v)
	output := sa.Wo.Forward(result)

	return output, k, v
}

func (sa *SelfAttention) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	for k, v := range sa.Wq.Parameters() {
		params["Wq."+k] = v
	}
	for k, v := range sa.Wk.Parameters() {
		params["Wk."+k] = v
	}
	for k, v := range sa.Wv.Parameters() {
		params["Wv."+k] = v
	}
	for k, v := range sa.Wo.Parameters() {
		params["Wo."+k] = v
	}
	return params
}


