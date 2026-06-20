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
	n_heads    int
	DropRate   float64
}

func NewSelfAttention(d_model int, n_heads int, dropRate float64) *SelfAttention {
	if d_model%n_heads != 0 {
		panic("d_model must be divisible by n_heads")
	}
	return &SelfAttention{
		Wq:       linear.NewLinear(d_model, d_model, true),
		Wk:       linear.NewLinear(d_model, d_model, true),
		Wv:       linear.NewLinear(d_model, d_model, true),
		Wo:       linear.NewLinear(d_model, d_model, true),
		d_k:      d_model / n_heads,
		n_heads:  n_heads,
		DropRate: dropRate,
	}
}

func (sa *SelfAttention) extractHead(t *tensor.Tensor, start int) *tensor.Tensor {
	batch, seq := t.Batch(), t.Seq()
	result := tensor.NewTensor(batch, seq, sa.d_k)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for e := 0; e < sa.d_k; e++ {
				result.Set(b, s, e, t.At(b, s, start+e))
			}
		}
	}
	return result
}

func (sa *SelfAttention) Forward(x *tensor.Tensor, mask *tensor.Tensor) *tensor.Tensor {
	q := sa.Wq.Forward(x)
	k := sa.Wk.Forward(x)
	v := sa.Wv.Forward(x)

	batch, seq, dModel := x.Batch(), x.Seq(), x.Embed()
	scale := 1.0 / math.Sqrt(float64(sa.d_k))

	headOutputs := make([]*tensor.Tensor, sa.n_heads)
	for h := 0; h < sa.n_heads; h++ {
		start := h * sa.d_k
		qHead := sa.extractHead(q, start)
		kHead := sa.extractHead(k, start)
		vHead := sa.extractHead(v, start)

		scores := qHead.MatMul(kHead.Transpose())
		scaledScores := scores.Scale(scale)
		if mask != nil {
			scaledScores = scaledScores.Add(mask)
		}
		weights := llmmath.Softmax(scaledScores, -1)
		weights = llmmath.Dropout(weights, sa.DropRate, false)
		headOutputs[h] = weights.MatMul(vHead)
	}

	result := tensor.NewTensor(batch, seq, dModel)
	for h := 0; h < sa.n_heads; h++ {
		start := h * sa.d_k
		headOut := headOutputs[h]
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				for e := 0; e < sa.d_k; e++ {
					result.Set(b, s, start+e, headOut.At(b, s, e))
				}
			}
		}
	}
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

	batch, seq, dModel := x.Batch(), x.Seq(), x.Embed()
	scale := 1.0 / math.Sqrt(float64(sa.d_k))

	headOutputs := make([]*tensor.Tensor, sa.n_heads)
	for h := 0; h < sa.n_heads; h++ {
		start := h * sa.d_k
		qHead := sa.extractHead(q, start)
		kHead := sa.extractHead(k, start)
		vHead := sa.extractHead(v, start)

		scores := qHead.MatMul(kHead.Transpose())
		scaledScores := scores.Scale(scale)
		if mask != nil {
			scaledScores = scaledScores.Add(mask)
		}
		weights := llmmath.Softmax(scaledScores, -1)
		weights = llmmath.Dropout(weights, sa.DropRate, false)
		headOutputs[h] = weights.MatMul(vHead)
	}

	result := tensor.NewTensor(batch, seq, dModel)
	for h := 0; h < sa.n_heads; h++ {
		start := h * sa.d_k
		headOut := headOutputs[h]
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				for e := 0; e < sa.d_k; e++ {
					result.Set(b, s, start+e, headOut.At(b, s, e))
				}
			}
		}
	}
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


