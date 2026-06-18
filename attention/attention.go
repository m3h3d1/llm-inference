package attention

import (
	"math"

	"github.com/llm/linear"
	llmmath "github.com/llm/math"
	"github.com/llm/tensor"
)

// MultiHeadAttention handles multiple attention heads
type MultiHeadAttention struct {
	nHeads   int
	dModel   int
	dK       int
	Wq, Wk, Wv *linear.Linear
	Wo       *linear.Linear
}

func NewMultiHeadAttention(dModel, nHeads int) *MultiHeadAttention {
	dk := dModel / nHeads
	return &MultiHeadAttention{
		nHeads: nHeads,
		dModel: dModel,
		dK:     dk,
		Wq:     linear.NewLinear(dModel, dModel, false),
		Wk:     linear.NewLinear(dModel, dModel, false),
		Wv:     linear.NewLinear(dModel, dModel, false),
		Wo:     linear.NewLinear(dModel, dModel, false),
	}
}

func (mha *MultiHeadAttention) Forward(x *tensor.Tensor, mask *tensor.Tensor) *tensor.Tensor {
	// 1. Project Q, K, V
	qAll := mha.Wq.Forward(x) // (batch, seq, embed)
	kAll := mha.Wk.Forward(x) // (batch, seq, embed)
	vAll := mha.Wv.Forward(x) // (batch, seq, embed)

	// 2. Split into heads and compute attention for each
	headResults := make([]*tensor.Tensor, mha.nHeads)
	
	for h := 0; h < mha.nHeads; h++ {
		// Extract head slices
		qHead := mha.extractHead(qAll, h)
		kHead := mha.extractHead(kAll, h)
		vHead := mha.extractHead(vAll, h)

		// Attention logic: softmax(QK^T / sqrt(dk))V
		scores := qHead.MatMul(kHead.Transpose())
		scaledScores := scores.Scale(1.0 / math.Sqrt(float64(mha.dK)))
		
		if mask != nil {
			scaledScores = scaledScores.Add(mask)
		}
		
		weights := llmmath.Softmax(scaledScores, -1)
		headResults[h] = weights.MatMul(vHead)
	}

	// 3. Concatenate heads
	concatenated := tensor.Concat(headResults) // (batch, seq, embed)

	// 4. Final projection
	return mha.Wo.Forward(concatenated)
}


func (mha *MultiHeadAttention) extractHead(t *tensor.Tensor, headIdx int) *tensor.Tensor {
	dims := t.Dimensions()
	batch, seq := dims[0], dims[1]
	dk := mha.dK
	
	result := tensor.NewTensor(batch, seq, dk)
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for e := 0; e < dk; e++ {
				result.Set(b, s, e, t.At(b, s, headIdx*dk+e))
			}
		}
	}
	return result
}

// SelfAttention handles the core attention logic
type SelfAttention struct {
	Wq, Wk, Wv *linear.Linear
	Wo         *linear.Linear
	d_k        int
	DropRate   float64
}

func NewSelfAttention(d_model int, dropRate float64) *SelfAttention {
	return &SelfAttention{
		Wq:       linear.NewLinear(d_model, d_model, false),
		Wk:       linear.NewLinear(d_model, d_model, false),
		Wv:       linear.NewLinear(d_model, d_model, false),
		Wo:       linear.NewLinear(d_model, d_model, false),
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

// SimpleAttention remains for backward compatibility or simple tests
func SimpleAttention(x *tensor.Tensor) *tensor.Tensor {
	embed := x.Dimensions()[2]
	dk := float64(embed)

	scores := x.MatMul(x.Transpose())
	scaledScores := scores.Scale(1.0 / math.Sqrt(dk))
	weights := llmmath.Softmax(scaledScores, -1)
	return weights.MatMul(x)
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

func (mha *MultiHeadAttention) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	for k, v := range mha.Wq.Parameters() {
		params["Wq."+k] = v
	}
	for k, v := range mha.Wk.Parameters() {
		params["Wk."+k] = v
	}
	for k, v := range mha.Wv.Parameters() {
		params["Wv."+k] = v
	}
	for k, v := range mha.Wo.Parameters() {
		params["Wo."+k] = v
	}
	return params
}
