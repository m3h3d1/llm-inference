package llama

import (
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/llm/internal/benchprof"
	"github.com/llm/linear"
	llmmath "github.com/llm/math"
	"github.com/llm/tensor"
)

type LlamaAttention struct {
	Wq       *linear.Linear
	Wk       *linear.Linear
	Wv       *linear.Linear
	Wo       *linear.Linear
	NHeads   int
	NKVHeads int
	HeadDim  int
	Tables   *RopeTables
}

func NewLlamaAttention(embDim, nHeads, nKVHeads, headDim, ropeDim int, tables *RopeTables) *LlamaAttention {
	return &LlamaAttention{
		Wq:       linear.NewLinear(embDim, nHeads*headDim, false),
		Wk:       linear.NewLinear(embDim, nKVHeads*headDim, false),
		Wv:       linear.NewLinear(embDim, nKVHeads*headDim, false),
		Wo:       linear.NewLinear(nHeads*headDim, embDim, false),
		NHeads:   nHeads,
		NKVHeads: nKVHeads,
		HeadDim:  headDim,
		Tables:   tables,
	}
}

func (la *LlamaAttention) extractHead(t *tensor.Tensor, headIdx int, headDim int) *tensor.Tensor {
	batch, seq := t.Batch(), t.Seq()
	result := tensor.NewTensor(batch, seq, headDim)
	start := headIdx * headDim
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for e := 0; e < headDim; e++ {
				result.Set(b, s, e, t.At(b, s, start+e))
			}
		}
	}
	return result
}

func (la *LlamaAttention) mergeHeads(heads []*tensor.Tensor) *tensor.Tensor {
	batch, seq := heads[0].Batch(), heads[0].Seq()
	headDim := heads[0].Embed()
	result := tensor.NewTensor(batch, seq, len(heads)*headDim)
	for h, head := range heads {
		start := h * headDim
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				for e := 0; e < headDim; e++ {
					result.Set(b, s, start+e, head.At(b, s, e))
				}
			}
		}
	}
	return result
}

func (la *LlamaAttention) Forward(x *tensor.Tensor, mask *tensor.Tensor, startPos int) *tensor.Tensor {
	if benchprof.Enabled() {
		defer func(start time.Time) {
			benchprof.RecordAttentionForward(time.Since(start), x.Seq(), la.NHeads, la.NKVHeads, la.HeadDim)
		}(time.Now())
	}

	q := applyRoPE(la.Wq.Forward(x), la.Tables, startPos, la.HeadDim)
	k := applyRoPE(la.Wk.Forward(x), la.Tables, startPos, la.HeadDim)
	v := la.Wv.Forward(x)

	nGroups := la.NHeads / la.NKVHeads
	scale := 1.0 / math.Sqrt(float64(la.HeadDim))

	headOutputs := make([]*tensor.Tensor, la.NHeads)

	nw := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		startH := w * la.NHeads / nw
		endH := (w + 1) * la.NHeads / nw
		go func(sH, eH int) {
			defer wg.Done()
			for h := sH; h < eH; h++ {
				kvIdx := h / nGroups

				qHead := la.extractHead(q, h, la.HeadDim)
				kHead := la.extractHead(k, kvIdx, la.HeadDim)
				vHead := la.extractHead(v, kvIdx, la.HeadDim)

				scores := qHead.MatMul(kHead.Transpose())
				scaledScores := scores.Scale(scale)
				if mask != nil {
					scaledScores = scaledScores.Add(mask)
				}
				weights := llmmath.Softmax(scaledScores, -1)
				headOutputs[h] = weights.MatMul(vHead)
			}
		}(startH, endH)
	}
	wg.Wait()

	return la.Wo.Forward(la.mergeHeads(headOutputs))
}

func (la *LlamaAttention) ForwardWithCache(x *tensor.Tensor, mask *tensor.Tensor, pastK, pastV *tensor.Tensor, startPos int) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	var k, v *tensor.Tensor

	if benchprof.Enabled() {
		defer func(start time.Time) {
			benchprof.RecordAttentionForwardWithCache(time.Since(start), x.Seq(), k.Seq(), la.NHeads, la.NKVHeads, la.HeadDim)
		}(time.Now())
	}

	q := applyRoPE(la.Wq.Forward(x), la.Tables, startPos, la.HeadDim)
	k = applyRoPE(la.Wk.Forward(x), la.Tables, startPos, la.HeadDim)
	v = la.Wv.Forward(x)

	if pastK != nil {
		k = tensor.ConcatSeq([]*tensor.Tensor{pastK, k})
		v = tensor.ConcatSeq([]*tensor.Tensor{pastV, v})
	}

	nGroups := la.NHeads / la.NKVHeads
	scale := 1.0 / math.Sqrt(float64(la.HeadDim))

	headOutputs := make([]*tensor.Tensor, la.NHeads)

	nw := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		startH := w * la.NHeads / nw
		endH := (w + 1) * la.NHeads / nw
		go func(sH, eH int) {
			defer wg.Done()
			for h := sH; h < eH; h++ {
				kvIdx := h / nGroups

				qHead := la.extractHead(q, h, la.HeadDim)
				kHead := la.extractHead(k, kvIdx, la.HeadDim)
				vHead := la.extractHead(v, kvIdx, la.HeadDim)

				scores := qHead.MatMul(kHead.Transpose())
				scaledScores := scores.Scale(scale)
				if mask != nil {
					scaledScores = scaledScores.Add(mask)
				}
				weights := llmmath.Softmax(scaledScores, -1)
				headOutputs[h] = weights.MatMul(vHead)
			}
		}(startH, endH)
	}
	wg.Wait()

	output := la.Wo.Forward(la.mergeHeads(headOutputs))
	return output, k, v
}

func (la *LlamaAttention) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	for k, v := range la.Wq.Parameters() {
		params["Wq."+k] = v
	}
	for k, v := range la.Wk.Parameters() {
		params["Wk."+k] = v
	}
	for k, v := range la.Wv.Parameters() {
		params["Wv."+k] = v
	}
	for k, v := range la.Wo.Parameters() {
		params["Wo."+k] = v
	}
	return params
}
