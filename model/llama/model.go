package llama

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/llm/config"
	"github.com/llm/internal/benchprof"
	llmmath "github.com/llm/math"
	"github.com/llm/model"
	"github.com/llm/tensor"
)

type Model struct {
	Cfg            config.Config
	TokenEmbedding *tensor.Tensor
	Blocks         []*LlamaBlock
	FinalNorm      *RMSNorm
	OutputWeight   *tensor.Tensor
	tables         *RopeTables
}

func NewModel(cfg config.Config) *Model {
	headDim := cfg.EmbDim / cfg.NHeads
	tables := NewRopeTables(cfg.ContextLen, cfg.RopeDim, cfg.RopeTheta)

	tokenEmb := tensor.NewTensor(cfg.VocabSize, 1, cfg.EmbDim)

	blocks := make([]*LlamaBlock, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		blocks[i] = NewLlamaBlock(cfg.EmbDim, cfg.NHeads, cfg.NKVHeads, headDim, cfg.DFF, tables, cfg.RmsNormEps)
	}

	finalNorm := NewRMSNorm(cfg.EmbDim, cfg.RmsNormEps)

	m := &Model{
		Cfg:            cfg,
		TokenEmbedding: tokenEmb,
		Blocks:         blocks,
		FinalNorm:      finalNorm,
		OutputWeight:   tokenEmb,
		tables:         tables,
	}

	return m
}

func (m *Model) embed(tokenIDs []int, startPos int) *tensor.Tensor {
	batch := 1
	seq := len(tokenIDs)
	result := tensor.NewTensor(batch, seq, m.Cfg.EmbDim)
	for s := 0; s < seq; s++ {
		tokenID := tokenIDs[s]
		for d := 0; d < m.Cfg.EmbDim; d++ {
			result.Set(0, s, d, m.TokenEmbedding.At(tokenID, 0, d))
		}
	}
	return result
}

func (m *Model) Forward(tokenIDs []int) *tensor.Tensor {
	x := m.embed(tokenIDs, 0)
	mask := llmmath.CreateCausalMask(len(tokenIDs))

	for _, block := range m.Blocks {
		x = block.Forward(x, mask, 0)
	}

	x = m.FinalNorm.Forward(x)
	logits := outputLogits(x, m.OutputWeight)
	return logits
}

func (m *Model) ForwardWithCache(tokenIDs []int, pastCache *model.KVCache) (*tensor.Tensor, *model.KVCache) {
	startPos := 0
	if pastCache != nil {
		startPos = pastCache.SeqLen
	}

	x := m.embed(tokenIDs, startPos)

	newCache := &model.KVCache{
		Keys:   make([]*tensor.Tensor, len(m.Blocks)),
		Values: make([]*tensor.Tensor, len(m.Blocks)),
		SeqLen: startPos + len(tokenIDs),
	}

	var mask *tensor.Tensor
	if pastCache == nil {
		mask = llmmath.CreateCausalMask(len(tokenIDs))
	} else if len(tokenIDs) > 1 {
		pastLen := pastCache.SeqLen
		newLen := len(tokenIDs)
		totalLen := pastLen + newLen
		mask = tensor.NewTensor(1, newLen, totalLen)
		for i := 0; i < newLen; i++ {
			for j := 0; j < totalLen; j++ {
				if j < pastLen || j <= pastLen+i {
					mask.Set(0, i, j, 0.0)
				} else {
					mask.Set(0, i, j, -math.Inf(1))
				}
			}
		}
	}

	for i, block := range m.Blocks {
		var pastK, pastV *tensor.Tensor
		if pastCache != nil {
			pastK = pastCache.Keys[i]
			pastV = pastCache.Values[i]
		}
		x, newCache.Keys[i], newCache.Values[i] = block.ForwardWithCache(x, mask, pastK, pastV, startPos)
	}

	x = m.FinalNorm.Forward(x)
	logits := outputLogits(x, m.OutputWeight)
	return logits, newCache
}

func outputLogits(x *tensor.Tensor, outputWeight *tensor.Tensor) *tensor.Tensor {
	dims := x.Dimensions()
	batch, seq, embDim := dims[0], dims[1], dims[2]
	vocabSize := outputWeight.Dimensions()[0]

	if benchprof.Enabled() {
		defer func(start time.Time) {
			benchprof.RecordOutputLogits(time.Since(start), batch, seq, embDim, vocabSize)
		}(time.Now())
	}

	result := tensor.NewTensor(batch, seq, vocabSize)

	nw := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		startV := w * vocabSize / nw
		endV := (w + 1) * vocabSize / nw
		go func(sV, eV int) {
			defer wg.Done()
			for b := 0; b < batch; b++ {
				for s := 0; s < seq; s++ {
					inp := x.Row(b, s)
					out := result.Row(b, s)
					for v := sV; v < eV; v++ {
						var sum float64
						w := outputWeight.Row(v, 0)
						for e := 0; e < embDim; e++ {
							sum += inp[e] * w[e]
						}
						out[v] = sum
					}
				}
			}
		}(startV, endV)
	}
	wg.Wait()

	return result
}

func (m *Model) SetParameter(name string, data *tensor.Tensor) {
	res := func(d *tensor.Tensor, batch, seq, embed int) *tensor.Tensor {
		r := d.Reshape(batch, seq, embed)
		if r == nil {
			panic("shape mismatch loading " + name)
		}
		return r
	}

	switch name {
	case "token_embd.weight":
		m.TokenEmbedding = res(data, m.Cfg.VocabSize, 1, m.Cfg.EmbDim)
		m.OutputWeight = m.TokenEmbedding
		return
	case "output.weight":
		return
	case "output_norm.weight":
		m.FinalNorm.Weight = res(data, 1, 1, m.Cfg.EmbDim)
		return
	}

	if len(name) < 4 || name[:4] != "blk." {
		return
	}
	rest := name[4:]
	dotIdx := -1
	for i, c := range rest {
		if c == '.' {
			dotIdx = i
			break
		}
	}
	if dotIdx < 0 {
		return
	}
	layer := 0
	for _, c := range rest[:dotIdx] {
		layer = layer*10 + int(c-'0')
	}
	component := rest[dotIdx+1:]

	if layer < 0 || layer >= len(m.Blocks) {
		return
	}
	block := m.Blocks[layer]
	hd := m.Cfg.EmbDim / m.Cfg.NHeads

	switch component {
	case "attn_norm.weight":
		block.AttentionNorm.Weight = res(data, 1, 1, m.Cfg.EmbDim)
	case "ffn_norm.weight":
		block.FFNNorm.Weight = res(data, 1, 1, m.Cfg.EmbDim)
	case "attn_q.weight":
		block.Attention.Wq.Weight = res(data, 1, m.Cfg.NHeads*hd, m.Cfg.EmbDim)
	case "attn_k.weight":
		block.Attention.Wk.Weight = res(data, 1, m.Cfg.NKVHeads*hd, m.Cfg.EmbDim)
	case "attn_v.weight":
		block.Attention.Wv.Weight = res(data, 1, m.Cfg.NKVHeads*hd, m.Cfg.EmbDim)
	case "attn_output.weight":
		block.Attention.Wo.Weight = res(data, 1, m.Cfg.EmbDim, m.Cfg.NHeads*hd)
	case "ffn_gate.weight":
		block.FFN.Gate.Weight = res(data, 1, m.Cfg.DFF, m.Cfg.EmbDim)
	case "ffn_up.weight":
		block.FFN.Up.Weight = res(data, 1, m.Cfg.DFF, m.Cfg.EmbDim)
	case "ffn_down.weight":
		block.FFN.Down.Weight = res(data, 1, m.Cfg.EmbDim, m.Cfg.DFF)
	}
}

func (m *Model) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	params["TokenEmbedding"] = m.TokenEmbedding
	for i, block := range m.Blocks {
		for k, v := range block.Parameters() {
			params["Blocks."+fmt.Sprintf("%d", i)+"."+k] = v
		}
	}
	params["FinalNorm.Weight"] = m.FinalNorm.Weight
	return params
}
