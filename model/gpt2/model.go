package gpt2

import (
	"fmt"

	"github.com/llm/config"
	"github.com/llm/linear"
	"github.com/llm/math"
	"github.com/llm/model"
	"github.com/llm/tensor"
)

type Model struct {
	Cfg            config.Config
	Embeddings     *Embeddings
	Blocks         []*TransformerBlock
	FinalNormGamma *tensor.Tensor
	FinalNormBeta  *tensor.Tensor
	OutputProj     *linear.Linear
}

func NewModel(cfg config.Config) *Model {
	blocks := make([]*TransformerBlock, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		blocks[i] = NewTransformerBlock(cfg.EmbDim, cfg.NHeads, cfg.DropRate)
	}

	finalNormGamma := tensor.NewTensor(1, 1, cfg.EmbDim)
	finalNormBeta := tensor.NewTensor(1, 1, cfg.EmbDim)
	for i := 0; i < cfg.EmbDim; i++ {
		finalNormGamma.Set(0, 0, i, 1.0)
	}

	m := &Model{
		Cfg:            cfg,
		Embeddings:     NewEmbeddings(cfg.VocabSize, cfg.ContextLen, cfg.EmbDim),
		Blocks:         blocks,
		FinalNormGamma: finalNormGamma,
		FinalNormBeta:  finalNormBeta,
		OutputProj:     linear.NewLinear(cfg.EmbDim, cfg.VocabSize, false),
	}
	// Weight tying: share token embedding with output projection.
	// Matches OpenAI's GPT-2 architecture (logits = h @ wte^T).
	m.OutputProj.Weight = m.Embeddings.TokenEmbedding
	return m
}

func (m *Model) Forward(tokenIDs []int) *tensor.Tensor {
	// 1. Embedding
	x := m.Embeddings.Forward(tokenIDs, 0)

	// 2. Dropout on embeddings (training only)
	x = math.Dropout(x, m.Cfg.DropRate, false)

	// 3. Transformer Blocks with causal mask
	mask := math.CreateCausalMask(len(tokenIDs))
	for _, block := range m.Blocks {
		x = block.Forward(x, mask)
	}

	// 4. Final LayerNorm
	x = math.LayerNorm(x, m.FinalNormGamma, m.FinalNormBeta, 1e-5)

	// 5. Output Projection
	logits := m.OutputProj.Forward(x)

	return logits
}

// ForwardWithCache processes tokenIDs through the model, accepting and returning
// a KV cache. Pass pastCache=nil for the first call (prefill); subsequent calls
// pass the cache returned by the previous call (decode).
func (m *Model) ForwardWithCache(tokenIDs []int, pastCache *model.KVCache) (*tensor.Tensor, *model.KVCache) {
	startPos := 0
	if pastCache != nil {
		startPos = pastCache.SeqLen
	}
	x := m.Embeddings.Forward(tokenIDs, startPos)
	x = math.Dropout(x, m.Cfg.DropRate, false)

	newCache := &model.KVCache{
		Keys:   make([]*tensor.Tensor, len(m.Blocks)),
		Values: make([]*tensor.Tensor, len(m.Blocks)),
		SeqLen: startPos + len(tokenIDs),
	}

	var mask *tensor.Tensor
	if pastCache == nil {
		// Prefill: apply causal mask to prevent attending to future tokens
		mask = math.CreateCausalMask(len(tokenIDs))
	}

	for i, block := range m.Blocks {
		var pastK, pastV *tensor.Tensor
		if pastCache != nil {
			pastK = pastCache.Keys[i]
			pastV = pastCache.Values[i]
		}
		x, newCache.Keys[i], newCache.Values[i] = block.ForwardWithCache(x, mask, pastK, pastV)
	}

	x = math.LayerNorm(x, m.FinalNormGamma, m.FinalNormBeta, 1e-5)
	logits := m.OutputProj.Forward(x)

	return logits, newCache
}

func (m *Model) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	
	for k, v := range m.Embeddings.Parameters() {
		params["Embeddings."+k] = v
	}
	
	for i, block := range m.Blocks {
		for k, v := range block.Parameters() {
			params["Blocks."+fmt.Sprintf("%d", i)+"."+k] = v
		}
	}
	
	params["FinalNorm.Gamma"] = m.FinalNormGamma
	params["FinalNorm.Beta"] = m.FinalNormBeta
	
	for k, v := range m.OutputProj.Parameters() {
		if k == "Weight" {
			continue // shared with Embeddings.TokenEmbedding
		}
		params["OutputProj."+k] = v
	}
	
	return params
}
