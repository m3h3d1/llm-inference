package model

import (
	"github.com/llm/config"
	"github.com/llm/linear"
	"github.com/llm/math"
	"github.com/llm/tensor"
)

type GPTModel struct {
	Cfg            config.Config
	Embeddings     *Embeddings
	Blocks         []*TransformerBlock
	FinalNormGamma *tensor.Tensor
	FinalNormBeta  *tensor.Tensor
	OutputProj     *linear.Linear
}

func NewGPTModel(cfg config.Config) *GPTModel {
	blocks := make([]*TransformerBlock, cfg.NLayers)
	for i := 0; i < cfg.NLayers; i++ {
		blocks[i] = NewTransformerBlock(cfg.EmbDim)
	}

	finalNormGamma := tensor.NewTensor(1, 1, cfg.EmbDim)
	finalNormBeta := tensor.NewTensor(1, 1, cfg.EmbDim)
	for i := 0; i < cfg.EmbDim; i++ {
		finalNormGamma.Set(0, 0, i, 1.0)
	}

	return &GPTModel{
		Cfg:            cfg,
		Embeddings:     NewEmbeddings(cfg.VocabSize, cfg.ContextLen, cfg.EmbDim),
		Blocks:         blocks,
		FinalNormGamma: finalNormGamma,
		FinalNormBeta:  finalNormBeta,
		OutputProj:     linear.NewLinear(cfg.EmbDim, cfg.VocabSize, false),
	}
}

func (m *GPTModel) Forward(tokenIDs []int) *tensor.Tensor {
	// 1. Embedding
	x := m.Embeddings.Forward(tokenIDs)

	// 2. Transformer Blocks
	for _, block := range m.Blocks {
		x = block.Forward(x, nil) // Using nil mask for simple forward
	}

	// 3. Final LayerNorm
	x = math.LayerNorm(x, m.FinalNormGamma, m.FinalNormBeta, 1e-5)

	// 4. Output Projection
	logits := m.OutputProj.Forward(x)

	return logits
}
