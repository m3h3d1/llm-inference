package model

import (
	"fmt"

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
		blocks[i] = NewTransformerBlock(cfg.EmbDim, cfg.DropRate)
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

	// 2. Dropout on embeddings (training only)
	x = math.Dropout(x, m.Cfg.DropRate, false)

	// 3. Transformer Blocks
	for _, block := range m.Blocks {
		x = block.Forward(x, nil) // Using nil mask for simple forward
	}

	// 4. Final LayerNorm
	x = math.LayerNorm(x, m.FinalNormGamma, m.FinalNormBeta, 1e-5)

	// 5. Output Projection
	logits := m.OutputProj.Forward(x)

	return logits
}

func (m *GPTModel) Parameters() map[string]*tensor.Tensor {
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
		params["OutputProj."+k] = v
	}
	
	return params
}
