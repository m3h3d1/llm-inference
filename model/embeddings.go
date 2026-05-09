package model

import (
	"github.com/llm/tensor"
)

type Embeddings struct {
	TokenEmbedding     *tensor.Tensor
	PositionalEmbedding *tensor.Tensor
	VocabSize          int
	ContextLen         int
	EmbDim             int
}

func NewEmbeddings(vocabSize, contextLen, embDim int) *Embeddings {
	// Token embeddings: (vocabSize, 1, embDim)
	// We use (vocabSize, 1, embDim) to make it compatible with our 3D Tensor system
	tokenEmb := tensor.NewTensor(vocabSize, 1, embDim)
	
	// Initialize with small random-ish values (deterministic for this phase)
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embDim; j++ {
			tokenEmb.Set(i, 0, j, float64(i+j)*0.001)
		}
	}

	// Positional embeddings: (contextLen, 1, embDim)
	posEmb := tensor.NewTensor(contextLen, 1, embDim)
	for i := 0; i < contextLen; i++ {
		for j := 0; j < embDim; j++ {
			posEmb.Set(i, 0, j, float64(i+j)*0.001)
		}
	}

	return &Embeddings{
		TokenEmbedding:     tokenEmb,
		PositionalEmbedding: posEmb,
		VocabSize:          vocabSize,
		ContextLen:         contextLen,
		EmbDim:             embDim,
	}
}

func (e *Embeddings) Forward(tokenIDs []int) *tensor.Tensor {
	batch := 1 // Simplified for now
	seq := len(tokenIDs)
	
	result := tensor.NewTensor(batch, seq, e.EmbDim)

	for s := 0; s < seq; s++ {
		tokenID := tokenIDs[s]
		
		// 1. Get token embedding
		// tokenEmb is (vocabSize, 1, embDim), we want row tokenID
		for d := 0; d < e.EmbDim; d++ {
			val := e.TokenEmbedding.At(tokenID, 0, d)
			
			// 2. Add positional embedding
			// posEmb is (contextLen, 1, embDim)
			posVal := e.PositionalEmbedding.At(s, 0, d)
			
			result.Set(0, s, d, val+posVal)
		}
	}

	return result
}
