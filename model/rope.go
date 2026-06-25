package model

import (
	"math"

	"github.com/llm/tensor"
)

type RopeTables struct {
	Cos       []float64
	Sin       []float64
	MaxSeqLen int
	RopeDim   int
}

func NewRopeTables(maxSeqLen int, ropeDim int, theta float64) *RopeTables {
	half := ropeDim / 2
	cos := make([]float64, maxSeqLen*half)
	sin := make([]float64, maxSeqLen*half)
	for pos := 0; pos < maxSeqLen; pos++ {
		for k := 0; k < half; k++ {
			freq := 1.0 / math.Pow(theta, float64(2*k)/float64(ropeDim))
			angle := float64(pos) * freq
			idx := pos*half + k
			cos[idx] = math.Cos(angle)
			sin[idx] = math.Sin(angle)
		}
	}
	return &RopeTables{Cos: cos, Sin: sin, MaxSeqLen: maxSeqLen, RopeDim: ropeDim}
}

func applyRoPE(t *tensor.Tensor, tables *RopeTables, startPos int, headDim int) *tensor.Tensor {
	batch, seq, totalDim := t.Dimensions()[0], t.Dimensions()[1], t.Dimensions()[2]
	ropeDim := tables.RopeDim
	half := ropeDim / 2

	result := tensor.NewTensor(batch, seq, totalDim)

	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			pos := startPos + s
			cosRow := tables.Cos[pos*half:]
			sinRow := tables.Sin[pos*half:]

			for e := 0; e < totalDim; e++ {
				headStart := (e / headDim) * headDim
				offset := e - headStart

				if offset >= ropeDim {
					result.Set(b, s, e, t.At(b, s, e))
					continue
				}
				parity := offset % 2
				pairIdx := offset / 2
				x := t.At(b, s, e)
				if parity == 0 {
					x2 := t.At(b, s, e+1)
					result.Set(b, s, e, x*cosRow[pairIdx]-x2*sinRow[pairIdx])
				} else {
					xPrev := t.At(b, s, e-1)
					result.Set(b, s, e, x*cosRow[pairIdx]+xPrev*sinRow[pairIdx])
				}
			}
		}
	}
	return result
}
