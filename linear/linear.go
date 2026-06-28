package linear

import (
	"runtime"
	"sync"
	"time"

	"github.com/llm/internal/benchprof"
	"github.com/llm/tensor"
)

type Linear struct {
	InFeatures  int
	OutFeatures int
	HasBias     bool
	Weight     *tensor.Tensor
	Bias       *tensor.Tensor
}

func NewLinear(inFeatures, outFeatures int, hasBias bool) *Linear {
	l := &Linear{
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
		HasBias:     hasBias,
	}

	l.Weight = tensor.NewTensor(1, outFeatures, inFeatures)
	for i := 0; i < outFeatures; i++ {
		for j := 0; j < inFeatures; j++ {
			l.Weight.Set(0, i, j, float64(i)*0.01)
		}
	}

	if hasBias {
		l.Bias = tensor.NewTensor(1, 1, outFeatures)
		for i := 0; i < outFeatures; i++ {
			l.Bias.Set(0, 0, i, 1.0)
		}
	}

	return l
}

func (l *Linear) Forward(input *tensor.Tensor) *tensor.Tensor {
	dims := input.Dimensions()
	batch := dims[0]
	seq := dims[1]

	if benchprof.Enabled() {
		defer func(start time.Time) {
			benchprof.RecordLinearForward(time.Since(start), batch, seq, l.InFeatures, l.OutFeatures)
		}(time.Now())
	}

	result := tensor.NewTensor(batch, seq, l.OutFeatures)

	var biasSlice []float64
	if l.HasBias {
		biasSlice = l.Bias.Row(0, 0)
	}

	opCount := seq * l.OutFeatures * l.InFeatures
	if opCount < 500000 {
		for b := 0; b < batch; b++ {
			for s := 0; s < seq; s++ {
				inp := input.Row(b, s)
				out := result.Row(b, s)
				for o := 0; o < l.OutFeatures; o++ {
					var sum float64
					w := l.Weight.Row(0, o)
					for i := 0; i < l.InFeatures; i++ {
						sum += inp[i] * w[i]
					}
					if biasSlice != nil {
						sum += biasSlice[o]
					}
					out[o] = sum
				}
			}
		}
		return result
	}

	nw := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg.Add(1)
		startO := w * l.OutFeatures / nw
		endO := (w + 1) * l.OutFeatures / nw
		go func(sO, eO int) {
			defer wg.Done()
			for b := 0; b < batch; b++ {
				for s := 0; s < seq; s++ {
					inp := input.Row(b, s)
					out := result.Row(b, s)
					for o := sO; o < eO; o++ {
						var sum float64
						w := l.Weight.Row(0, o)
						for i := 0; i < l.InFeatures; i++ {
							sum += inp[i] * w[i]
						}
						if biasSlice != nil {
							sum += biasSlice[o]
						}
						out[o] = sum
					}
				}
			}
		}(startO, endO)
	}
	wg.Wait()

	return result
}

func (l *Linear) Parameters() map[string]*tensor.Tensor {
	params := make(map[string]*tensor.Tensor)
	params["Weight"] = l.Weight
	if l.HasBias {
		params["Bias"] = l.Bias
	}
	return params
}