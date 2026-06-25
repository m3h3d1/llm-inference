package weights

import "github.com/llm/tensor"

const (
	MagicNumber = 0x4C4C4D00
	Version     = 1
)

type WeightData struct {
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}

type WeightMap map[string]WeightData

func validateShape(t *tensor.Tensor, shape []int) bool {
	dims := t.Dimensions()
	if len(dims) != len(shape) {
		return false
	}
	for i := 0; i < len(dims); i++ {
		if dims[i] != shape[i] {
			return false
		}
	}
	return true
}

func copyDataToTensor(t *tensor.Tensor, data []float64) {
	dims := t.Dimensions()
	idx := 0
	for b := 0; b < dims[0]; b++ {
		for s := 0; s < dims[1]; s++ {
			for e := 0; e < dims[2]; e++ {
				t.Set(b, s, e, data[idx])
				idx++
			}
		}
	}
}
