package weights

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/llm/model"
	"github.com/llm/tensor"
)

type WeightData struct {
	Shape []int       `json:"shape"`
	Data  []float64   `json:"data"`
}

type WeightMap map[string]WeightData

func LoadWeightsJSON(gpt *model.GPTModel, path string, strict bool) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	var weights WeightMap
	if err := json.Unmarshal(data, &weights); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	params := gpt.Parameters()
	
	for key, tensorVal := range params {
		weightData, ok := weights[key]
		if !ok {
			if strict {
				return fmt.Errorf("missing weight in JSON: %s", key)
			}
			fmt.Printf("Warning: No weight found in JSON for key: %s\n", key)
			continue
		}

		if !validateShape(tensorVal, weightData.Shape) {
			return fmt.Errorf("shape mismatch for key %s: expected %v, got %v", key, tensorVal.Dimensions(), weightData.Shape)
		}

		copyDataToTensor(tensorVal, weightData.Data)
	}

	return nil
}

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

func SaveWeightsJSON(gpt *model.GPTModel, path string) error {
	params := gpt.Parameters()
	weights := make(WeightMap)

	for key, t := range params {
		dims := t.Dimensions()
		shape := []int{dims[0], dims[1], dims[2]}
		data := make([]float64, 0, dims[0]*dims[1]*dims[2])
		
		for b := 0; b < dims[0]; b++ {
			for s := 0; s < dims[1]; s++ {
				for e := 0; e < dims[2]; e++ {
					data = append(data, t.At(b, s, e))
				}
			}
		}
		
		weights[key] = WeightData{
			Shape: shape,
			Data:  data,
		}
	}

	data, err := json.MarshalIndent(weights, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}
