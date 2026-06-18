package weights

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"github.com/llm/model"
	"github.com/llm/tensor"
)

const (
	MagicNumber = 0x4C4C4D00
	Version     = 1
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

	if strict {
		for key := range weights {
			if _, ok := params[key]; !ok {
				return fmt.Errorf("extra key in JSON: %s", key)
			}
		}
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

func SaveWeightsBinary(gpt *model.GPTModel, path string) error {
	params := gpt.Parameters()
	
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	
	if err := binary.Write(writer, binary.LittleEndian, uint32(MagicNumber)); err != nil {
		return fmt.Errorf("failed to write magic number: %w", err)
	}
	if err := binary.Write(writer, binary.LittleEndian, int32(Version)); err != nil {
		return fmt.Errorf("failed to write version: %w", err)
	}
	if err := binary.Write(writer, binary.LittleEndian, int32(len(keys))); err != nil {
		return fmt.Errorf("failed to write tensor count: %w", err)
	}

	for _, key := range keys {
		t := params[key]
		dims := t.Dimensions()
		
		keyBytes := []byte(key)
		if err := binary.Write(writer, binary.LittleEndian, int32(len(keyBytes))); err != nil {
			return fmt.Errorf("failed to write key length: %w", err)
		}
		if _, err := writer.Write(keyBytes); err != nil {
			return fmt.Errorf("failed to write key: %w", err)
		}
		
		if err := binary.Write(writer, binary.LittleEndian, int32(len(dims))); err != nil {
			return fmt.Errorf("failed to write dim count: %w", err)
		}
		for _, d := range dims {
			if err := binary.Write(writer, binary.LittleEndian, int32(d)); err != nil {
				return fmt.Errorf("failed to write dim: %w", err)
			}
		}
		
		totalElements := dims[0] * dims[1] * dims[2]
		data := make([]float32, totalElements)
		idx := 0
		for b := 0; b < dims[0]; b++ {
			for s := 0; s < dims[1]; s++ {
				for e := 0; e < dims[2]; e++ {
					data[idx] = float32(t.At(b, s, e))
					idx++
				}
			}
		}
		
		if err := binary.Write(writer, binary.LittleEndian, data); err != nil {
			return fmt.Errorf("failed to write data: %w", err)
		}
	}

	return writer.Flush()
}

func LoadWeightsBinary(gpt *model.GPTModel, path string, strict bool) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	
	var magic uint32
	if err := binary.Read(reader, binary.LittleEndian, &magic); err != nil {
		return fmt.Errorf("failed to read magic number: %w", err)
	}
	if magic != MagicNumber {
		return fmt.Errorf("invalid magic number: expected 0x%X, got 0x%X", MagicNumber, magic)
	}
	
	var ver int32
	if err := binary.Read(reader, binary.LittleEndian, &ver); err != nil {
		return fmt.Errorf("failed to read version: %w", err)
	}
	if ver != Version {
		return fmt.Errorf("unsupported version: expected %d, got %d", Version, ver)
	}
	
	var count int32
	if err := binary.Read(reader, binary.LittleEndian, &count); err != nil {
		return fmt.Errorf("failed to read tensor count: %w", err)
	}

	params := gpt.Parameters()
	loaded := make(map[string]bool, len(params))

	for i := int32(0); i < count; i++ {
		var keyLen int32
		if err := binary.Read(reader, binary.LittleEndian, &keyLen); err != nil {
			return fmt.Errorf("failed to read key length: %w", err)
		}
		keyBytes := make([]byte, keyLen)
		if _, err := reader.Read(keyBytes); err != nil {
			return fmt.Errorf("failed to read key: %w", err)
		}
		key := string(keyBytes)

		var dimCount int32
		if err := binary.Read(reader, binary.LittleEndian, &dimCount); err != nil {
			return fmt.Errorf("failed to read dim count: %w", err)
		}
		shape := make([]int, dimCount)
		for j := int32(0); j < dimCount; j++ {
			var d int32
			if err := binary.Read(reader, binary.LittleEndian, &d); err != nil {
				return fmt.Errorf("failed to read dim: %w", err)
			}
			shape[j] = int(d)
		}

		tensorVal, ok := params[key]
		if !ok {
			if strict {
				return fmt.Errorf("missing weight in binary: %s", key)
			}
			fmt.Printf("Warning: No weight found in model for key: %s\n", key)
			totalElements := 1
			for _, s := range shape {
				totalElements *= s
			}
			floatData := make([]float32, totalElements)
			binary.Read(reader, binary.LittleEndian, floatData)
			continue
		}

		if !validateShape(tensorVal, shape) {
			return fmt.Errorf("shape mismatch for key %s: expected %v, got %v", key, tensorVal.Dimensions(), shape)
		}

		dims := tensorVal.Dimensions()
		totalElements := dims[0] * dims[1] * dims[2]
		floatData := make([]float32, totalElements)
		if err := binary.Read(reader, binary.LittleEndian, floatData); err != nil {
			return fmt.Errorf("failed to read data: %w", err)
		}

		idx := 0
		for b := 0; b < dims[0]; b++ {
			for s := 0; s < dims[1]; s++ {
				for e := 0; e < dims[2]; e++ {
					tensorVal.Set(b, s, e, float64(floatData[idx]))
					idx++
				}
			}
		}
		loaded[key] = true
	}

	if strict {
		for key := range params {
			if !loaded[key] {
				return fmt.Errorf("missing weight in binary: %s", key)
			}
		}
	}

	return nil
}
