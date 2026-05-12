package main

import (
	"fmt"

	"github.com/llm/config"
	"github.com/llm/model"
	"github.com/llm/weights"
)

func main() {
	cfg := config.Config{
		VocabSize:  1000,
		ContextLen: 32,
		EmbDim:     32,
		NHeads:     4,
		NLayers:    2,
		DropRate:   0.0,
		QKVBias:    false,
	}
	gpt := model.NewGPTModel(cfg)

	params := gpt.Parameters()
	fmt.Printf("Total parameters: %d\n", len(params))

	jsonPath := "test_weights.json"
	fmt.Printf("Generating dummy weights (JSON) to: %s\n", jsonPath)
	if err := weights.SaveWeightsJSON(gpt, jsonPath); err != nil {
		fmt.Printf("Error saving JSON weights: %v\n", err)
		return
	}
	fmt.Println("JSON weights generated successfully!")

	binPath := "test_weights.bin"
	fmt.Printf("Generating dummy weights (BIN) to: %s\n", binPath)
	if err := weights.SaveWeightsBinary(gpt, binPath); err != nil {
		fmt.Printf("Error saving binary weights: %v\n", err)
		return
	}
	fmt.Println("Binary weights generated successfully!")
}
