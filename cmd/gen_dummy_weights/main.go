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

	outputPath := "test_weights.json"
	fmt.Printf("Generating dummy weights to: %s\n", outputPath)
	
	if err := weights.SaveWeightsJSON(gpt, outputPath); err != nil {
		fmt.Printf("Error saving weights: %v\n", err)
		return
	}

	fmt.Println("Dummy weights generated successfully!")
	
	params := gpt.Parameters()
	fmt.Printf("Total parameters: %d\n", len(params))
	for k := range params {
		fmt.Printf("  - %s\n", k)
	}
}
