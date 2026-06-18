package main

import (
	"flag"
	"fmt"

	"github.com/llm/config"
	"github.com/llm/inference"
	"github.com/llm/model"
	"github.com/llm/tokenizer"
	"github.com/llm/weights"
)

func main() {
	profile := flag.String("profile", "debug", "Model profile: debug (small, fast) or full (GPT-2 Small 124M)")
	weightsPath := flag.String("weights", "", "Path to the weights file")
	format := flag.String("format", "json", "Weight format: json or bin")
	prompt := flag.String("prompt", "The", "Input prompt for text generation")
	maxTokens := flag.Int("max_tokens", 30, "Maximum number of tokens to generate")
	strict := flag.Bool("strict", false, "Fail if weights are missing")

	flag.Parse()

	var cfg config.Config
	switch *profile {
	case "full":
		cfg = config.DefaultConfig
	case "debug":
		cfg = config.Config{
			VocabSize:  1000,
			ContextLen: 32,
			EmbDim:     32,
			NHeads:     4,
			NLayers:    2,
			DropRate:   0.0,
			QKVBias:    false,
		}
	default:
		fmt.Printf("Unknown profile: %s (use debug or full)\n", *profile)
		return
	}
	gpt := model.NewGPTModel(cfg)
	var tok *tokenizer.Tokenizer
	if cfg.VocabSize < 10000 {
		tok = tokenizer.NewMock()
	} else {
		var err error
		tok, err = tokenizer.NewFromFiles("assets/tokenizer/vocab.json", "assets/tokenizer/merges.txt")
		if err != nil {
			fmt.Printf("Error loading tokenizer: %v\n", err)
			return
		}
	}

	if *weightsPath != "" {
		fmt.Printf("Loading weights from: %s (format: %s)\n", *weightsPath, *format)
		var err error
		if *format == "bin" {
			err = weights.LoadWeightsBinary(gpt, *weightsPath, *strict)
		} else {
			err = weights.LoadWeightsJSON(gpt, *weightsPath, *strict)
		}
		if err != nil {
			fmt.Printf("Error loading weights: %v\n", err)
			return
		}
		fmt.Println("Weights loaded successfully")
	}

	fmt.Printf("Generating text with prompt: %s\n", *prompt)
	output := inference.Generate(cfg, gpt, tok, *prompt, *maxTokens)
	fmt.Printf("Generated text: %s\n", output)
}
