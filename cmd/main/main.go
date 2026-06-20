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
	profile := flag.String("profile", "debug", "Model profile: debug (mock, no weights), small (GPT-2 Small 124M), or medium (GPT-2 Medium 355M)")
	weightsPath := flag.String("weights", "", "Path to the weights file")
	format := flag.String("format", "json", "Weight format: json or bin")
	prompt := flag.String("prompt", "The", "Input prompt for text generation")
	maxTokens := flag.Int("max_tokens", 30, "Maximum number of tokens to generate")
	strict := flag.Bool("strict", false, "Fail if weights are missing")
	repPenalty := flag.Float64("repetition_penalty", 1.0, "Repetition penalty (>1.0 penalizes repeated tokens)")
	temperature := flag.Float64("temperature", 1.0, "Sampling temperature (0 = greedy, 1.0 = default)")
	topP := flag.Float64("top_p", 1.0, "Nucleus sampling threshold (1.0 = disabled)")
	seed := flag.Int64("seed", 0, "Random seed (0 = time-based)")

	flag.Parse()

	var cfg config.Config
	switch *profile {
	case "small":
		cfg = config.DefaultConfig
	case "medium":
		cfg = config.GPT2Medium
	case "debug":
		cfg = config.Config{
			VocabSize:         1000,
			ContextLen:        32,
			EmbDim:            32,
			NHeads:            4,
			NLayers:           2,
			DropRate:          0.0,
			QKVBias:           true,
			RepetitionPenalty: 1.0,
			Temperature:       1.0,
			TopP:              1.0,
			Seed:              0,
		}
	default:
		fmt.Printf("Unknown profile: %s (use debug, small, or medium)\n", *profile)
		return
	}
	cfg.RepetitionPenalty = *repPenalty
	cfg.Temperature = *temperature
	cfg.TopP = *topP
	cfg.Seed = *seed

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
