package main

import (
	"flag"
	"fmt"

	"github.com/llm/config"
	"github.com/llm/gguf"
	"github.com/llm/inference"
	gpt2 "github.com/llm/model/gpt2"
	"github.com/llm/model/llama"
	"github.com/llm/tokenizer"
	"github.com/llm/weights"
)

func main() {
	profile := flag.String("profile", "debug", "Model profile: debug (mock, no weights), small (GPT-2 Small 124M), or medium (GPT-2 Medium 355M)")
	weightsPath := flag.String("weights", "", "Path to the weights file")
	format := flag.String("format", "json", "Weight format: json or bin")
	ggufPath := flag.String("gguf", "", "Path to a GGUF model file (overrides profile/weights/format)")
	prompt := flag.String("prompt", "The", "Input prompt for text generation")
	maxTokens := flag.Int("max_tokens", 30, "Maximum number of tokens to generate")
	strict := flag.Bool("strict", false, "Fail if weights are missing")
	repPenalty := flag.Float64("repetition_penalty", 1.0, "Repetition penalty (>1.0 penalizes repeated tokens)")
	temperature := flag.Float64("temperature", 1.0, "Sampling temperature (0 = greedy, 1.0 = default)")
	topP := flag.Float64("top_p", 1.0, "Nucleus sampling threshold (1.0 = disabled)")
	seed := flag.Int64("seed", 0, "Random seed (0 = time-based)")

	flag.Parse()

	if *ggufPath != "" {
		runGGUF(*ggufPath, *prompt, *maxTokens, *repPenalty, *temperature, *topP, *seed)
		return
	}

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
			EOSTokenID:        50256,
		}
	default:
		fmt.Printf("Unknown profile: %s (use debug, small, or medium)\n", *profile)
		return
	}
	cfg.RepetitionPenalty = *repPenalty
	cfg.Temperature = *temperature
	cfg.TopP = *topP
	cfg.Seed = *seed

	gpt := gpt2.NewModel(cfg)
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
	fmt.Print("Generated text: ")
	inference.GenerateStreaming(cfg, gpt, tok, *prompt, *maxTokens, func(text string) {
		fmt.Print(text)
	})
	fmt.Println()
}

func runGGUF(path, prompt string, maxTokens int, repPenalty, temperature, topP float64, seed int64) {
	fmt.Printf("Loading GGUF model: %s\n", path)
	f, err := gguf.Open(path)
	if err != nil {
		fmt.Printf("Error opening GGUF file: %v\n", err)
		return
	}

	cfg, err := weights.LoadConfigFromGGUF(f)
	if err != nil {
		fmt.Printf("Error loading config from GGUF: %v\n", err)
		return
	}
	cfg.RepetitionPenalty = repPenalty
	cfg.Temperature = temperature
	cfg.TopP = topP
	cfg.Seed = seed

	archStr, _ := f.Metadata["general.architecture"].String()
	fmt.Printf("Architecture: %s, layers=%d, embDim=%d, heads=%d, kvHeads=%d\n",
		archStr,
		cfg.NLayers, cfg.EmbDim, cfg.NHeads, cfg.NKVHeads)

	m := llama.NewModel(cfg)

	fmt.Println("Loading weights from GGUF...")
	if err := weights.LoadWeightsFromGGUF(m, f); err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		return
	}
	fmt.Println("Weights loaded successfully")

	tok, err := tokenizer.NewFromGGUF(f)
	if err != nil {
		fmt.Printf("Error loading tokenizer from GGUF: %v\n", err)
		return
	}
	fmt.Printf("Tokenizer loaded: %d tokens\n", len(tok.Vocab))

	fmt.Printf("Generating text with prompt: %s\n", prompt)
	fmt.Print("Result: ")
	inference.GenerateStreaming(cfg, m, tok, prompt, maxTokens, func(text string) {
		fmt.Print(text)
	})
	fmt.Println()
}
