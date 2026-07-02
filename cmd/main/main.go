package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/llm/config"
	"github.com/llm/gguf"
	"github.com/llm/inference"
	"github.com/llm/model"
	gpt2 "github.com/llm/model/gpt2"
	"github.com/llm/model/llama"
	"github.com/llm/tensor"
	"github.com/llm/tokenizer"
	"github.com/llm/weights"
)

func main() {
	profile := flag.String("profile", "debug", "Model profile: debug (mock, no weights), 124M (GPT-2 Small), or 355M (GPT-2 Medium)")
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
	chatMode := flag.Bool("chat", false, "Use ChatML format (for instruct models)")
	interactive := flag.Bool("interactive", false, "Interactive multi-turn chat (requires --chat)")

	flag.Parse()

	if *interactive && !*chatMode {
		fmt.Println("Error: --interactive requires --chat")
		return
	}

	if *ggufPath != "" {
		runGGUF(*ggufPath, *prompt, *maxTokens, *repPenalty, *temperature, *topP, *seed, *chatMode, *interactive)
		return
	}

	var cfg config.Config
	switch *profile {
	case "124M":
		cfg = config.GPT2_124M
		if *weightsPath == "" {
			*weightsPath = "models/gpt2/gpt2_124M.bin"
			*format = "bin"
		}
	case "355M":
		cfg = config.GPT2_355M
		if *weightsPath == "" {
			*weightsPath = "models/gpt2/gpt2_355M.bin"
			*format = "bin"
		}
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
		fmt.Printf("Unknown profile: %s (use debug, 124M, or 355M)\n", *profile)
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
		tok, err = tokenizer.NewFromFiles("assets/gpt2/bpe-tokenizer-vocab.json", "assets/gpt2/bpe-tokenizer-merges.txt")
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

func runGGUF(path, prompt string, maxTokens int, repPenalty, temperature, topP float64, seed int64, chatMode, interactive bool) {
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

	if chatMode {
		if id, ok := tok.Vocab["<|im_end|>"]; ok {
			cfg.StopTokens = append(cfg.StopTokens, id)
		}
	}

	if interactive {
		interactiveChat(cfg, m, tok, maxTokens)
		return
	}

	if chatMode {
		systemPrompt := "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
		prompt = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	}

	fmt.Printf("Generating text with prompt: %s\n", prompt)
	fmt.Print("Result: ")
	inference.GenerateStreaming(cfg, m, tok, prompt, maxTokens, func(text string) {
		fmt.Print(strings.TrimSuffix(text, "<|im_end|>"))
	})
	fmt.Println()
}

func interactiveChat(cfg config.Config, m *llama.Model, tok *tokenizer.Tokenizer, maxTokens int) {
	type message struct {
		role    string
		content string
	}
	var history []message
	systemPrompt := "You are a helpful AI assistant named SmolLM, trained by Hugging Face"

	var rng *rand.Rand
	if cfg.Seed != 0 {
		rng = rand.New(rand.NewSource(cfg.Seed))
	} else {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	var cache *model.KVCache
	var allIDs []int

	systemText := "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n"
	systemIDs := tok.Encode(systemText)
	_, cache = m.ForwardWithCache(systemIDs, nil)
	allIDs = append(allIDs, systemIDs...)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\n> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		line = strings.TrimSpace(line)
		if line == "" || line == "exit" || line == "quit" {
			break
		}

		history = append(history, message{"user", line})

		turnText := "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n"
		turnIDs := tok.Encode(turnText)
		allIDs = append(allIDs, turnIDs...)

		var logits *tensor.Tensor
		logits, cache = m.ForwardWithCache(turnIDs, cache)
		prevDecoded := tok.Decode(allIDs)

		var response strings.Builder
		for i := 0; i < maxTokens; i++ {
			lastPos := logits.Seq() - 1
			lastTokenLogits := make([]float64, cfg.VocabSize)
			for v := 0; v < cfg.VocabSize; v++ {
				lastTokenLogits[v] = logits.At(0, lastPos, v)
			}

			if cfg.RepetitionPenalty > 1.0 {
				lastTokenLogits = inference.ApplyRepetitionPenalty(lastTokenLogits, allIDs, cfg.RepetitionPenalty)
			}

			var nextTokenID int
			if cfg.Temperature == 0 {
				nextTokenID = inference.Argmax(lastTokenLogits)
			} else {
				l := inference.ApplyTemperature(lastTokenLogits, cfg.Temperature)
				l = inference.ApplyTopP(l, cfg.TopP)
				probs := inference.Softmax(l)
				nextTokenID = inference.Sample(probs, rng)
			}

			if cfg.EOSTokenID != 0 && nextTokenID == cfg.EOSTokenID {
				break
			}
			isStop := false
			for _, stopID := range cfg.StopTokens {
				if nextTokenID == stopID {
					isStop = true
					break
				}
			}
			if isStop {
				break
			}

			allIDs = append(allIDs, nextTokenID)
			logits, cache = m.ForwardWithCache([]int{nextTokenID}, cache)

			decoded := tok.Decode(allIDs)
			delta := decoded[len(prevDecoded):]
			clean := strings.TrimSuffix(delta, "<|im_end|>")
			fmt.Print(clean)
			response.WriteString(clean)
			prevDecoded = decoded
		}
		fmt.Println()

		history = append(history, message{"assistant", strings.TrimSpace(response.String())})
	}
}


