package inference

import (
	"github.com/llm/config"
	"github.com/llm/model"
	"github.com/llm/tokenizer"
)

func Generate(cfg config.Config, gpt *model.GPTModel, tok *tokenizer.Tokenizer, prompt string, maxNewTokens int) string {
	// 1. Tokenize the prompt
	ids := tok.Encode(prompt)
	
	// 2. Generation Loop
	for i := 0; i < maxNewTokens; i++ {
		// 3. Forward Pass
		logits := gpt.Forward(ids)
		
		// 4. Get logits for the last token
		seqLen := len(ids)
		lastTokenLogits := make([]float64, cfg.VocabSize)
		for v := 0; v < cfg.VocabSize; v++ {
			lastTokenLogits[v] = logits.At(0, seqLen-1, v)
		}
		
		// 5. Greedy Search: Find argmax
		nextTokenID := argmax(lastTokenLogits)
		
		// 6. Append to sequence
		ids = append(ids, nextTokenID)
	}
	
	// 7. Decode and return
	return tok.Decode(ids)
}

func argmax(scores []float64) int {
	maxScore := scores[0]
	maxIndex := 0
	
	for i := 1; i < len(scores); i++ {
		if scores[i] > maxScore {
			maxScore = scores[i]
			maxIndex = i
		}
	}
	
	return maxIndex
}