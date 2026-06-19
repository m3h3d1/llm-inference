package inference

import (
	"github.com/llm/config"
	"github.com/llm/model"
	"github.com/llm/tokenizer"
)

const eosTokenID = 50256

func Generate(cfg config.Config, gpt *model.GPTModel, tok *tokenizer.Tokenizer, prompt string, maxNewTokens int) string {
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		return ""
	}

	// Prefill: process the entire prompt and build the initial KV cache
	logits, cache := gpt.ForwardWithCache(ids, nil)

	for i := 0; i < maxNewTokens; i++ {
		lastPos := logits.Seq() - 1
		lastTokenLogits := make([]float64, cfg.VocabSize)
		for v := 0; v < cfg.VocabSize; v++ {
			lastTokenLogits[v] = logits.At(0, lastPos, v)
		}

		nextTokenID := argmax(lastTokenLogits)
		if nextTokenID == eosTokenID {
			break
		}

		ids = append(ids, nextTokenID)

		// Decode: process only the new token with the cached past
		logits, cache = gpt.ForwardWithCache([]int{nextTokenID}, cache)
	}

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