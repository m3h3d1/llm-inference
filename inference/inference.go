package inference

import (
	"math/rand"
	"time"

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

	var rng *rand.Rand
	if cfg.Seed != 0 {
		rng = rand.New(rand.NewSource(cfg.Seed))
	} else {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	logits, cache := gpt.ForwardWithCache(ids, nil)

	for i := 0; i < maxNewTokens; i++ {
		lastPos := logits.Seq() - 1
		lastTokenLogits := make([]float64, cfg.VocabSize)
		for v := 0; v < cfg.VocabSize; v++ {
			lastTokenLogits[v] = logits.At(0, lastPos, v)
		}

		if cfg.RepetitionPenalty > 1.0 {
			lastTokenLogits = ApplyRepetitionPenalty(lastTokenLogits, ids, cfg.RepetitionPenalty)
		}

		var nextTokenID int
		if cfg.Temperature == 0 {
			nextTokenID = argmax(lastTokenLogits)
		} else {
			logits := ApplyTemperature(lastTokenLogits, cfg.Temperature)
			logits = ApplyTopP(logits, cfg.TopP)
			probs := Softmax(logits)
			nextTokenID = Sample(probs, rng)
		}

		if nextTokenID == eosTokenID {
			break
		}

		ids = append(ids, nextTokenID)
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
