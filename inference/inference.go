package inference

import (
	"math/rand"
	"time"

	"github.com/llm/config"
	"github.com/llm/model"
	"github.com/llm/tensor"
	"github.com/llm/tokenizer"
)

type Model interface {
	Forward(tokenIDs []int) *tensor.Tensor
	ForwardWithCache(tokenIDs []int, pastCache *model.KVCache) (*tensor.Tensor, *model.KVCache)
}

func GenerateStreaming(cfg config.Config, m Model, tok *tokenizer.Tokenizer, prompt string, maxNewTokens int, onToken func(string)) string {
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

	logits, cache := m.ForwardWithCache(ids, nil)
	prevDecoded := tok.Decode(ids)

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

		ids = append(ids, nextTokenID)
		logits, cache = m.ForwardWithCache([]int{nextTokenID}, cache)

		decoded := tok.Decode(ids)
		delta := decoded[len(prevDecoded):]
		if onToken != nil {
			onToken(delta)
		}
		prevDecoded = decoded
	}

	return tok.Decode(ids)
}

func Generate(cfg config.Config, m Model, tok *tokenizer.Tokenizer, prompt string, maxNewTokens int) string {
	return GenerateStreaming(cfg, m, tok, prompt, maxNewTokens, nil)
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
