package inference

func ApplyRepetitionPenalty(logits []float64, tokenIDs []int, penalty float64) []float64 {
	for _, id := range tokenIDs {
		if logits[id] < 0 {
			logits[id] *= penalty
		} else {
			logits[id] /= penalty
		}
	}
	return logits
}
