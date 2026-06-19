package inference

import (
	"math"
	"math/rand"
	"sort"
)

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

func Softmax(logits []float64) []float64 {
	n := len(logits)
	probs := make([]float64, n)

	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	expSum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxVal)
		expSum += probs[i]
	}

	for i := range probs {
		probs[i] /= expSum
	}

	return probs
}

func ApplyTemperature(logits []float64, temp float64) []float64 {
	if temp <= 0 {
		return logits
	}
	for i := range logits {
		logits[i] /= temp
	}
	return logits
}

func ApplyTopP(logits []float64, p float64) []float64 {
	if p >= 1.0 {
		return logits
	}
	if p <= 0 {
		p = 1e-10
	}

	n := len(logits)

	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return logits[indices[i]] > logits[indices[j]]
	})

	sortedLogits := make([]float64, n)
	for i, idx := range indices {
		sortedLogits[i] = logits[idx]
	}

	sortedProbs := Softmax(sortedLogits)

	cumsum := 0.0
	cutoff := n
	for i, prob := range sortedProbs {
		cumsum += prob
		if cumsum > p && cutoff == n {
			cutoff = i
		}
	}

	if cutoff == n {
		cutoff = n - 1
	}

	for i := cutoff + 1; i < n; i++ {
		logits[indices[i]] = math.Inf(-1)
	}

	return logits
}

func Sample(probs []float64, rng *rand.Rand) int {
	total := 0.0
	for _, p := range probs {
		total += p
	}
	if total <= 0 {
		return 0
	}

	r := rng.Float64() * total
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return len(probs) - 1
}
