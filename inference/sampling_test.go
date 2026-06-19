package inference

import (
	"math"
	"math/rand"
	"testing"
)

func TestApplyRepetitionPenalty_PenalizesPositiveLogit(t *testing.T) {
	logits := []float64{0.0, 10.0, 5.0, 0.0}
	result := ApplyRepetitionPenalty(logits, []int{1}, 1.2)
	if result[1] >= 10.0 {
		t.Fatalf("expected penalized[1] < 10.0, got %f", result[1])
	}
}

func TestApplyRepetitionPenalty_PenalizesNegativeLogit(t *testing.T) {
	logits := []float64{0.0, -5.0, 0.0, 0.0}
	result := ApplyRepetitionPenalty(logits, []int{1}, 1.2)
	if result[1] >= -5.0 {
		t.Fatalf("expected penalized[1] < -5.0, got %f", result[1])
	}
}

func TestApplyRepetitionPenalty_UnseenTokensUnchanged(t *testing.T) {
	logits := []float64{10.0, 5.0, 0.0}
	result := ApplyRepetitionPenalty(logits, []int{2}, 1.2)
	if result[0] != 10.0 || result[1] != 5.0 {
		t.Fatalf("expected unseen tokens unchanged, got %v", result)
	}
}

func TestApplyRepetitionPenalty_IdentityPenalty(t *testing.T) {
	logits := []float64{0.0, 10.0, 5.0, 0.0}
	original := make([]float64, len(logits))
	copy(original, logits)
	result := ApplyRepetitionPenalty(logits, []int{1}, 1.0)
	for i := range result {
		if result[i] != original[i] {
			t.Fatalf("expected logits unchanged at index %d: %f vs %f", i, result[i], original[i])
		}
	}
}

func TestApplyRepetitionPenalty_EmptyTokenIDs(t *testing.T) {
	logits := []float64{10.0, 5.0, 3.0}
	original := make([]float64, len(logits))
	copy(original, logits)
	result := ApplyRepetitionPenalty(logits, []int{}, 1.5)
	for i := range result {
		if result[i] != original[i] {
			t.Fatalf("expected logits unchanged with empty tokenIDs at index %d", i)
		}
	}
}

func TestSoftmax_SumsToOne(t *testing.T) {
	cases := [][]float64{
		{1.0, 2.0, 3.0},
		{0.0, 0.0, 0.0},
		{-100.0, 0.0, 100.0},
		{1.0},
	}
	for _, input := range cases {
		probs := Softmax(input)
		sum := 0.0
		for _, p := range probs {
			sum += p
		}
		if math.Abs(sum-1.0) > 1e-12 {
			t.Fatalf("Softmax(%v) sum = %f, want 1.0", input, sum)
		}
	}
}

func TestSoftmax_Monotonic(t *testing.T) {
	logits := []float64{0.5, 1.0, 2.0, 4.0}
	probs := Softmax(logits)
	for i := 1; i < len(probs); i++ {
		if probs[i] <= probs[i-1] {
			t.Fatalf("Softmax not monotonic: probs[%d]=%f <= probs[%d]=%f", i, probs[i], i-1, probs[i-1])
		}
	}
}

func TestSoftmax_NumericalStability(t *testing.T) {
	logits := []float64{-1e5, 0.0, 1e5}
	probs := Softmax(logits)
	if math.IsNaN(probs[0]) || math.IsNaN(probs[1]) || math.IsNaN(probs[2]) {
		t.Fatal("Softmax produced NaN")
	}
	// The largest logit should dominate
	if probs[2] < 0.99 {
		t.Fatalf("expected probs[2] near 1.0, got %f", probs[2])
	}
}

func TestApplyTemperature_NoOp(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	original := make([]float64, len(logits))
	copy(original, logits)
	result := ApplyTemperature(logits, 1.0)
	for i := range result {
		if result[i] != original[i] {
			t.Fatalf("temp=1 should preserve logits at %d", i)
		}
	}
}

func TestApplyTemperature_Scales(t *testing.T) {
	logits := []float64{2.0, 4.0, 6.0}
	result := ApplyTemperature(logits, 2.0)
	if result[0] != 1.0 || result[1] != 2.0 || result[2] != 3.0 {
		t.Fatalf("temp=2 should halve logits, got %v", result)
	}
}

func TestApplyTemperature_ZeroReturnsInput(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	original := make([]float64, len(logits))
	copy(original, logits)
	result := ApplyTemperature(logits, 0.0)
	for i := range result {
		if result[i] != original[i] {
			t.Fatalf("temp=0 should preserve logits, got %v", result)
		}
	}
}

func TestApplyTemperature_NegativeReturnsInput(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	original := make([]float64, len(logits))
	copy(original, logits)
	result := ApplyTemperature(logits, -0.5)
	for i := range result {
		if result[i] != original[i] {
			t.Fatalf("temp negative should preserve logits, got %v", result)
		}
	}
}

func TestApplyTopP_NoOp(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0}
	original := make([]float64, len(logits))
	copy(original, logits)
	result := ApplyTopP(logits, 1.0)
	for i := range result {
		if result[i] != original[i] {
			t.Fatalf("p=1.0 should preserve all logits, got %v", result)
		}
	}
}

func TestApplyTopP_FiltersTail(t *testing.T) {
	// 4 tokens with very uneven distribution — top 2 should cover >0.9 of probability
	logits := []float64{100.0, 90.0, -1000.0, -1000.0}
	result := ApplyTopP(logits, 0.9)
	// Tail tokens (indices 2, 3) should be -inf
	if !math.IsInf(result[2], -1) || !math.IsInf(result[3], -1) {
		t.Fatalf("expected tail tokens to be -inf, got %v", result)
	}
}

func TestApplyTopP_KeepsBoundaryToken(t *testing.T) {
	logits := []float64{0.0, 5.0, 10.0, 15.0}
	result := ApplyTopP(logits, 0.9)
	// At least the top token should survive
	survivors := 0
	for _, v := range result {
		if !math.IsInf(v, -1) {
			survivors++
		}
	}
	if survivors < 1 {
		t.Fatal("expected at least 1 survivor with default min_tokens_to_keep")
	}
}

func TestApplyTopP_ZeroPKeepsOneToken(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0}
	result := ApplyTopP(logits, 0.0)
	survivors := 0
	for _, v := range result {
		if !math.IsInf(v, -1) {
			survivors++
		}
	}
	if survivors != 1 {
		t.Fatalf("expected exactly 1 survivor for p=0, got %d", survivors)
	}
}

func TestSample_Deterministic(t *testing.T) {
	probs := []float64{0.6, 0.3, 0.1}
	rng1 := rand.New(rand.NewSource(42))
	rng2 := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		a := Sample(probs, rng1)
		b := Sample(probs, rng2)
		if a != b {
			t.Fatalf("same seed produced different samples at iteration %d: %d vs %d", i, a, b)
		}
	}
}

func TestSample_WithZeroProbs(t *testing.T) {
	probs := []float64{0.0, 0.0, 0.0, 0.0}
	rng := rand.New(rand.NewSource(42))
	result := Sample(probs, rng)
	if result != 0 {
		t.Fatalf("expected 0 for all-zero probs, got %d", result)
	}
}

func TestSample_DistributionShape(t *testing.T) {
	// Token 0 should be chosen ~60% of the time
	probs := []float64{0.6, 0.3, 0.1}
	rng := rand.New(rand.NewSource(42))
	counts := make([]int, 3)
	samples := 10000
	for i := 0; i < samples; i++ {
		counts[Sample(probs, rng)]++
	}
	// Token 0 should be the most common
	if counts[0] <= counts[1] || counts[0] <= counts[2] {
		t.Fatalf("expected token 0 to be most common, got counts %v", counts)
	}
}
