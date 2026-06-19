package inference

import (
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
