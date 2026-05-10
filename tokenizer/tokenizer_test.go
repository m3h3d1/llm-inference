package tokenizer

import (
	"testing"
)

func TestEncode(t *testing.T) {
	tok := NewMock()

	t.Run("encode hello", func(t *testing.T) {
		input := "hello"
		got := tok.Encode(input)
		want := []int{4, 5, 3} // "he" + "ll" + "o"

		// Simple check
		if len(got) != len(want) {
			t.Fatalf("Length mismatch: got %v, want %v", got, want)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("Encode mismatch at index %d: got %d, want %d", i, got[i], want[i])
			}
		}
	})
}

func TestDecode(t *testing.T) {
	tok := NewMock()

	t.Run("decode ids", func(t *testing.T) {
		input := []int{4, 5, 3}
		want := "hello"
		got := tok.Decode(input)

		if got != want {
			t.Errorf("Decode(%v) = %q, want %q", input, got, want)
		}
	})
}

func TestRoundTrip(t *testing.T) {
	tok := NewMock()
	original := "hello"

	encoded := tok.Encode(original)
	decoded := tok.Decode(encoded)

	if decoded != original {
		t.Errorf("RoundTrip failed: %q -> %v -> %q", original, encoded, decoded)
	}
}