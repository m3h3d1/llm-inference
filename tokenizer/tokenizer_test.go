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

func TestNewFromFiles(t *testing.T) {
	t.Run("load gpt2 assets", func(t *testing.T) {
		// Paths are relative to project root
		tok, err := NewFromFiles("../assets/tokenizer/vocab.json", "../assets/tokenizer/merges.txt")
		if err != nil {
			t.Fatalf("Failed to load files: %v", err)
		}

		// Check vocab is loaded
		if len(tok.Vocab) == 0 {
			t.Error("Vocab is empty")
		}

		// Check merges are loaded
		if len(tok.Merges) == 0 {
			t.Error("Merges is empty")
		}

		// Log counts
		t.Logf("Vocab size: %d", len(tok.Vocab))
		t.Logf("Merges size: %d", len(tok.Merges))
	})
}

func TestIntegration(t *testing.T) {
	t.Run("encode with real data", func(t *testing.T) {
		tok, err := NewFromFiles("../assets/tokenizer/vocab.json", "../assets/tokenizer/merges.txt")
		if err != nil {
			t.Fatalf("Failed to load: %v", err)
		}

		input := "hello world"
		ids := tok.Encode(input)
		if len(ids) == 0 {
			t.Error("Encode returned empty")
		}

		// Should be roundtripable
		decoded := tok.Decode(ids)
		if decoded != input {
			t.Errorf("RoundTrip failed: %q -> %v -> %q", input, ids, decoded)
		}
	})
}