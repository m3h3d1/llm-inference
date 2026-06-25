package tokenizer

import (
	"os"
	"path/filepath"
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
		tok, err := NewFromFiles("../assets/tokenizer/vocab.json", "../assets/tokenizer/merges.txt")
		if err != nil {
			t.Fatalf("Failed to load files: %v", err)
		}

		// Known GPT-2 vocab: 50257 entries
		if len(tok.Vocab) != 50257 {
			t.Errorf("Expected 50257 vocab entries, got %d", len(tok.Vocab))
		}
		// Known GPT-2 merges: 50000 entries (file may contain trailing newline)
		if len(tok.Merges) < 50000 {
			t.Errorf("Expected at least 50000 merges, got %d", len(tok.Merges))
		}

		// Verify specific known token mappings (from assets/vocab.json)
		if tok.Vocab["hello"] != 31373 {
			t.Errorf("Expected 'hello' token ID 31373, got %d", tok.Vocab["hello"])
		}
		if tok.Vocab["world"] != 6894 {
			t.Errorf("Expected 'world' token ID 6894, got %d", tok.Vocab["world"])
		}

		// Verify round-trip
		input := "hello world"
		ids := tok.Encode(input)
		decoded := tok.Decode(ids)
		if decoded != input {
			t.Errorf("RoundTrip failed: %q -> %v -> %q", input, ids, decoded)
		}
	})
}

func TestNewFromFilesErrors(t *testing.T) {
	t.Run("missing vocab file", func(t *testing.T) {
		_, err := NewFromFiles("/nonexistent/path/vocab.json", "../assets/tokenizer/merges.txt")
		if err == nil {
			t.Fatal("expected error for missing vocab file")
		}
	})

	t.Run("malformed JSON vocab", func(t *testing.T) {
		dir := t.TempDir()
		vocabPath := filepath.Join(dir, "vocab.json")
		mergesPath := filepath.Join(dir, "merges.txt")
		if err := os.WriteFile(vocabPath, []byte("{invalid json}"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(mergesPath, []byte("h e\n"), 0644); err != nil {
			t.Fatal(err)
		}
		_, err := NewFromFiles(vocabPath, mergesPath)
		if err == nil {
			t.Fatal("expected error for malformed JSON")
		}
	})

	t.Run("missing merges file", func(t *testing.T) {
		dir := t.TempDir()
		vocabPath := filepath.Join(dir, "vocab.json")
		if err := os.WriteFile(vocabPath, []byte(`{"hello": 0}`), 0644); err != nil {
			t.Fatal(err)
		}
		_, err := NewFromFiles(vocabPath, "/nonexistent/path/merges.txt")
		if err == nil {
			t.Fatal("expected error for missing merges file")
		}
	})
}

func TestAddedTokens(t *testing.T) {
	tok := &Tokenizer{
		Vocab: map[string]int{
			"h": 0, "e": 1, "l": 2, "o": 3,
			"he": 4, "ll": 5,
			"<|im_start|>": 6,
			"<|im_end|>":   7,
		},
		Merges: [][2]string{
			{"h", "e"},
			{"l", "l"},
		},
		AddedTokens: []string{"<|im_start|>", "<|im_end|>"},
	}
	tok.buildRevVocab()

	t.Run("special token stays atomic", func(t *testing.T) {
		input := "he<|im_start|>llo"
		ids := tok.Encode(input)
		// Expect: "he" -> [4], "<|im_start|>" -> [6], "llo" -> sub-tokens of "ll"+"o" -> [5,3]
		want := []int{4, 6, 5, 3}
		if len(ids) != len(want) {
			t.Fatalf("Encode(%q) = %v, want %v", input, ids, want)
		}
		for i := range ids {
			if ids[i] != want[i] {
				t.Errorf("index %d: got %d, want %d", i, ids[i], want[i])
			}
		}
	})

	t.Run("multiple special tokens", func(t *testing.T) {
		input := "<|im_start|>hello<|im_end|>"
		ids := tok.Encode(input)
		if ids[0] != 6 || ids[len(ids)-1] != 7 {
			t.Errorf("Expected start/end tokens, got %v", ids)
		}
	})

	t.Run("decode round-trip", func(t *testing.T) {
		// With mock Tokenizer, decode returns the token strings joined
		// since we don't have bytesToUnicode mapping. The special tokens
		// surive encode, so they should surive decode and be present.
		input := "he<|im_start|>o"
		ids := tok.Encode(input)
		decoded := tok.Decode(ids)
		if decoded != input {
			t.Errorf("Round-trip: %q -> %v -> %q", input, ids, decoded)
		}
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