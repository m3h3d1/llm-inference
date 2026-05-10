package tokenizer

import (
	"testing"
)

func TestDebug(t *testing.T) {
	tok, err := NewFromFiles("../assets/tokenizer/vocab.json", "../assets/tokenizer/merges.txt")
	if err != nil {
		t.Fatal(err)
	}

	input := "hello world"
	
	// Encode
	ids := tok.Encode(input)
	t.Logf("Input: %q", input)
	t.Logf("IDs: %v", ids)

	// Decode
	decoded := tok.Decode(ids)
	t.Logf("Decoded: %q", decoded)

	// Show first few IDs and their tokens
	revVocab := make(map[int]string)
	for k, v := range tok.Vocab {
		revVocab[v] = k
	}
	
	for i, id := range ids {
		if i > 10 {
			break
		}
		t.Logf("ID %d: %q", id, revVocab[id])
	}
}