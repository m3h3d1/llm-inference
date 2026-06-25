package tokenizer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/llm/gguf"
)

func writeGGUF(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.gguf")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeUint16LE(b []byte, offset int, v uint16) {
	b[offset] = byte(v)
	b[offset+1] = byte(v >> 8)
}

func writeUint32LE(b []byte, offset int, v uint32) {
	b[offset] = byte(v)
	b[offset+1] = byte(v >> 8)
	b[offset+2] = byte(v >> 16)
	b[offset+3] = byte(v >> 24)
}

func writeUint64LE(b []byte, offset int, v uint64) {
	b[offset] = byte(v)
	b[offset+1] = byte(v >> 8)
	b[offset+2] = byte(v >> 16)
	b[offset+3] = byte(v >> 24)
	b[offset+4] = byte(v >> 32)
	b[offset+5] = byte(v >> 40)
	b[offset+6] = byte(v >> 48)
	b[offset+7] = byte(v >> 56)
}

func writeString(b []byte, offset int, s string) int {
	writeUint64LE(b, offset, uint64(len(s)))
	copy(b[offset+8:], s)
	return offset + 8 + len(s)
}

func writeMetadataString(b []byte, offset int, key, value string) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(gguf.TypeSTRING))
	offset += 4
	return writeString(b, offset, value)
}

func writeMetadataArray(b []byte, offset int, key string, elementType gguf.ValueType, values []string) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(gguf.TypeARRAY))
	offset += 4

	writeUint32LE(b, offset, uint32(elementType))
	offset += 4
	writeUint64LE(b, offset, uint64(len(values)))
	offset += 8

	for _, v := range values {
		offset = writeString(b, offset, v)
	}
	return offset
}

func TestNewFromGGUF(t *testing.T) {
	buf := make([]byte, 4096)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 3); offset += 8

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "gpt2")
	offset = writeMetadataArray(buf, offset, "tokenizer.ggml.tokens", gguf.TypeSTRING,
		[]string{"hello", "world", "foo", "bar"})
	offset = writeMetadataArray(buf, offset, "tokenizer.ggml.merges", gguf.TypeSTRING,
		[]string{"h e", "w o"})

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	tok, err := NewFromGGUF(f)
	if err != nil {
		t.Fatalf("NewFromGGUF: %v", err)
	}

	if len(tok.Vocab) != 4 {
		t.Errorf("expected 4 vocab entries, got %d", len(tok.Vocab))
	}
	if tok.Vocab["hello"] != 0 {
		t.Errorf("expected hello=0, got %d", tok.Vocab["hello"])
	}
	if tok.Vocab["world"] != 1 {
		t.Errorf("expected world=1, got %d", tok.Vocab["world"])
	}
	if tok.Vocab["bar"] != 3 {
		t.Errorf("expected bar=3, got %d", tok.Vocab["bar"])
	}

	if len(tok.Merges) != 2 {
		t.Fatalf("expected 2 merges, got %d", len(tok.Merges))
	}
	if tok.Merges[0] != [2]string{"h", "e"} {
		t.Errorf("expected merge [h e], got %v", tok.Merges[0])
	}
	if tok.Merges[1] != [2]string{"w", "o"} {
		t.Errorf("expected merge [w o], got %v", tok.Merges[1])
	}
}

func TestNewFromGGUF_EncodeDecode(t *testing.T) {
	buf := make([]byte, 8192)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 2); offset += 8

	// Build a small GPT-2-like vocab: all single bytes and some common pairs
	vocab := make([]string, 0, 262)
	for b := 0; b < 256; b++ {
		var r rune
		if b >= 33 && b <= 126 {
			r = rune(b)
		} else {
			r = rune(b + 256)
		}
		vocab = append(vocab, string(r))
	}
	vocab = append(vocab, "hello", "world", "he", "ll", "lo", "wo", "rl", "or")
	merges := []string{"h e", "l l", "l o", "w o", "r l", "o r", "he ll", "hel lo", "wo rl", "wor ld"}

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "gpt2")
	offset = writeMetadataArray(buf, offset, "tokenizer.ggml.tokens", gguf.TypeSTRING, vocab)
	offset = writeMetadataArray(buf, offset, "tokenizer.ggml.merges", gguf.TypeSTRING, merges)

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	tok, err := NewFromGGUF(f)
	if err != nil {
		t.Fatalf("NewFromGGUF: %v", err)
	}

	ids := tok.Encode("hello")
	if len(ids) == 0 {
		t.Fatal("Encode returned empty ids")
	}
	decoded := tok.Decode(ids)
	if decoded != "hello" {
		t.Errorf("round-trip: expected 'hello', got %q", decoded)
	}

	ids = tok.Encode("hello world")
	decoded = tok.Decode(ids)
	if decoded != "hello world" {
		t.Errorf("round-trip: expected 'hello world', got %q", decoded)
	}
}

func TestNewFromGGUF_WrongModel(t *testing.T) {
	buf := make([]byte, 256)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 1); offset += 8

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "llama")

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = NewFromGGUF(f)
	if err == nil {
		t.Fatal("expected error for wrong model, got nil")
	}
}

func TestNewFromGGUF_MissingTokens(t *testing.T) {
	buf := make([]byte, 256)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 1); offset += 8

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "gpt2")

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = NewFromGGUF(f)
	if err == nil {
		t.Fatal("expected error for missing tokens, got nil")
	}
}

func TestNewFromGGUF_TokensNotArray(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 2); offset += 8

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "gpt2")
	// Write tokens as a string instead of array
	offset = writeMetadataString(buf, offset, "tokenizer.ggml.tokens", "not_an_array")

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = NewFromGGUF(f)
	if err == nil {
		t.Fatal("expected error for tokens not an array, got nil")
	}
}

func TestNewFromGGUF_TokenNotString(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 2); offset += 8

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "gpt2")
	// Write tokens array with an int32 element instead of string
	offset = writeString(buf, offset, "tokenizer.ggml.tokens")
	writeUint32LE(buf, offset, uint32(gguf.TypeARRAY)); offset += 4
	writeUint32LE(buf, offset, uint32(gguf.TypeINT32)); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint32LE(buf, offset, 42); offset += 4

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = NewFromGGUF(f)
	if err == nil {
		t.Fatal("expected error for token not a string, got nil")
	}
}

func TestNewFromGGUF_BuildBytesToUnicode(t *testing.T) {
	tok := &Tokenizer{}
	tok.buildRevVocab()
	tok.buildBytesToUnicode()

	if len(tok.bytesToUnicode) != 256 {
		t.Errorf("expected 256 byte mappings, got %d", len(tok.bytesToUnicode))
	}
	// Check printable ASCII maps to itself
	if tok.bytesToUnicode[byte('A')] != 'A' {
		t.Error("expected 'A' to map to itself")
	}
	// Check non-printable maps to byte+256
	if tok.bytesToUnicode[byte(0)] != rune(256) {
		t.Errorf("expected byte 0 to map to 256, got %d", tok.bytesToUnicode[byte(0)])
	}
	// Check round-trip
	for b := 0; b < 256; b++ {
		byteVal := byte(b)
		r := tok.bytesToUnicode[byteVal]
		back := tok.unicodeToBytes[r]
		if back != byteVal {
			t.Errorf("round-trip failed for byte %d: got %d", b, back)
		}
	}
}

func TestNewFromGGUF_NoMerges(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 2); offset += 8

	offset = writeMetadataString(buf, offset, "tokenizer.ggml.model", "gpt2")
	offset = writeMetadataArray(buf, offset, "tokenizer.ggml.tokens", gguf.TypeSTRING,
		[]string{"a", "b"})

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	tok, err := NewFromGGUF(f)
	if err != nil {
		t.Fatalf("NewFromGGUF: %v", err)
	}

	if len(tok.Merges) != 0 {
		t.Errorf("expected 0 merges, got %d", len(tok.Merges))
	}
}
