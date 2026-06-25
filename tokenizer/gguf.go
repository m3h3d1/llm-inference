package tokenizer

import (
	"fmt"
	"sort"
	"strings"

	"github.com/llm/gguf"
)

func NewFromGGUF(f *gguf.File) (*Tokenizer, error) {
	modelVal, ok := f.Metadata["tokenizer.ggml.model"]
	if !ok {
		return nil, fmt.Errorf("gguf: missing tokenizer.ggml.model")
	}
	modelStr, _ := modelVal.String()
	if modelStr != "gpt2" {
		return nil, fmt.Errorf("gguf: unsupported tokenizer model %q (expected gpt2)", modelStr)
	}

	tokensVal, ok := f.Metadata["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("gguf: missing tokenizer.ggml.tokens")
	}
	tokensArr, ok := tokensVal.Array()
	if !ok {
		return nil, fmt.Errorf("gguf: tokenizer.ggml.tokens is not an array")
	}

	vocab := make(map[string]int, len(tokensArr))
	for i, v := range tokensArr {
		s, ok := v.String()
		if !ok {
			return nil, fmt.Errorf("gguf: tokenizer.ggml.tokens[%d] is not a string", i)
		}
		vocab[s] = i
	}

	merges := make([][2]string, 0)
	if mergesVal, ok := f.Metadata["tokenizer.ggml.merges"]; ok {
		mergesArr, ok := mergesVal.Array()
		if !ok {
			return nil, fmt.Errorf("gguf: tokenizer.ggml.merges is not an array")
		}
		for _, v := range mergesArr {
			s, ok := v.String()
			if !ok {
				continue
			}
			parts := strings.Fields(s)
			if len(parts) == 2 {
				merges = append(merges, [2]string{parts[0], parts[1]})
			}
		}
	}

	tok := &Tokenizer{
		Vocab:  vocab,
		Merges: merges,
	}
	tok.buildRevVocab()
	tok.buildBytesToUnicode()

	for token := range vocab {
		if strings.Contains(token, "<") || strings.Contains(token, ">") {
			tok.AddedTokens = append(tok.AddedTokens, token)
		}
	}
	sort.Slice(tok.AddedTokens, func(i, j int) bool {
		return len(tok.AddedTokens[i]) > len(tok.AddedTokens[j])
	})

	return tok, nil
}
