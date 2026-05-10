package tokenizer

import (
	"bufio"
	"encoding/json"
	"os"
)

type Tokenizer struct {
	Vocab  map[string]int
	Merges [][2]string
}

func NewMock() *Tokenizer {
	return &Tokenizer{
		Vocab: map[string]int{
			"h":  0,
			"e":  1,
			"l":  2,
			"o":  3,
			"he": 4,
			"ll": 5,
		},
		Merges: [][2]string{
			{"h", "e"}, // Merge "h"+"e" -> "he" (Priority 0)
			{"l", "l"}, // Merge "l"+"l" -> "ll" (Priority 1)
		},
	}
}

func NewFromFiles(vocabPath, mergesPath string) (*Tokenizer, error) {
	tok := &Tokenizer{
		Vocab:  make(map[string]int),
		Merges: make([][2]string, 0),
	}

	// Load Vocab (JSON)
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &tok.Vocab); err != nil {
		return nil, err
	}

	// Load Merges (Text: line by line)
	file, err := os.Open(mergesPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// Parse: "a b" -> ["a", "b"]
		parts := []string{}
		current := ""
		for _, r := range line {
			if string(r) == " " {
				if current != "" {
					parts = append(parts, current)
					current = ""
				}
			} else {
				current += string(r)
			}
		}
		if current != "" {
			parts = append(parts, current)
		}

		if len(parts) == 2 {
			tok.Merges = append(tok.Merges, [2]string{parts[0], parts[1]})
		}
	}

	return tok, scanner.Err()
}

func (t *Tokenizer) Encode(text string) []int {
	// 1. Split into individual rune strings
	tokens := []string{}
	for _, r := range text {
		tokens = append(tokens, string(r))
	}

	// 2. BPE Loop (Limit iterations for safety)
	for iterations := 0; iterations < 100; iterations++ {
		merged := false

		// Scan merges by priority
		for _, merge := range t.Merges {
			// Scan tokens to find this pair
			for i := 0; i < len(tokens)-1; i++ {
				if tokens[i] == merge[0] && tokens[i+1] == merge[1] {
					// Merge found! Replace tokens[i] and tokens[i+1] with merged token
					newToken := merge[0] + merge[1]
					tokens[i] = newToken
					// Remove tokens[i+1]
					copy(tokens[i+1:], tokens[i+2:])
					tokens = tokens[:len(tokens)-1]
					merged = true
					break
				}
			}
			if merged {
				break // Restart merge search with new tokens
			}
		}

		if !merged {
			break // No more merges found
		}
	}

	// 3. Map to IDs
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		id, ok := t.Vocab[token]
		if !ok {
			id = 0 // Handle unknown
		}
		ids[i] = id
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	// 1. Build reverse vocab
	revVocab := make(map[int]string)
	for k, v := range t.Vocab {
		revVocab[v] = k
	}

	// 2. Map IDs to strings
	tokens := make([]string, len(ids))
	for i, id := range ids {
		token, ok := revVocab[id]
		if !ok {
			token = ""
		}
		tokens[i] = token
	}

	// 3. Join
	result := ""
	for _, token := range tokens {
		result += token
	}
	return result
}