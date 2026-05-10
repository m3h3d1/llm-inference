package tokenizer

import (
	"bufio"
	"encoding/json"
	"os"
)

type Tokenizer struct {
	Vocab           map[string]int
	Merges          [][2]string
	bytesToUnicode  map[byte]rune
	unicodeToBytes  map[rune]byte
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
			{"h", "e"},
			{"l", "l"},
		},
		bytesToUnicode: make(map[byte]rune),
		unicodeToBytes: make(map[rune]byte),
	}
}

func NewFromFiles(vocabPath, mergesPath string) (*Tokenizer, error) {
	tok := &Tokenizer{
		Vocab:  make(map[string]int),
		Merges: make([][2]string, 0),
	}

	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &tok.Vocab); err != nil {
		return nil, err
	}

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

	// GPT-2 official bytesToUnicode mapping
	tok.bytesToUnicode = make(map[byte]rune)
	tok.unicodeToBytes = make(map[rune]byte)
	for b := 0; b < 256; b++ {
		byteVal := byte(b)
		var r rune
		if (b >= 33 && b <= 126) || b == 10 || b == 13 {
			r = rune(b)
		} else {
			r = rune(b + 256)
		}
		tok.bytesToUnicode[byteVal] = r
		tok.unicodeToBytes[r] = byteVal
	}

	return tok, scanner.Err()
}

func (t *Tokenizer) Encode(text string) []int {
	bytes := []byte(text)
	tokens := make([]string, len(bytes))
	for i, b := range bytes {
		if t.bytesToUnicode != nil {
			tokens[i] = string(t.bytesToUnicode[b])
		} else {
			tokens[i] = string(b)
		}
	}

	for iterations := 0; iterations < 10000; iterations++ {
		merged := false
		for _, merge := range t.Merges {
			for i := 0; i < len(tokens)-1; i++ {
				if tokens[i] == merge[0] && tokens[i+1] == merge[1] {
					newToken := merge[0] + merge[1]
					tokens[i] = newToken
					copy(tokens[i+1:], tokens[i+2:])
					tokens = tokens[:len(tokens)-1]
					merged = true
					break
				}
			}
			if merged {
				break
			}
		}
		if !merged {
			break
		}
	}

	ids := make([]int, len(tokens))
	for i, token := range tokens {
		id, ok := t.Vocab[token]
		if !ok {
			id = 0
		}
		ids[i] = id
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	revVocab := make(map[int]string)
	for k, v := range t.Vocab {
		revVocab[v] = k
	}

	tokens := make([]string, len(ids))
	for i, id := range ids {
		token, ok := revVocab[id]
		if !ok {
			token = ""
		}
		tokens[i] = token
	}

	var resultBytes []byte
	for _, token := range tokens {
		for _, r := range token {
			if t.unicodeToBytes != nil {
				if b, ok := t.unicodeToBytes[r]; ok {
					resultBytes = append(resultBytes, b)
				} else {
					resultBytes = append(resultBytes, []byte(string(r))...)
				}
			} else {
				resultBytes = append(resultBytes, []byte(string(r))...)
			}
		}
	}

	return string(resultBytes)
}