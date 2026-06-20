package config

type Config struct {
	VocabSize         int
	ContextLen        int
	EmbDim            int
	NHeads            int
	NLayers           int
	DropRate          float64
	QKVBias           bool
	RepetitionPenalty float64
	Temperature       float64
	TopP              float64
	Seed              int64
}

var DefaultConfig = Config{
	VocabSize:         50257,
	ContextLen:        1024,
	EmbDim:            768,
	NHeads:            12,
	NLayers:           12,
	DropRate:          0.1,
	QKVBias:           true,
	RepetitionPenalty: 1.0,
	Temperature:       1.0,
	TopP:              1.0,
	Seed:              0,
}

var GPT2Medium = Config{
	VocabSize:         50257,
	ContextLen:        1024,
	EmbDim:            1024,
	NHeads:            16,
	NLayers:           24,
	DropRate:          0.1,
	QKVBias:           true,
	RepetitionPenalty: 1.0,
	Temperature:       1.0,
	TopP:              1.0,
	Seed:              0,
}
