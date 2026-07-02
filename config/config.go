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

	// Llama architecture fields
	NKVHeads   int
	DFF        int
	RopeDim    int
	RopeTheta  float64
	RmsNormEps float64
	EOSTokenID int

	// Extra stop tokens (e.g., <|im_end|> for ChatML instruct models)
	StopTokens []int
}

var GPT2_124M = Config{
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
	EOSTokenID:        50256,
}

var GPT2_355M = Config{
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
	EOSTokenID:        50256,
}

var SmolLM2_135M = Config{
	VocabSize:         49152,
	ContextLen:        8192,
	EmbDim:            576,
	NHeads:            9,
	NKVHeads:          3,
	NLayers:           30,
	DFF:               1536,
	DropRate:          0.0,
	RopeDim:           64,
	RopeTheta:         100000.0,
	RmsNormEps:        1e-5,
	QKVBias:           false,
	Temperature:       1.0,
	TopP:              1.0,
	RepetitionPenalty: 1.0,
	Seed:              0,
	EOSTokenID:        49152,
}

var SmolLM2_360M = Config{
	VocabSize:         49152,
	ContextLen:        8192,
	EmbDim:            960,
	NHeads:            15,
	NKVHeads:          5,
	NLayers:           32,
	DFF:               2560,
	DropRate:          0.0,
	RopeDim:           64,
	RopeTheta:         100000.0,
	RmsNormEps:        1e-5,
	QKVBias:           false,
	Temperature:       1.0,
	TopP:              1.0,
	RepetitionPenalty: 1.0,
	Seed:              0,
	EOSTokenID:        49152,
}

var SmolLM2_1_7B = Config{
	VocabSize:         49152,
	ContextLen:        8192,
	EmbDim:            2048,
	NHeads:            32,
	NKVHeads:          32,
	NLayers:           24,
	DFF:               8192,
	DropRate:          0.0,
	RopeDim:           64,
	RopeTheta:         130000.0,
	RmsNormEps:        1e-5,
	QKVBias:           false,
	Temperature:       1.0,
	TopP:              1.0,
	RepetitionPenalty: 1.0,
	Seed:              0,
}
