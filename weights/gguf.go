package weights

import (
	"fmt"

	"github.com/llm/gguf"
	"github.com/llm/config"
	"github.com/llm/tensor"
)

func LoadConfigFromGGUF(f *gguf.File) (config.Config, error) {
	arch, ok := f.Metadata["general.architecture"]
	if !ok {
		return config.Config{}, fmt.Errorf("gguf: missing general.architecture")
	}
	archStr, _ := arch.String()

	readUint := func(key string) (uint64, error) {
		prefixed := archStr + "." + key
		v, ok := f.Metadata[prefixed]
		if !ok {
			return 0, fmt.Errorf("gguf: missing metadata %q", prefixed)
		}
		n, ok := v.Uint64()
		if !ok {
			return 0, fmt.Errorf("gguf: metadata %q is not an integer", prefixed)
		}
		return n, nil
	}

	readFloat := func(key string) (float64, error) {
		prefixed := archStr + "." + key
		v, ok := f.Metadata[prefixed]
		if !ok {
			return 0, fmt.Errorf("gguf: missing metadata %q", prefixed)
		}
		n, ok := v.Float64()
		if !ok {
			return 0, fmt.Errorf("gguf: metadata %q is not a float", prefixed)
		}
		return n, nil
	}

	embDim, err := readUint("embedding_length")
	if err != nil {
		return config.Config{}, err
	}

	blockCount, err := readUint("block_count")
	if err != nil {
		return config.Config{}, err
	}

	headCount, err := readUint("attention.head_count")
	if err != nil {
		return config.Config{}, err
	}

	nKVHeads := headCount
	if kv, err := readUint("attention.head_count_kv"); err == nil {
		nKVHeads = kv
	}

	dff, err := readUint("feed_forward_length")
	if err != nil {
		return config.Config{}, err
	}

	ctxLen := uint64(0)
	if v, err := readUint("context_length"); err == nil {
		ctxLen = v
	}

	vocabSize := uint64(0)
	if v, err := readUint("vocab_size"); err == nil {
		vocabSize = v
	}

	ropeDim := uint64(0)
	if v, err := readUint("rope.dimension_count"); err == nil {
		ropeDim = v
	}

	ropeTheta := 10000.0
	if v, err := readFloat("rope.freq_base"); err == nil {
		ropeTheta = v
	}

	rmsEps := 1e-5
	if v, err := readFloat("attention.layer_norm_rms_epsilon"); err == nil {
		rmsEps = v
	}

	eosID := int(vocabSize) - 1
	if eosID < 0 {
		eosID = 0
	}
	if v, ok := f.Metadata["tokenizer.ggml.eos_token_id"]; ok {
		if n, ok2 := v.Uint64(); ok2 {
			eosID = int(n)
		}
	}

	return config.Config{
		VocabSize:         int(vocabSize),
		ContextLen:        int(ctxLen),
		EmbDim:            int(embDim),
		NHeads:            int(headCount),
		NKVHeads:          int(nKVHeads),
		NLayers:           int(blockCount),
		DFF:               int(dff),
		DropRate:          0.0,
		RopeDim:           int(ropeDim),
		RopeTheta:         ropeTheta,
		RmsNormEps:        rmsEps,
		QKVBias:           false,
		UseTiedEmbeddings: true,
		EOSTokenID:        eosID,
		Temperature:       1.0,
		TopP:              1.0,
		RepetitionPenalty: 1.0,
		Seed:              0,
	}, nil
}

func LoadWeightsFromGGUF(m interface {
	SetParameter(path string, data *tensor.Tensor)
}, f *gguf.File) error {
	for _, ti := range f.TensorInfos {
		if ti.Name == "output.weight" {
			continue
		}

		data, err := f.ReadTensor(ti)
		if err != nil {
			return fmt.Errorf("gguf: reading %q: %v", ti.Name, err)
		}

		nDims := len(ti.Dimensions)
		var t *tensor.Tensor

		switch nDims {
		case 1:
			n := int(ti.Dimensions[0])
			t = tensor.NewTensor(1, 1, n)
			for i := 0; i < n; i++ {
				t.Set(0, 0, i, data[i])
			}

		case 2:
			dim0 := int(ti.Dimensions[0])
			dim1 := int(ti.Dimensions[1])

			t = tensor.NewTensor(1, dim0, dim1)
			for i := 0; i < dim0; i++ {
				for j := 0; j < dim1; j++ {
					t.Set(0, i, j, data[j*dim0+i])
				}
			}
			// GGUF stores dimensions reversed: header [dim0, dim1], actual data
			// layout [dim1, dim0] (same as PyTorch convention). Transpose to
			// match our Linear layer format [1, outFeatures, inFeatures].
			t = t.Transpose()

		default:
			return fmt.Errorf("gguf: unsupported tensor %q dimensions: %v", ti.Name, ti.Dimensions)
		}

		m.SetParameter(ti.Name, t)
	}

	return nil
}
