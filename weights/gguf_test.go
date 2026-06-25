package weights

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/llm/gguf"
	"github.com/llm/tensor"
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

func writeMetadataUint32(b []byte, offset int, key string, value uint32) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(gguf.TypeUINT32))
	offset += 4
	writeUint32LE(b, offset, value)
	return offset + 4
}

func writeMetadataFloat32(b []byte, offset int, key string, value float32) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(gguf.TypeFLOAT32))
	offset += 4
	writeUint32LE(b, offset, math.Float32bits(value))
	return offset + 4
}

func writeMetadataString(b []byte, offset int, key, value string) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(gguf.TypeSTRING))
	offset += 4
	return writeString(b, offset, value)
}

func writeTensorInfo(b []byte, offset int, name string, dims []uint64, typ gguf.TensorType, dataOffset uint64) int {
	offset = writeString(b, offset, name)
	writeUint32LE(b, offset, uint32(len(dims)))
	offset += 4
	for _, d := range dims {
		writeUint64LE(b, offset, d)
		offset += 8
	}
	writeUint32LE(b, offset, uint32(typ))
	offset += 4
	writeUint64LE(b, offset, dataOffset)
	return offset + 8
}

func TestLoadConfigFromGGUF(t *testing.T) {
	buf := make([]byte, 2048)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8 // tensor_count
	writeUint64LE(buf, offset, 10); offset += 8 // metadata_kv_count

	offset = writeMetadataString(buf, offset, "general.architecture", "llama")
	offset = writeMetadataUint32(buf, offset, "llama.embedding_length", 576)
	offset = writeMetadataUint32(buf, offset, "llama.block_count", 30)
	offset = writeMetadataUint32(buf, offset, "llama.attention.head_count", 9)
	offset = writeMetadataUint32(buf, offset, "llama.attention.head_count_kv", 3)
	offset = writeMetadataUint32(buf, offset, "llama.feed_forward_length", 1536)
	offset = writeMetadataUint32(buf, offset, "llama.context_length", 8192)
	offset = writeMetadataUint32(buf, offset, "llama.vocab_size", 49152)
	offset = writeMetadataUint32(buf, offset, "llama.rope.dimension_count", 64)
	offset = writeMetadataFloat32(buf, offset, "llama.rope.freq_base", 100000.0)

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	cfg, err := LoadConfigFromGGUF(f)
	if err != nil {
		t.Fatalf("LoadConfigFromGGUF: %v", err)
	}

	if cfg.EmbDim != 576 {
		t.Errorf("EmbDim: expected 576, got %d", cfg.EmbDim)
	}
	if cfg.NLayers != 30 {
		t.Errorf("NLayers: expected 30, got %d", cfg.NLayers)
	}
	if cfg.NHeads != 9 {
		t.Errorf("NHeads: expected 9, got %d", cfg.NHeads)
	}
	if cfg.NKVHeads != 3 {
		t.Errorf("NKVHeads: expected 3, got %d", cfg.NKVHeads)
	}
	if cfg.DFF != 1536 {
		t.Errorf("DFF: expected 1536, got %d", cfg.DFF)
	}
	if cfg.ContextLen != 8192 {
		t.Errorf("ContextLen: expected 8192, got %d", cfg.ContextLen)
	}
	if cfg.VocabSize != 49152 {
		t.Errorf("VocabSize: expected 49152, got %d", cfg.VocabSize)
	}
	if cfg.RopeDim != 64 {
		t.Errorf("RopeDim: expected 64, got %d", cfg.RopeDim)
	}
	if math.Abs(cfg.RopeTheta-100000.0) > 1e-6 {
		t.Errorf("RopeTheta: expected 100000, got %f", cfg.RopeTheta)
	}
}

func TestLoadConfigFromGGUF_MissingArch(t *testing.T) {
	buf := make([]byte, 128)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 0); offset += 8

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = LoadConfigFromGGUF(f)
	if err == nil {
		t.Fatal("expected error for missing architecture, got nil")
	}
}

func TestLoadConfigFromGGUF_MissingField(t *testing.T) {
	buf := make([]byte, 128)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 1); offset += 8

	offset = writeMetadataString(buf, offset, "general.architecture", "llama")

	path := writeGGUF(t, buf[:offset])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = LoadConfigFromGGUF(f)
	if err == nil {
		t.Fatal("expected error for missing field, got nil")
	}
}

type mockParamSetter struct {
	params map[string]*tensor.Tensor
}

func (m *mockParamSetter) SetParameter(path string, data *tensor.Tensor) {
	m.params[path] = data
}

func TestLoadWeightsFromGGUF(t *testing.T) {
	buf := make([]byte, 2048)
	offset := 0

	writeUint32LE(buf, offset, gguf.Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 2); offset += 8 // tensor_count
	writeUint64LE(buf, offset, 1); offset += 8 // metadata_kv_count
	offset = writeMetadataString(buf, offset, "general.architecture", "llama")

	// 2D tensor: [3, 4] F32 (12 elements)
	offset = writeTensorInfo(buf, offset, "token_embd.weight", []uint64{3, 4}, gguf.TypeF32, 0)

	// 1D tensor: [5] F32 (5 elements)
	offset = writeTensorInfo(buf, offset, "output_norm.weight", []uint64{5}, gguf.TypeF32, 12*4)

	// Pad to alignment
	aligned := int(gguf.AlignOffset(int64(offset), gguf.DefaultAlign))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}
	dataStart := aligned

	// 2D tensor data: 3x4 F32 = 1..12
	for i := 0; i < 12; i++ {
		writeUint32LE(buf, dataStart+i*4, math.Float32bits(float32(i+1)))
	}

	// 1D tensor data: 5 F32 = 10..50 step 10
	for i := 0; i < 5; i++ {
		writeUint32LE(buf, dataStart+12*4+i*4, math.Float32bits(float32((i+1)*10)))
	}

	totalSize := dataStart + 12*4 + 5*4
	path := writeGGUF(t, buf[:totalSize])
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	mock := &mockParamSetter{params: make(map[string]*tensor.Tensor)}
	if err := LoadWeightsFromGGUF(mock, f); err != nil {
		t.Fatalf("LoadWeightsFromGGUF: %v", err)
	}

	// Check token_embd.weight (2D)
	te, ok := mock.params["token_embd.weight"]
	if !ok {
		t.Fatal("missing token_embd.weight")
	}
	dims := te.Dimensions()
	if dims[1] != 4 || dims[2] != 3 {
		t.Fatalf("token_embd dims: expected [1 4 3], got %v", dims)
	}
	// GGUF stores dimensions reversed: header [dim0, dim1], actual data
	// layout [dim1, dim0]. For header [3,4], actual data layout is [4,3].
	// Data access: t.Set(0, i, j, data[j*dim0+i]) = data[j*3+i]
	// After transpose: At(0, j, i) = data[j*3+i] = j*3 + i + 1
	for j := 0; j < 4; j++ {
		for i := 0; i < 3; i++ {
			got := te.At(0, j, i)
			want := float64(j*3 + i + 1)
			if got != want {
				t.Errorf("token_embd[%d,%d]: expected %f, got %f", j, i, want, got)
			}
		}
	}

	// Check output_norm.weight (1D)
	on, ok := mock.params["output_norm.weight"]
	if !ok {
		t.Fatal("missing output_norm.weight")
	}
	dims = on.Dimensions()
	if dims[2] != 5 {
		t.Fatalf("output_norm dims: expected [1 1 5], got %v", dims)
	}
	for i := 0; i < 5; i++ {
		got := on.At(0, 0, i)
		want := float64((i + 1) * 10)
		if got != want {
			t.Errorf("output_norm[%d]: expected %f, got %f", i, want, got)
		}
	}
}


