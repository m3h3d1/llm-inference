package gguf

import (
	"math"
	"os"
	"path/filepath"
	"testing"
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
	writeUint32LE(b, offset, uint32(TypeUINT32))
	offset += 4
	writeUint32LE(b, offset, value)
	return offset + 4
}

func writeMetadataFloat32(b []byte, offset int, key string, value float32) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeFLOAT32))
	offset += 4
	writeUint32LE(b, offset, math.Float32bits(value))
	return offset + 4
}

func writeMetadataString(b []byte, offset int, key, value string) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeSTRING))
	offset += 4
	return writeString(b, offset, value)
}

func writeTensorInfo(b []byte, offset int, name string, dims []uint64, typ TensorType, dataOffset uint64) int {
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

func TestSyntheticGGUF(t *testing.T) {
	buf := make([]byte, 1024)
	offset := 0

	writeUint32LE(buf, offset, Magic)
	offset += 4
	writeUint32LE(buf, offset, 3)
	offset += 4
	writeUint64LE(buf, offset, 2) // tensor_count
	offset += 8
	writeUint64LE(buf, offset, 3) // metadata_kv_count
	offset += 8

	offset = writeMetadataString(buf, offset, "general.architecture", "llama")
	offset = writeMetadataUint32(buf, offset, "llama.block_count", 30)
	offset = writeMetadataFloat32(buf, offset, "test.float", 3.14)

	tensorAOff := uint64(0)
	offset = writeTensorInfo(buf, offset, "test_tensor_a", []uint64{2, 3}, TypeF32, tensorAOff)
	tensorBOff := uint64(6 * 4) // 6 F32 elements
	offset = writeTensorInfo(buf, offset, "tensor_b", []uint64{4}, TypeF16, tensorBOff)

	// Pad to alignment (32)
	aligned := int(AlignOffset(int64(offset), DefaultAlign))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}
	dataStart := aligned

	// Tensor A data: 6 x F32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	f32vals := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	for i, v := range f32vals {
		writeUint32LE(buf, dataStart+i*4, math.Float32bits(v))
	}

	// Tensor B data: 4 x F16 = [0.0, 1.0, -2.0, 65504.0]
	f16vals := []uint16{0x0000, 0x3C00, 0xC000, 0x7BFF}
	for i, v := range f16vals {
		writeUint16LE(buf, dataStart+6*4+i*2, v)
	}

	path := writeGGUF(t, buf[:dataStart+6*4+4*2])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if f.Version != 3 {
		t.Errorf("expected version 3, got %d", f.Version)
	}
	if len(f.TensorInfos) != 2 {
		t.Errorf("expected 2 tensor infos, got %d", len(f.TensorInfos))
	}

	arch, ok := f.Metadata["general.architecture"]
	if !ok {
		t.Fatal("missing general.architecture")
	}
	archStr, _ := arch.String()
	if archStr != "llama" {
		t.Errorf("expected architecture 'llama', got %q", archStr)
	}

	bc, ok := f.Metadata["llama.block_count"]
	if !ok {
		t.Fatal("missing llama.block_count")
	}
	bcVal, _ := bc.Uint64()
	if bcVal != 30 {
		t.Errorf("expected block_count 30, got %d", bcVal)
	}

	tf, ok := f.Metadata["test.float"]
	if !ok {
		t.Fatal("missing test.float")
	}
	tfVal, _ := tf.Float64()
	if math.Abs(tfVal-3.14) > 1e-6 {
		t.Errorf("expected test.float 3.14, got %f", tfVal)
	}

	if f.TensorInfos[0].Name != "test_tensor_a" {
		t.Errorf("expected tensor 0 name 'test_tensor_a', got %q", f.TensorInfos[0].Name)
	}
	if len(f.TensorInfos[0].Dimensions) != 2 || f.TensorInfos[0].Dimensions[0] != 2 || f.TensorInfos[0].Dimensions[1] != 3 {
		t.Errorf("tensor 0 dims: %v", f.TensorInfos[0].Dimensions)
	}

	dataA, err := f.ReadTensor(f.TensorInfos[0])
	if err != nil {
		t.Fatalf("ReadTensor A: %v", err)
	}
	if len(dataA) != 6 {
		t.Fatalf("expected 6 elements, got %d", len(dataA))
	}
	expectedA := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	for i := range expectedA {
		if dataA[i] != expectedA[i] {
			t.Errorf("tensor A[%d]: expected %f, got %f", i, expectedA[i], dataA[i])
		}
	}

	dataB, err := f.ReadTensor(f.TensorInfos[1])
	if err != nil {
		t.Fatalf("ReadTensor B: %v", err)
	}
	if len(dataB) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(dataB))
	}
	expectedB := []float64{0.0, 1.0, -2.0, 65504.0}
	for i := range expectedB {
		diff := dataB[i] - expectedB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Errorf("tensor B[%d]: expected %f, got %f", i, expectedB[i], dataB[i])
		}
	}
}

func TestF16Conversion(t *testing.T) {
	cases := []struct {
		bits uint16
		want float64
	}{
		{0x0000, 0.0},
		{0x3C00, 1.0},
		{0x4000, 2.0},
		{0xC000, -2.0},
		{0xBC00, -1.0},
		{0x7BFF, 65504.0},  // max finite
		{0xFBFF, -65504.0}, // min finite
		{0x7C00, math.Inf(1)},
		{0xFC00, math.Inf(-1)},
		{0x7C01, math.NaN()}, // NaN signaling
		{0x3555, 0.33325195}, // ~1/3
	}
	for _, c := range cases {
		got := f16ToF64(c.bits)
		if math.IsNaN(c.want) {
			if !math.IsNaN(got) {
				t.Errorf("f16(0x%04X): expected NaN, got %f", c.bits, got)
			}
		} else if math.Abs(got-c.want) > 1e-6 {
			t.Errorf("f16(0x%04X): expected %f, got %f", c.bits, c.want, got)
		}
	}
}

func TestQ8_0Dequantization(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint64LE(buf, offset, 0); offset += 8

	offset = writeTensorInfo(buf, offset, "q8_tensor", []uint64{4}, TypeQ8_0, 0)

	aligned := int(AlignOffset(int64(offset), DefaultAlign))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}
	dataStart := aligned

	// Q8_0: 1 block of 4 elements (padded to 32 but we'll use 4 actual)
	// Scale = F16 1.0 = 0x3C00
	writeUint16LE(buf, dataStart, 0x3C00)
	// 4 weights: 10, 20, -30, 127
	buf[dataStart+2] = 10
	buf[dataStart+3] = 20
	buf[dataStart+4] = byte(256 - 30) // -30 as uint8
	buf[dataStart+5] = 127

	path := writeGGUF(t, buf[:dataStart+34])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	data, err := f.ReadTensor(f.TensorInfos[0])
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	if len(data) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(data))
	}

	expected := []float64{10.0, 20.0, -30.0, 127.0}
	for i := range expected {
		if data[i] != expected[i] {
			t.Errorf("data[%d]: expected %f, got %f", i, expected[i], data[i])
		}
	}
}

func TestUnsupportedTensorType(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint64LE(buf, offset, 0); offset += 8

	offset = writeTensorInfo(buf, offset, "bad_tensor", []uint64{4}, TensorType(99), 0)

	aligned := int(AlignOffset(int64(offset), DefaultAlign))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}

	path := writeGGUF(t, buf[:aligned])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = f.ReadTensor(f.TensorInfos[0])
	if err == nil {
		t.Fatal("expected error for unsupported tensor type, got nil")
	}
}

func TestCustomAlignment(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint64LE(buf, offset, 1); offset += 8

	offset = writeMetadataUint32(buf, offset, "general.alignment", 64)

	offset = writeTensorInfo(buf, offset, "t", []uint64{1}, TypeF32, 0)

	aligned := int(AlignOffset(int64(offset), 64))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}

	dataStart := aligned

	writeUint32LE(buf, dataStart, math.Float32bits(42.0))

	path := writeGGUF(t, buf[:dataStart+4])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	if f.alignment != 64 {
		t.Errorf("expected alignment 64, got %d", f.alignment)
	}

	if f.tensorDataOffset != int64(aligned) {
		t.Errorf("expected tensorDataOffset %d, got %d", aligned, f.tensorDataOffset)
	}

	data, err := f.ReadTensor(f.TensorInfos[0])
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	if len(data) != 1 || data[0] != 42.0 {
		t.Errorf("expected [42.0], got %v", data)
	}
}

func TestInvalidMagic(t *testing.T) {
	buf := make([]byte, 24)
	path := writeGGUF(t, buf)
	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestEmptyMetadata(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 0); offset += 8

	path := writeGGUF(t, buf[:offset])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(f.Metadata) != 0 {
		t.Errorf("expected empty metadata, got %d entries", len(f.Metadata))
	}
	if len(f.TensorInfos) != 0 {
		t.Errorf("expected 0 tensor infos, got %d", len(f.TensorInfos))
	}
}

func TestMetadataArray(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 1); offset += 8

	offset = writeString(buf, offset, "test.array")
	writeUint32LE(buf, offset, uint32(TypeARRAY)); offset += 4
	writeUint32LE(buf, offset, uint32(TypeINT32)); offset += 4
	writeUint64LE(buf, offset, 3); offset += 8
	writeUint32LE(buf, offset, 10); offset += 4
	writeUint32LE(buf, offset, 20); offset += 4
	writeUint32LE(buf, offset, 30); offset += 4

	path := writeGGUF(t, buf[:offset])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	v, ok := f.Metadata["test.array"]
	if !ok {
		t.Fatal("missing test.array")
	}
	arr, ok := v.Array()
	if !ok {
		t.Fatal("test.array is not an array")
	}
	if len(arr) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(arr))
	}
	for i, expected := range []int64{10, 20, 30} {
		got, _ := arr[i].Int64()
		if got != expected {
			t.Errorf("arr[%d]: expected %d, got %d", i, expected, got)
		}
	}
}

func writeMetadataUint8(b []byte, offset int, key string, value uint8) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeUINT8))
	offset += 4
	b[offset] = value
	return offset + 1
}

func writeMetadataInt8(b []byte, offset int, key string, value int8) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeINT8))
	offset += 4
	b[offset] = byte(value)
	return offset + 1
}

func writeMetadataUint16(b []byte, offset int, key string, value uint16) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeUINT16))
	offset += 4
	writeUint16LE(b, offset, value)
	return offset + 2
}

func writeMetadataInt16(b []byte, offset int, key string, value int16) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeINT16))
	offset += 4
	writeUint16LE(b, offset, uint16(value))
	return offset + 2
}

func writeMetadataInt32(b []byte, offset int, key string, value int32) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeINT32))
	offset += 4
	writeUint32LE(b, offset, uint32(value))
	return offset + 4
}

func writeMetadataInt64(b []byte, offset int, key string, value int64) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeINT64))
	offset += 4
	writeUint64LE(b, offset, uint64(value))
	return offset + 8
}

func writeMetadataUint64(b []byte, offset int, key string, value uint64) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeUINT64))
	offset += 4
	writeUint64LE(b, offset, value)
	return offset + 8
}

func writeMetadataFloat64(b []byte, offset int, key string, value float64) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeFLOAT64))
	offset += 4
	writeUint64LE(b, offset, math.Float64bits(value))
	return offset + 8
}

func writeMetadataBool(b []byte, offset int, key string, value bool) int {
	offset = writeString(b, offset, key)
	writeUint32LE(b, offset, uint32(TypeBOOL))
	offset += 4
	if value {
		b[offset] = 1
	} else {
		b[offset] = 0
	}
	return offset + 1
}

func TestReadValueTypes(t *testing.T) {
	buf := make([]byte, 1024)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 0); offset += 8
	writeUint64LE(buf, offset, 11); offset += 8

	offset = writeMetadataUint8(buf, offset, "val_uint8", 200)
	offset = writeMetadataInt8(buf, offset, "val_int8", -50)
	offset = writeMetadataUint16(buf, offset, "val_uint16", 60000)
	offset = writeMetadataInt16(buf, offset, "val_int16", -20000)
	offset = writeMetadataInt32(buf, offset, "val_int32", -100000)
	offset = writeMetadataUint32(buf, offset, "val_uint32", 3000000000)
	offset = writeMetadataInt64(buf, offset, "val_int64", -5000000000000)
	offset = writeMetadataUint64(buf, offset, "val_uint64", 10000000000000)
	offset = writeMetadataFloat64(buf, offset, "val_float64", 3.141592653589793)
	offset = writeMetadataBool(buf, offset, "val_bool_true", true)
	offset = writeMetadataBool(buf, offset, "val_bool_false", false)

	path := writeGGUF(t, buf[:offset])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	cases := []struct {
		key string
		check func(t *testing.T, v Value)
	}{
		{"val_uint8", func(t *testing.T, v Value) {
			n, ok := v.Uint64()
			if !ok || n != 200 {
				t.Errorf("uint8: expected 200, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_int8", func(t *testing.T, v Value) {
			n, ok := v.Int64()
			if !ok || n != -50 {
				t.Errorf("int8: expected -50, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_uint16", func(t *testing.T, v Value) {
			n, ok := v.Uint64()
			if !ok || n != 60000 {
				t.Errorf("uint16: expected 60000, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_int16", func(t *testing.T, v Value) {
			n, ok := v.Int64()
			if !ok || n != -20000 {
				t.Errorf("int16: expected -20000, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_int32", func(t *testing.T, v Value) {
			n, ok := v.Int64()
			if !ok || n != -100000 {
				t.Errorf("int32: expected -100000, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_uint32", func(t *testing.T, v Value) {
			n, ok := v.Uint64()
			if !ok || n != 3000000000 {
				t.Errorf("uint32: expected 3000000000, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_int64", func(t *testing.T, v Value) {
			n, ok := v.Int64()
			if !ok || n != -5000000000000 {
				t.Errorf("int64: expected -5000000000000, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_uint64", func(t *testing.T, v Value) {
			n, ok := v.Uint64()
			if !ok || n != 10000000000000 {
				t.Errorf("uint64: expected 10000000000000, got %d (ok=%v)", n, ok)
			}
		}},
		{"val_float64", func(t *testing.T, v Value) {
			n, ok := v.Float64()
			if !ok || math.Abs(n-3.141592653589793) > 1e-15 {
				t.Errorf("float64: expected ~3.14159, got %f (ok=%v)", n, ok)
			}
		}},
		{"val_bool_true", func(t *testing.T, v Value) {
			b, ok := v.Bool()
			if !ok || !b {
				t.Errorf("bool true: expected true, got %v (ok=%v)", b, ok)
			}
		}},
		{"val_bool_false", func(t *testing.T, v Value) {
			b, ok := v.Bool()
			if !ok || b {
				t.Errorf("bool false: expected false, got %v (ok=%v)", b, ok)
			}
		}},
	}

	for _, c := range cases {
		t.Run(c.key, func(t *testing.T) {
			v, ok := f.Metadata[c.key]
			if !ok {
				t.Fatal("missing metadata key")
			}
			c.check(t, v)
		})
	}
}

func TestFileTooSmall(t *testing.T) {
	buf := make([]byte, 10)
	path := writeGGUF(t, buf)
	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error for file too small")
	}
}

func TestUnsupportedVersion(t *testing.T) {
	buf := make([]byte, 24)
	writeUint32LE(buf, 0, Magic)
	writeUint32LE(buf, 4, 4) // version 4
	path := writeGGUF(t, buf)
	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error for unsupported version")
	}
}

func TestReadTensorTruncated(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint64LE(buf, offset, 0); offset += 8

	offset = writeTensorInfo(buf, offset, "truncated", []uint64{10}, TypeF32, 0)

	aligned := int(AlignOffset(int64(offset), DefaultAlign))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}

	// Only write 20 bytes of data (need 40 for 10 F32)
	dataStart := aligned
	for i := 0; i < 5; i++ {
		writeUint32LE(buf, dataStart+i*4, math.Float32bits(float32(i)))
	}

	path := writeGGUF(t, buf[:dataStart+20])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = f.ReadTensor(f.TensorInfos[0])
	if err == nil {
		t.Fatal("expected error for truncated tensor data")
	}
}

func TestReadTensorQ4_0(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint64LE(buf, offset, 0); offset += 8

	offset = writeTensorInfo(buf, offset, "q4_tensor", []uint64{4}, TypeQ4_0, 0)

	aligned := int(AlignOffset(int64(offset), DefaultAlign))
	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}

	path := writeGGUF(t, buf[:aligned])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, err = f.ReadTensor(f.TensorInfos[0])
	if err == nil {
		t.Fatal("expected error for Q4_0 unsupported type")
	}
}

func TestBoolValue(t *testing.T) {
	v := Value{Type: TypeBOOL, data: true}
	b, ok := v.Bool()
	if !ok || !b {
		t.Errorf("Bool true: expected true, got %v (ok=%v)", b, ok)
	}
	v2 := Value{Type: TypeBOOL, data: false}
	b2, ok2 := v2.Bool()
	if !ok2 || b2 {
		t.Errorf("Bool false: expected false, got %v (ok=%v)", b2, ok2)
	}
	// Non-bool Value returns ok=false
	v3 := Value{Type: TypeUINT32, data: uint64(1)}
	_, ok3 := v3.Bool()
	if ok3 {
		t.Error("UINT32 Bool should return ok=false")
	}
}

func TestTypeSize(t *testing.T) {
	if got := typeSize(TypeF32); got != 4 {
		t.Errorf("F32 size: expected 4, got %d", got)
	}
	if got := typeSize(TypeF16); got != 2 {
		t.Errorf("F16 size: expected 2, got %d", got)
	}
	if got := typeSize(TypeQ8_0); got != 1 {
		t.Errorf("Q8_0 size: expected 1, got %d", got)
	}
	if got := typeSize(TensorType(99)); got != 0 {
		t.Errorf("unknown type: expected 0, got %d", got)
	}
}

func TestCustomAlignmentOne(t *testing.T) {
	buf := make([]byte, 512)
	offset := 0

	writeUint32LE(buf, offset, Magic); offset += 4
	writeUint32LE(buf, offset, 3); offset += 4
	writeUint64LE(buf, offset, 1); offset += 8
	writeUint64LE(buf, offset, 1); offset += 8

	offset = writeMetadataUint32(buf, offset, "general.alignment", 1)

	offset = writeTensorInfo(buf, offset, "t", []uint64{1}, TypeF32, 0)

	aligned := int(AlignOffset(int64(offset), 1))
	if aligned != offset {
		t.Errorf("alignment=1: expected dataStart=%d, got %d", offset, aligned)
	}

	for i := offset; i < aligned; i++ {
		buf[i] = 0
	}

	writeUint32LE(buf, aligned, math.Float32bits(99.0))

	path := writeGGUF(t, buf[:aligned+4])
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	data, err := f.ReadTensor(f.TensorInfos[0])
	if err != nil {
		t.Fatalf("ReadTensor: %v", err)
	}
	if len(data) != 1 || data[0] != 99.0 {
		t.Errorf("expected [99.0], got %v", data)
	}
}

func TestReadString(t *testing.T) {
	buf := make([]byte, 20)
	_ = writeString(buf, 0, "hello")
	got, newOffset := readString(buf, 0)
	if got != "hello" {
		t.Errorf("expected 'hello', got %q", got)
	}
	if newOffset != 5+8 {
		t.Errorf("expected offset 13, got %d", newOffset)
	}

	writeUint64LE(buf, 0, 0)
	got2, _ := readString(buf, 0)
	if got2 != "" {
		t.Errorf("expected empty string, got %q", got2)
	}
}
