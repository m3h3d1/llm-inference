package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type File struct {
	Path             string
	Version          uint32
	Metadata         map[string]Value
	TensorInfos      []TensorInfo
	alignment        int64
	tensorDataOffset int64
	data             []byte
}

func Open(path string) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < 24 {
		return nil, fmt.Errorf("gguf: file too small (%d bytes)", len(data))
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	if magic != Magic {
		return nil, fmt.Errorf("gguf: invalid magic: 0x%08X", magic)
	}

	version := binary.LittleEndian.Uint32(data[4:8])
	if version < 1 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	tensorCount := binary.LittleEndian.Uint64(data[8:16])
	kvCount := binary.LittleEndian.Uint64(data[16:24])

	offset := 24

	metadata := make(map[string]Value, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, newOffset := readString(data, offset)
		offset = newOffset

		if offset+4 > len(data) {
			return nil, fmt.Errorf("gguf: truncated metadata value type at offset %d", offset)
		}
		valueType := ValueType(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		var value Value
		value, offset, err = readValue(data, offset, valueType)
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata key %q: %v", key, err)
		}
		metadata[key] = value
	}

	tensorInfos := make([]TensorInfo, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, newOffset := readString(data, offset)
		offset = newOffset

		if offset+4 > len(data) {
			return nil, fmt.Errorf("gguf: truncated tensor n_dimensions at offset %d", offset)
		}
		nDims := binary.LittleEndian.Uint32(data[offset:])
		offset += 4

		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			if offset+8 > len(data) {
				return nil, fmt.Errorf("gguf: truncated tensor dimension at offset %d", offset)
			}
			dims[d] = binary.LittleEndian.Uint64(data[offset:])
			offset += 8
		}

		if offset+4 > len(data) {
			return nil, fmt.Errorf("gguf: truncated tensor type at offset %d", offset)
		}
		tensorType := TensorType(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		if offset+8 > len(data) {
			return nil, fmt.Errorf("gguf: truncated tensor offset at offset %d", offset)
		}
		tensorOffset := binary.LittleEndian.Uint64(data[offset:])
		offset += 8

		tensorInfos = append(tensorInfos, TensorInfo{
			Name:       name,
			Dimensions: dims,
			Type:       tensorType,
			Offset:     tensorOffset,
		})
	}

	alignment := int64(DefaultAlign)
	if v, ok := metadata["general.alignment"]; ok {
		if n, ok := v.Uint64(); ok {
			alignment = int64(n)
		}
	}

	tensorDataOffset := AlignOffset(int64(offset), alignment)

	return &File{
		Path:             path,
		Version:          version,
		Metadata:         metadata,
		TensorInfos:      tensorInfos,
		alignment:        alignment,
		tensorDataOffset: tensorDataOffset,
		data:             data,
	}, nil
}

func (f *File) ReadTensor(info TensorInfo) ([]float64, error) {
	start := f.tensorDataOffset + int64(info.Offset)
	nElements := info.NumElements()

	switch info.Type {
	case TypeF32:
		expectedBytes := nElements * 4
		if start+int64(expectedBytes) > int64(len(f.data)) {
			return nil, fmt.Errorf("gguf: tensor %q data exceeds file size", info.Name)
		}
		result := make([]float64, nElements)
		for i := uint64(0); i < nElements; i++ {
			bits := binary.LittleEndian.Uint32(f.data[start+int64(i*4):])
			result[i] = float64(math.Float32frombits(bits))
		}
		return result, nil

	case TypeF16:
		expectedBytes := nElements * 2
		if start+int64(expectedBytes) > int64(len(f.data)) {
			return nil, fmt.Errorf("gguf: tensor %q data exceeds file size", info.Name)
		}
		result := make([]float64, nElements)
		for i := uint64(0); i < nElements; i++ {
			bits := binary.LittleEndian.Uint16(f.data[start+int64(i*2):])
			result[i] = f16ToF64(bits)
		}
		return result, nil

	case TypeQ8_0:
		// Q8_0: blocks of 32 elements; each block = F16 scale + 32×int8
		blockSize := 34 // 2 bytes scale + 32 bytes weights
		nBlocks := (nElements + 31) / 32
		expectedBytes := nBlocks * uint64(blockSize)
		if start+int64(expectedBytes) > int64(len(f.data)) {
			return nil, fmt.Errorf("gguf: tensor %q data exceeds file size", info.Name)
		}
		return dequantizeQ8_0(f.data[start:start+int64(expectedBytes)], int(nElements)), nil

	default:
		return nil, fmt.Errorf("gguf: unsupported tensor type %d for %q (only F32=0, F16=1, Q8_0=2 supported)", info.Type, info.Name)
	}
}

func readValue(data []byte, offset int, typ ValueType) (Value, int, error) {
	switch typ {
	case TypeUINT8:
		if offset+1 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated uint8 at offset %d", offset)
		}
		return Value{Type: typ, data: uint64(data[offset])}, offset + 1, nil

	case TypeINT8:
		if offset+1 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated int8 at offset %d", offset)
		}
		return Value{Type: typ, data: int64(int8(data[offset]))}, offset + 1, nil

	case TypeUINT16:
		if offset+2 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated uint16 at offset %d", offset)
		}
		return Value{Type: typ, data: uint64(binary.LittleEndian.Uint16(data[offset:]))}, offset + 2, nil

	case TypeINT16:
		if offset+2 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated int16 at offset %d", offset)
		}
		return Value{Type: typ, data: int64(int16(binary.LittleEndian.Uint16(data[offset:])))}, offset + 2, nil

	case TypeUINT32:
		if offset+4 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated uint32 at offset %d", offset)
		}
		return Value{Type: typ, data: uint64(binary.LittleEndian.Uint32(data[offset:]))}, offset + 4, nil

	case TypeINT32:
		if offset+4 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated int32 at offset %d", offset)
		}
		return Value{Type: typ, data: int64(int32(binary.LittleEndian.Uint32(data[offset:])))}, offset + 4, nil

	case TypeFLOAT32:
		if offset+4 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated float32 at offset %d", offset)
		}
		bits := binary.LittleEndian.Uint32(data[offset:])
		return Value{Type: typ, data: float64(math.Float32frombits(bits))}, offset + 4, nil

	case TypeBOOL:
		if offset+1 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated bool at offset %d", offset)
		}
		return Value{Type: typ, data: data[offset] != 0}, offset + 1, nil

	case TypeSTRING:
		str, newOffset := readString(data, offset)
		return Value{Type: typ, data: str}, newOffset, nil

	case TypeARRAY:
		if offset+4 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated array element type at offset %d", offset)
		}
		elemType := ValueType(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		if offset+8 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated array length at offset %d", offset)
		}
		count := binary.LittleEndian.Uint64(data[offset:])
		offset += 8

		elements := make([]Value, 0, count)
		for i := uint64(0); i < count; i++ {
			elem, newOffset, err := readValue(data, offset, elemType)
			if err != nil {
				return Value{}, 0, fmt.Errorf("array element %d: %v", i, err)
			}
			offset = newOffset
			elements = append(elements, elem)
		}
		return Value{Type: typ, data: elements}, offset, nil

	case TypeUINT64:
		if offset+8 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated uint64 at offset %d", offset)
		}
		return Value{Type: typ, data: binary.LittleEndian.Uint64(data[offset:])}, offset + 8, nil

	case TypeINT64:
		if offset+8 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated int64 at offset %d", offset)
		}
		return Value{Type: typ, data: int64(binary.LittleEndian.Uint64(data[offset:]))}, offset + 8, nil

	case TypeFLOAT64:
		if offset+8 > len(data) {
			return Value{}, 0, fmt.Errorf("truncated float64 at offset %d", offset)
		}
		bits := binary.LittleEndian.Uint64(data[offset:])
		return Value{Type: typ, data: math.Float64frombits(bits)}, offset + 8, nil

	default:
		return Value{}, 0, fmt.Errorf("unknown value type %d", typ)
	}
}
