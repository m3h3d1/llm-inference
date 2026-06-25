package gguf

import (
	"encoding/binary"
	"math"
)

const (
	Magic       uint32 = 0x46554747
	DefaultAlign       = 32
)

type ValueType uint32

const (
	TypeUINT8   ValueType = 0
	TypeINT8              = 1
	TypeUINT16            = 2
	TypeINT16             = 3
	TypeUINT32            = 4
	TypeINT32             = 5
	TypeFLOAT32           = 6
	TypeBOOL              = 7
	TypeSTRING            = 8
	TypeARRAY             = 9
	TypeUINT64            = 10
	TypeINT64             = 11
	TypeFLOAT64           = 12
)

type TensorType uint32

const (
	TypeF32  TensorType = 0
	TypeF16             = 1
	TypeQ4_0            = 2
	TypeQ8_0            = 8
)

type Value struct {
	Type ValueType
	data interface{}
}

func (v Value) Uint64() (uint64, bool) {
	x, ok := v.data.(uint64)
	return x, ok
}

func (v Value) Int64() (int64, bool) {
	x, ok := v.data.(int64)
	return x, ok
}

func (v Value) Float64() (float64, bool) {
	x, ok := v.data.(float64)
	return x, ok
}

func (v Value) String() (string, bool) {
	x, ok := v.data.(string)
	return x, ok
}

func (v Value) Array() ([]Value, bool) {
	x, ok := v.data.([]Value)
	return x, ok
}

func (v Value) Bool() (bool, bool) {
	x, ok := v.data.(bool)
	return x, ok
}

type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       TensorType
	Offset     uint64
}

func (t TensorInfo) NumElements() uint64 {
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}
	return n
}

func typeSize(typ TensorType) int {
	switch typ {
	case TypeF32:
		return 4
	case TypeF16:
		return 2
	case TypeQ8_0:
		return 1 // per element, stored in blocks of 32
	default:
		return 0
	}
}

func dequantizeQ8_0(data []byte, nElements int) []float64 {
	result := make([]float64, nElements)
	blockSize := 32
	for i := 0; i < nElements; i += blockSize {
		blockIdx := i / blockSize
		scaleBits := binary.LittleEndian.Uint16(data[blockIdx*34:])
		scale := f16ToF64(scaleBits)
		weights := data[blockIdx*34+2:]
		nInBlock := blockSize
		if i+nInBlock > nElements {
			nInBlock = nElements - i
		}
		for j := 0; j < nInBlock; j++ {
			result[i+j] = float64(int8(weights[j])) * scale
		}
	}
	return result
}

func AlignOffset(offset int64, alignment int64) int64 {
	if alignment <= 0 {
		alignment = DefaultAlign
	}
	mask := alignment - 1
	return (offset + mask) & ^mask
}

func f16ToF64(bits uint16) float64 {
	sign := (bits >> 15) & 1
	exp := (bits >> 10) & 0x1F
	mant := bits & 0x3FF

	if exp == 0 {
		if mant == 0 {
			if sign == 0 {
				return 0
			}
			return math.Float64frombits(1 << 63)
		}
		v := float64(mant) / 1024.0 * math.Exp2(-14)
		if sign == 1 {
			return -v
		}
		return v
	}
	if exp == 31 {
		if mant == 0 {
			if sign == 0 {
				return math.Inf(1)
			}
			return math.Inf(-1)
		}
		return math.NaN()
	}
	v := (1.0 + float64(mant)/1024.0) * math.Exp2(float64(exp)-15.0)
	if sign == 1 {
		return -v
	}
	return v
}

func readString(data []byte, offset int) (string, int) {
	length := binary.LittleEndian.Uint64(data[offset:])
	offset += 8
	str := string(data[offset : offset+int(length)])
	offset += int(length)
	return str, offset
}
