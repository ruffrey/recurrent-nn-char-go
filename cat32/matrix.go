package cat32

import (
	"github.com/bjwbell/gensimd/simd"
)

/*
Mat holds a matrix. It is in chunks of 4.
*/
type Mat struct {
	RowCount    int
	ColumnCount int
	W           []simd.F32x4
	DW          []simd.F32x4
}

/*
Value steps through the array of float32 chunks and returns
the value at the requested index.
*/
func (m *Mat) Value(index int) (val float32) {
	cursor := 0
	for i := 0; i < len(m.W); i++ {
		for f := 0; f < 4; f++ {
			if cursor >= index {
				val = m.W[i][f]
				break
			}
			cursor++
		}
	}
	return val
}

func zeros(size int) (m []simd.F32x4) {
	if size%4 != 0 {
		panic("mat size must be multiple of 4")
	}
	// no need to initialize zero values
	m = make([]simd.F32x4, (size/4)+1)
	return m
}

/*
NewMat instantiates a new matrix.
*/
func NewMat(n int, d int) (m *Mat) {
	m = &Mat{RowCount: n, ColumnCount: d}
	m.W = zeros(n * d)
	m.DW = zeros(n * d)
	return m
}

/*
RandMat fills a Mat with random numbers and returns it.
*/
func RandMat(n int, d int, std float32) (m *Mat) {
	m = NewMat(n, d)
	last := len(m.W)

	for i := 0; i < last; i++ {
		m.W[i] = simd.F32x4{
			float32(Randf(-std, std)),
			float32(Randf(-std, std)),
			float32(Randf(-std, std)),
			float32(Randf(-std, std)),
		}
	}

	return m
}
