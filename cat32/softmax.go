package cat32

import (
	"math"

	"github.com/bjwbell/gensimd/simd"
)

/*
Softmax computes the softmax of a matrix, I guess.
*/
func Softmax(m *Mat) *Mat {
	out := NewMat(m.RowCount, m.ColumnCount) // probability volume
	var maxval float32 = -999999.0
	i := 0
	n := len(m.W)
	f := 0

	for ; i < n; i++ {
		for f = 0; f < 4; f++ {
			if m.W[i][f] > maxval {
				maxval = m.W[i][f]
			}
		}
	}

	var s float32 = 0.0
	i = 0
	for ; i < n; i++ {
		for f = 0; f < 4; f++ {
			out.W[i][f] = float32(math.Exp(float64(m.W[i][f]) - float64(maxval)))
			s += out.W[i][f]
		}
	}

	i = 0
	for ; i < n; i++ {
		for f = 0; f < 4; f++ {
			out.W[i][f] /= s
		}
	}

	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out
}

/*
SampleArgmaxI does something with sampling and integers, maybe.

Old comment: sample argmax from w, assuming w are probabilities that sum to one
*/
func SampleArgmaxI(w []simd.F32x4) int {
	r := Randf(0, 1)
	var x float32 = 0.0
	i := 0
	f := 0
	for {
		for f = 0; f < 4; f++ {
			x += w[i][f]
			if x > r {
				return i
			}
		}
		i++
	}
	return len(w) - 1
}
