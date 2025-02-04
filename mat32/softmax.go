package mat32

import (
	"math"
)

/*
Softmax computes the softmax of a matrix, I guess.
*/
func Softmax(m *Mat) *Mat {
	out := NewMat(m.RowCount, m.ColumnCount) // probability volume
	var maxval float32 = -999999.0
	i := 0
	n := len(m.W)

	for ; i < n; i++ {
		if m.W[i] > maxval {
			maxval = m.W[i]
		}
	}

	var s float32 = 0.0
	i = 0
	for ; i < n; i++ {
		out.W[i] = float32(math.Exp(float64(m.W[i]) - float64(maxval)))
		s += out.W[i]
	}

	i = 0
	for ; i < n; i++ {
		out.W[i] /= s
	}

	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out
}

/*
ArgmaxI does something with max and i. maybe integer.

Old comment: argmax of array w
*/
func ArgmaxI(w []float32) int {
	maxv := w[0]
	maxix := 0
	i := 1
	n := len(w)
	for ; i < n; i++ {
		v := w[i]
		if v > maxv {
			maxix = i
			maxv = v
		}
	}
	return maxix
}

/*
SampleArgmaxI does something with sampling and integers, maybe.

Old comment: sample argmax from w, assuming w are probabilities that sum to one
*/
func SampleArgmaxI(w []float32) int {
	r := Randf(0, 1)
	var x float32 = 0.0
	i := 0

	for {
		x += w[i]
		if x > r {
			return i
		}
		i++
	}
	return len(w) - 1
}
