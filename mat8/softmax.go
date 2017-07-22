package mat8

import (
	"math"
)

/*
Softmax computes the softmax of a matrix.
*/
func Softmax(m *Mat) *Mat {
	out := NewMat(m.RowCount, m.ColumnCount) // probability volume
	var maxval int8 = -127
	i := 0
	n := len(m.W)

	for ; i < n; i++ {
		if m.W[i] > maxval {
			maxval = m.W[i]
		}
	}
	maxval64 := float64(maxval)

	var s int8 = 0
	i = 0
	for ; i < n; i++ {
		out.W[i] = int8(math.Exp(float64(m.W[i]) - maxval64))
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
ArgmaxI seems to return the index of the max

Old comment: argmax of array w
*/
func ArgmaxI(w []int8) int {
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
func SampleArgmaxI(w []int8) int {
	r := Randf(0, 1)
	var x int8 = 0
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
