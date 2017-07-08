package recurrent

import "math"

/*
Assert ensures our code is not breaking down and halts the program.
*/
func Assert(assertion bool, msg string) {
	if !assertion {
		panic(msg)
	}
}

/*
Softmax computes the softmax of a matrix, I guess.
*/
func Softmax(m *Mat) Mat {
	out := NewMat(m.RowCount, m.ColumnCount) // probability volume
	maxval := -999999.0
	i := 0
	n := len(m.W)

	for ; i < n; i++ {
		if m.W[i] > maxval {
			maxval = m.W[i]
		}
	}

	s := 0.0
	i = 0
	for ; i < n; i++ {
		out.W[i] = math.Exp(m.W[i] - maxval)
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
func ArgmaxI(w []float64) int {
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
func SampleArgmaxI(w []float64) int {
	r := Randf(0, 1)
	x := 0.0
	i := 0

	for {
		x += w[i]
		if x > r {
			return i
		}
	}
	return len(w) - 1
}
