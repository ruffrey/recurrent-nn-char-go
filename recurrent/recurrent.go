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
