package recurrent

import "encoding/json"

/*
Mat holds a matrix.
*/
type Mat struct {
	RowCount    int
	ColumnCount int
	W           []float64
	DW          []float64
}

func (m *Mat) toJSON() (string, error) {
	b, err := json.Marshal(m)

	if err != nil {
		return "", err
	}

	return string(b[:]), err
}

/*
NewMat instantiates a new matrix.
*/
func NewMat(n int, d int) Mat {
	m := Mat{RowCount: n, ColumnCount: d}
	m.W = make([]float64, n*d)
	m.DW = make([]float64, n*d)
	return m
}

/*
RandMat fills a Mat with random numbers and returns it.
*/
func RandMat(n int, d int, mu int, std float64) Mat {
	m := NewMat(n, d)
	last := len(m.W)

	for i := 0; i < last; i++ {
		m.W[i] = Randf(-std, std)
	}

	return m
}
