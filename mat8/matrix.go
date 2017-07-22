package mat8

import "encoding/json"

/*
Mat holds a matrix.
*/
type Mat struct {
	RowCount    int
	ColumnCount int
	W           []int8
	DW          []int8
}

func (m *Mat) toJSON() (string, error) {
	b, err := json.Marshal(m)

	if err != nil {
		return "", err
	}

	return string(b[:]), err
}

func zeros(size int) []int8 {
	// no need to initialize zero values
	return make([]int8, size)
}

/*
NewMat instantiates a new matrix.
*/
func NewMat(n int, d int) *Mat {
	m := Mat{RowCount: n, ColumnCount: d}
	m.W = zeros(n * d)
	m.DW = zeros(n * d)
	return &m
}

/*
RandMat fills a Mat with random numbers and returns it.
*/
func RandMat(n int, d int, mu int, std int8) *Mat {
	m := NewMat(n, d)
	last := len(m.W)

	for i := 0; i < last; i++ {
		m.W[i] = Randf(-std, std)
	}

	return m
}
