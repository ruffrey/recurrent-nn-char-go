package main

import "encoding/json"

/*
Mat holds a matrix.
*/
type Mat struct {
	RowCount    int
	ColumnCount int
	W           []float32
	DW          []float32
}

func (m *Mat) toJSON() (string, error) {
	b, err := json.Marshal(m)

	if err != nil {
		return "", err
	}

	return string(b[:]), err
}

func zeros(size int) []float32 {
	// no need to initialize zero values
	return make([]float32, size)
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
func RandMat(n int, d int, mu int, std float32) *Mat {
	m := NewMat(n, d)
	last := len(m.W)

	for i := 0; i < last; i++ {
		m.W[i] = float32(Randf(-std, std))
	}

	return m
}
