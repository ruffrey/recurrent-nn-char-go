package main

import (
	"strconv"
	"github.com/ruffrey/recurrent-nn-char-go/mat32"
)

/*
Model is the graph model.
*/
type Model map[string]*mat32.Mat

/*
CellMemory is apparently passed around during foward LSTM sessions.
*/
type CellMemory struct {
	Hidden []*mat32.Mat
	Cell   []*mat32.Mat
	Output *mat32.Mat
}

/*
NewLSTMModel initializes a Long Short Term Memory Recurrent Neural Network model.
*/
func NewLSTMModel(inputSize int, hiddenSizes []int, outputSize int) Model {
	model := Model{}
	var prevSize int
	var hiddenSize int

	for d := 0; d < len(hiddenSizes); d++ { // loop over depths
		if d == 0 {
			prevSize = inputSize
		} else {
			prevSize = hiddenSizes[d-1]
		}
		hiddenSize = hiddenSizes[d]

		ds := strconv.Itoa(d)
		// gates parameters
		model["Wix"+ds] = mat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Wih"+ds] = mat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bi"+ds] = mat32.NewMat(hiddenSize, 1)
		model["Wfx"+ds] = mat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Wfh"+ds] = mat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bf"+ds] = mat32.NewMat(hiddenSize, 1)
		model["Wox"+ds] = mat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Woh"+ds] = mat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bo"+ds] = mat32.NewMat(hiddenSize, 1)
		// cell write params
		model["Wcx"+ds] = mat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Wch"+ds] = mat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bc"+ds] = mat32.NewMat(hiddenSize, 1)
	}
	// decoder params
	model["Whd"] = mat32.RandMat(outputSize, hiddenSize, 0.08)
	model["bd"] = mat32.NewMat(outputSize, 1)

	return model
}
