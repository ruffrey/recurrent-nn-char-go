package main

import (
	"strconv"
	"github.com/ruffrey/recurrent-nn-char-go/cat32"
)

/*
Model is the graph model.
*/
type Model map[string]*cat32.Mat

/*
CellMemory is apparently passed around during foward LSTM sessions.
*/
type CellMemory struct {
	Hidden []*cat32.Mat
	Cell   []*cat32.Mat
	Output *cat32.Mat
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
		model["Wix"+ds] = cat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Wih"+ds] = cat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bi"+ds] = cat32.NewMat(hiddenSize, 1)
		model["Wfx"+ds] = cat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Wfh"+ds] = cat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bf"+ds] = cat32.NewMat(hiddenSize, 1)
		model["Wox"+ds] = cat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Woh"+ds] = cat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bo"+ds] = cat32.NewMat(hiddenSize, 1)
		// cell write params
		model["Wcx"+ds] = cat32.RandMat(hiddenSize, prevSize, 0.08)
		model["Wch"+ds] = cat32.RandMat(hiddenSize, hiddenSize, 0.08)
		model["bc"+ds] = cat32.NewMat(hiddenSize, 1)
	}
	// decoder params
	model["Whd"] = cat32.RandMat(outputSize, hiddenSize, 0.08)
	model["bd"] = cat32.NewMat(outputSize, 1)

	return model
}
