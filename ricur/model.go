package main

import (
	"strconv"
	"github.com/ruffrey/recurrent-nn-char-go/mat8"
)

/*
Model is the graph model.
*/
type Model map[string]*mat8.Mat

/*
CellMemory is apparently passed around during foward LSTM sessions.
*/
type CellMemory struct {
	Hidden []*mat8.Mat
	Cell   []*mat8.Mat
	Output *mat8.Mat
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
		model["Wix"+ds] = mat8.RandMat(hiddenSize, prevSize, 0, defaultGateWeight)
		model["Wih"+ds] = mat8.RandMat(hiddenSize, hiddenSize, 0, defaultGateWeight)
		model["bi"+ds] = mat8.NewMat(hiddenSize, 1)
		model["Wfx"+ds] = mat8.RandMat(hiddenSize, prevSize, 0, defaultGateWeight)
		model["Wfh"+ds] = mat8.RandMat(hiddenSize, hiddenSize, 0, defaultGateWeight)
		model["bf"+ds] = mat8.NewMat(hiddenSize, 1)
		model["Wox"+ds] = mat8.RandMat(hiddenSize, prevSize, 0, defaultGateWeight)
		model["Woh"+ds] = mat8.RandMat(hiddenSize, hiddenSize, 0, defaultGateWeight)
		model["bo"+ds] = mat8.NewMat(hiddenSize, 1)
		// cell write params
		model["Wcx"+ds] = mat8.RandMat(hiddenSize, prevSize, 0, defaultGateWeight)
		model["Wch"+ds] = mat8.RandMat(hiddenSize, hiddenSize, 0, defaultGateWeight)
		model["bc"+ds] = mat8.NewMat(hiddenSize, 1)
	}
	// decoder params
	model["Whd"] = mat8.RandMat(outputSize, hiddenSize, 0, defaultGateWeight)
	model["bd"] = mat8.NewMat(outputSize, 1)

	return model
}
