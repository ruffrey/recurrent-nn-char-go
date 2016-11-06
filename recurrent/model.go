package recurrent

import "strconv"

/*
Model is the graph model.
*/
type Model map[string]Mat

/*
CellMemory is apparently passed around during foward LSTM sessions.
*/
type CellMemory struct {
	Hidden []Mat
	Cell   []Mat
	Output Mat
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
		model["Wix"+ds] = RandMat(hiddenSize, prevSize, 0, 0.08)
		model["Wih"+ds] = RandMat(hiddenSize, hiddenSize, 0, 0.08)
		model["bi"+ds] = NewMat(hiddenSize, 1)
		model["Wfx"+ds] = RandMat(hiddenSize, prevSize, 0, 0.08)
		model["Wfh"+ds] = RandMat(hiddenSize, hiddenSize, 0, 0.08)
		model["bf"+ds] = NewMat(hiddenSize, 1)
		model["Wox"+ds] = RandMat(hiddenSize, prevSize, 0, 0.08)
		model["Woh"+ds] = RandMat(hiddenSize, hiddenSize, 0, 0.08)
		model["bo"+ds] = NewMat(hiddenSize, 1)
		// cell write params
		model["Wcx"+ds] = RandMat(hiddenSize, prevSize, 0, 0.08)
		model["Wch"+ds] = RandMat(hiddenSize, hiddenSize, 0, 0.08)
		model["bc"+ds] = NewMat(hiddenSize, 1)
	}
	// decoder params
	model["Whd"] = RandMat(outputSize, hiddenSize, 0, 0.08)
	model["bd"] = NewMat(outputSize, 1)

	return model
}
