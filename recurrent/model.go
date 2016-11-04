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
InitLSTM initializes a Long Short Term Memory Recurrent Neural Network model.
*/
func InitLSTM(inputSize int, hiddenSizes []int, outputSize int) Model {
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

/*
ForwardLSTM does things
forward prop for a single tick of LSTM

G is graph to append ops to
model contains LSTM parameters
x is 1D column vector with observation
prev is a struct containing hidden and cell
from previous iteration

This may be causing leaks
*/
func ForwardLSTM(G *Graph, model Model, hiddenSizes []int, x Mat, prev CellMemory) (Model, CellMemory) {
	var hiddenPrevs []Mat
	var cellPrevs []Mat

	// initialize when not yet initialized. we know there will always be hidden layers.
	if len(prev.Hidden) == 0 {
		hiddenPrevs = make([]Mat, len(hiddenSizes))
		cellPrevs = make([]Mat, len(hiddenSizes))
		for s := 0; s < len(hiddenSizes); s++ {
			hiddenPrevs[s] = NewMat(hiddenSizes[s], 1)
			cellPrevs[s] = NewMat(hiddenSizes[s], 1)
		}
	} else {
		hiddenPrevs = prev.Hidden
		cellPrevs = prev.Cell
	}

	var hidden []Mat
	var cell []Mat
	var inputVector Mat
	var hiddenPrev Mat
	var cellPrev Mat

	for d := 0; d < len(hiddenSizes); d++ {

		if d == 0 {
			inputVector = x
		} else {
			inputVector = hidden[d-1]
		}
		hiddenPrev = hiddenPrevs[d]
		cellPrev = cellPrevs[d]

		// ds is the index but as a string
		ds := strconv.Itoa(d)

		// input gate
		h0 := G.Mul(model["Wix"+ds], inputVector)
		h1 := G.Mul(model["Wih"+ds], hiddenPrev)
		add1 := G.Add(h0, h1)
		add2 := G.Add(add1, model["bi"+ds])
		inputGate := G.Sigmoid(add2)

		// forget gate
		h2 := G.Mul(model["Wfx"+ds], inputVector)
		h3 := G.Mul(model["Wfh"+ds], hiddenPrev)
		add3 := G.Add(h2, h3)
		add4 := G.Add(add3, model["bf"+ds])
		forgetGate := G.Sigmoid(add4)

		// output gate
		h4 := G.Mul(model["Wox"+ds], inputVector)
		h5 := G.Mul(model["Woh"+ds], hiddenPrev)
		add45 := G.Add(h4, h5)
		add45bods := G.Add(add45, model["bo"+ds])
		outputGate := G.Sigmoid(add45bods)

		// write operation on cells
		h6 := G.Mul(model["Wcx"+ds], inputVector)
		h7 := G.Mul(model["Wch"+ds], hiddenPrev)
		add67 := G.Add(h6, h7)
		add67bcds := G.Add(add67, model["bc"+ds])
		cellWrite := G.Tanh(add67bcds)

		// compute new cell activation
		retainCell := G.Eltmul(forgetGate, cellPrev) // what do we keep from cell
		writeCell := G.Eltmul(inputGate, cellWrite)  // what do we write to cell
		cellD := G.Add(retainCell, writeCell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		tahncellD := G.Tanh(cellD)
		hiddenD := G.Eltmul(outputGate, tahncellD)

		hidden = append(hidden, hiddenD)
		cell = append(cell, cellD)

		// TODO: clear pointer leaks?
	}

	// one decoder to outputs at end
	whdlasthidden := G.Mul(model["Whd"], hidden[len(hidden)-1])
	output := G.Add(whdlasthidden, model["bd"])

	G = nil // avoid leaks

	// return cell memory, hidden representation and output
	return model, CellMemory{
		Hidden: hidden,
		Cell:   cell,
		Output: output,
	}
}
