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
func ForwardLSTM(G *Graph, model Model, hiddenSizes []int, x Mat, prev *CellMemory) CellMemory {
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
		wixds := model["Wix"+ds]
		h0 := G.Mul(&wixds, &inputVector)
		wihds := model["Wih"+ds]
		h1 := G.Mul(&wihds, &hiddenPrev)
		add1 := G.Add(&h0, &h1)
		bids := model["bi"+ds]
		add2 := G.Add(&add1, &bids)
		inputGate := G.Sigmoid(&add2)

		// forget gate
		wfxds := model["Wfx"+ds]
		h2 := G.Mul(&wfxds, &inputVector)
		wfhds := model["Wfh"+ds]
		h3 := G.Mul(&wfhds, &hiddenPrev)
		add3 := G.Add(&h2, &h3)
		bfds := model["bf"+ds]
		add4 := G.Add(&add3, &bfds)
		forgetGate := G.Sigmoid(&add4)

		// output gate
		woxds := model["Wox"+ds]
		h4 := G.Mul(&woxds, &inputVector)
		wohds := model["Woh"+ds]
		h5 := G.Mul(&wohds, &hiddenPrev)
		bods := model["bo"+ds]
		add45 := G.Add(&h4, &h5)
		add45bods := G.Add(&add45, &bods)
		outputGate := G.Sigmoid(&add45bods)

		// write operation on cells
		wcxds := model["Wcx"+ds]
		h6 := G.Mul(&wcxds, &inputVector)
		wchds := model["Wch"+ds]
		h7 := G.Mul(&wchds, &hiddenPrev)
		add67 := G.Add(&h6, &h7)
		bcds := model["bc"+ds]
		add67bcds := G.Add(&add67, &bcds)
		cellWrite := G.Tanh(&add67bcds)

		// compute new cell activation
		retainCell := G.Eltmul(&forgetGate, &cellPrev) // what do we keep from cell
		writeCell := G.Eltmul(&inputGate, &cellWrite)  // what do we write to cell
		cellD := G.Add(&retainCell, &writeCell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		tahncellD := G.Tanh(&cellD)
		hiddenD := G.Eltmul(&outputGate, &tahncellD)

		hidden = append(hidden, hiddenD)
		cell = append(cell, cellD)

		// TODO: clear pointer leaks?
	}

	// one decoder to outputs at end
	whd := model["Whd"]
	bd := model["bd"]
	whdlasthidden := G.Mul(&whd, &hidden[len(hidden)-1])
	output := G.Add(&whdlasthidden, &bd)

	prev = nil // avoid leaks
	G = nil    // avoid leaks

	// return cell memory, hidden representation and output
	return CellMemory{
		Hidden: hidden,
		Cell:   cell,
		Output: output,
	}
}
