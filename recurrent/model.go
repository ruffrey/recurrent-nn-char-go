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
func InitLSTM(input_size int, hiddenSizes []int, output_size int) Model {
	model := Model{}
	var prev_size int
	var hidden_size int

	for d := 0; d < len(hiddenSizes); d++ { // loop over depths
		if d == 0 {
			prev_size = input_size
		} else {
			prev_size = hiddenSizes[d-1]
		}
		hidden_size = hiddenSizes[d]

		ds := strconv.Itoa(d)
		// gates parameters
		model["Wix"+ds] = RandMat(hidden_size, prev_size, 0, 0.08)
		model["Wih"+ds] = RandMat(hidden_size, hidden_size, 0, 0.08)
		model["bi"+ds] = NewMat(hidden_size, 1)
		model["Wfx"+ds] = RandMat(hidden_size, prev_size, 0, 0.08)
		model["Wfh"+ds] = RandMat(hidden_size, hidden_size, 0, 0.08)
		model["bf"+ds] = NewMat(hidden_size, 1)
		model["Wox"+ds] = RandMat(hidden_size, prev_size, 0, 0.08)
		model["Woh"+ds] = RandMat(hidden_size, hidden_size, 0, 0.08)
		model["bo"+ds] = NewMat(hidden_size, 1)
		// cell write params
		model["Wcx"+ds] = RandMat(hidden_size, prev_size, 0, 0.08)
		model["Wch"+ds] = RandMat(hidden_size, hidden_size, 0, 0.08)
		model["bc"+ds] = NewMat(hidden_size, 1)
	}
	// decoder params
	model["Whd"] = RandMat(output_size, hidden_size, 0, 0.08)
	model["bd"] = NewMat(output_size, 1)

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
*/
func ForwardLSTM(G *Graph, model Model, hiddenSizes []int, x Mat, prev CellMemory) CellMemory {
	var hidden_prevs []Mat
	var cell_prevs []Mat

	if prev.Hidden == nil {
		hidden_prevs := make([]Mat, len(hiddenSizes))
		cell_prevs = make([]Mat, len(hiddenSizes))
		for s := 0; s < len(hiddenSizes); s++ {
			hidden_prevs[s] = NewMat(hiddenSizes[s], 1)
			cell_prevs[s] = NewMat(hiddenSizes[s], 1)
		}
	} else {
		hidden_prevs = prev.Hidden
		cell_prevs = prev.Cell
	}

	var hidden []Mat
	var cell []Mat

	for d := 0; d < len(hiddenSizes); d++ {
		var input_vector Mat
		if d == 0 {
			input_vector = x
		} else {
			input_vector = hidden[d-1]
		}
		hidden_prev := hidden_prevs[d]
		cell_prev := cell_prevs[d]

		// ds is the index but as a string
		ds := strconv.Itoa(d)

		// input gate
		wixds := model["Wix"+ds]
		h0 := G.Mul(&wixds, &input_vector)
		wihds := model["Wih"+ds]
		h1 := G.Mul(&wihds, &hidden_prev)
		add1 := G.Add(&h0, &h1)
		bids := model["bi"+ds]
		add2 := G.Add(&add1, &bids)
		input_gate := G.Sigmoid(&add2)

		// forget gate
		wfxds := model["Wfx"+ds]
		h2 := G.Mul(&wfxds, &input_vector)
		wfhds := model["Wfh"+ds]
		h3 := G.Mul(&wfhds, &hidden_prev)
		add3 := G.Add(&h2, &h3)
		bfds := model["bf"+ds]
		add4 := G.Add(&add3, &bfds)
		forget_gate := G.Sigmoid(&add4)

		// output gate
		woxds := model["Wox"+ds]
		h4 := G.Mul(&woxds, &input_vector)
		wohds := model["Woh"+ds]
		h5 := G.Mul(&wohds, &hidden_prev)
		bods := model["bo"+ds]
		add45 := G.Add(&h4, &h5)
		add45bods := G.Add(&add45, &bods)
		output_gate := G.Sigmoid(&add45bods)

		// write operation on cells
		wcxds := model["Wcx"+ds]
		h6 := G.Mul(&wcxds, &input_vector)
		wchds := model["Wch"+ds]
		h7 := G.Mul(&wchds, &hidden_prev)
		add67 := G.Add(&h6, &h7)
		bcds := model["bc"+ds]
		add67bcds := G.Add(&add67, &bcds)
		cell_write := G.Tanh(&add67bcds)

		// compute new cell activation
		retain_cell := G.Eltmul(&forget_gate, &cell_prev) // what do we keep from cell
		write_cell := G.Eltmul(&input_gate, &cell_write)  // what do we write to cell
		cell_d := G.Add(&retain_cell, &write_cell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		tahncell_d := G.Tanh(&cell_d)
		hidden_d := G.Eltmul(&output_gate, &tahncell_d)

		hidden = append(hidden, hidden_d)
		cell = append(cell, cell_d)
	}

	// one decoder to outputs at end
	whd := model["Whd"]
	bd := model["bd"]
	whdlasthidden := G.Mul(&whd, &hidden[len(hidden)-1])
	output := G.Add(&whdlasthidden, &bd)

	// return cell memory, hidden representation and output
	return CellMemory{
		Hidden: hidden,
		Cell:   cell,
		Output: output,
	}
}
