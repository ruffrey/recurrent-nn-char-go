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
	h []Mat // hidden
	c []Mat // cell
	o Mat   // output
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

	if prev.h == nil {
		hidden_prevs := make([]Mat, len(hiddenSizes))
		cell_prevs = make([]Mat, len(hiddenSizes))
		for s := 0; s < len(hiddenSizes); s++ {
			hidden_prevs[s] = NewMat(hiddenSizes[s], 1)
			cell_prevs[s] = NewMat(hiddenSizes[s], 1)
		}
	} else {
		hidden_prevs = prev.h
		cell_prevs = prev.c
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

		ds := strconv.Itoa(d)

		// input gate
		h0 := G.Mul(&model["Wix"+ds], &input_vector)
		h1 := G.Mul(&model["Wih"+ds], &hidden_prev)
		add1 := G.Add(&h0, &h1)
		add2 := G.Add(&add1, model["bi"+ds])
		input_gate := G.Sigmoid(&add2)

		// forget gate
		h2 := G.Mul(&model["Wfx"+ds], &input_vector)
		h3 := G.Mul(&model["Wfh"+ds], &hidden_prev)
		forget_gate := G.Sigmoid(G.Add(G.Add(&h2, &h3), model["bf"+ds]))

		// output gate
		h4 := G.Mul(model["Wox"+ds], &input_vector)
		h5 := G.Mul(model["Woh"+ds], &hidden_prev)
		output_gate := G.Sigmoid(G.Add(G.Add(&h4, &h5), model["bo"+ds]))

		// write operation on cells
		h6 := G.Mul(model["Wcx"+ds], input_vector)
		h7 := G.Mul(model["Wch"+ds], hidden_prev)
		cell_write := G.Tanh(G.Add(G.Add(&h6, &h7), model["bc"+ds]))

		// compute new cell activation
		retain_cell := G.Eltmul(forget_gate, cell_prev) // what do we keep from cell
		write_cell := G.Eltmul(input_gate, cell_write)  // what do we write to cell
		cell_d := G.Add(retain_cell, write_cell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		hidden_d := G.Eltmul(output_gate, G.Tanh(cell_d))

		hidden = append(hidden, hidden_d)
		cell = append(cell, cell_d)
	}

	// one decoder to outputs at end
	output := G.add(G.Mul(model["Whd"], hidden[hidden.length-1]), model["bd"])

	// return cell memory, hidden representation and output
	return CellMemory{
		h: hidden,
		c: cell,
		o: output,
	}
}
