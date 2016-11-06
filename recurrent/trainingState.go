package recurrent

import (
	"fmt"
	"strconv"
	"strings"
)

/*
TrainingState is the representation of the training data which gets saved or loaded
to disk between sessions.
*/
type TrainingState struct {
	HiddenSizes    []int
	LetterSize     int
	Model          Model
	G              Graph
	Solver         Solver
	LetterToIndex  map[string]int
	IndexToLetter  map[int]string
	Vocab          []string
	PerplexityList []float64

	// the following do not need to be persisted between training sessions
	EpochSize     int
	InputSize     int
	OutputSize    int
	DataSentences []string
	TickIterator  int
}

/*
ForwardIndex forwards the index
*/
func (state *TrainingState) ForwardIndex(ix int, prev CellMemory) CellMemory {
	x := state.G.RowPluck(state.Model["Wil"], ix)
	// forward prop the sequence learner
	return state.ForwardLSTM(state.HiddenSizes, x, prev)
}

/*
InitVocab helps initialize this instance's vocab array.
*/
func (state *TrainingState) InitVocab(sents []string, countThreshold int) {
	// go over all characters and keep track of all unique ones seen
	txt := strings.Join(sents, "")

	// count up all characters
	d := make(map[string]int)
	i := 0
	n := len(txt)
	for ; i < n; i++ {
		txti := string(txt[i])
		if _, ok := d[txti]; ok {
			d[txti]++
		} else {
			d[txti] = 1
		}
	}

	// filter by count threshold and create pointers
	state.LetterToIndex = make(map[string]int)
	state.IndexToLetter = make(map[int]string)
	state.Vocab = make([]string, 0)
	// NOTE: start at one because we will have START and END tokens!
	// that is, START token will be index 0 in model letter vectors
	// and END token will be index 0 in the next character softmax
	q := 1
	for ch := range d {
		if d[ch] >= countThreshold {
			// add character to vocab
			state.LetterToIndex[ch] = q
			state.IndexToLetter[q] = ch
			state.Vocab = append(state.Vocab, ch)
			q++
		}
	}

	state.InputSize = len(state.Vocab)
	state.OutputSize = len(state.Vocab)
	state.EpochSize = len(sents)
	fmt.Println("found", len(state.Vocab), " distinct characters: ", state.Vocab)
}

/*
InitModel inits its own Model
*/
func (state *TrainingState) InitModel() {
	// letter embedding vectors
	tempModel := Model{}
	tempModel["Wil"] = RandMat(state.InputSize, state.LetterSize, 0, 0.08)

	lstm := NewLSTMModel(state.LetterSize, state.HiddenSizes, state.OutputSize)
	utilAddToModel(tempModel, lstm)

	state.Model = tempModel
}

func utilAddToModel(modelto Model, modelfrom Model) {
	for k := range modelfrom {
		// copy over the pointer but change the key to use the append
		modelto[k] = modelfrom[k]
	}
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
func (state *TrainingState) ForwardLSTM(hiddenSizes []int, x Mat, prev CellMemory) CellMemory {
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
		h0 := state.G.Mul(state.Model["Wix"+ds], inputVector)
		h1 := state.G.Mul(state.Model["Wih"+ds], hiddenPrev)
		add1 := state.G.Add(h0, h1)
		add2 := state.G.Add(add1, state.Model["bi"+ds])
		inputGate := state.G.Sigmoid(add2)

		// forget gate
		h2 := state.G.Mul(state.Model["Wfx"+ds], inputVector)
		h3 := state.G.Mul(state.Model["Wfh"+ds], hiddenPrev)
		add3 := state.G.Add(h2, h3)
		add4 := state.G.Add(add3, state.Model["bf"+ds])
		forgetGate := state.G.Sigmoid(add4)

		// output gate
		h4 := state.G.Mul(state.Model["Wox"+ds], inputVector)
		h5 := state.G.Mul(state.Model["Woh"+ds], hiddenPrev)
		add45 := state.G.Add(h4, h5)
		add45bods := state.G.Add(add45, state.Model["bo"+ds])
		outputGate := state.G.Sigmoid(add45bods)

		// write operation on cells
		h6 := state.G.Mul(state.Model["Wcx"+ds], inputVector)
		h7 := state.G.Mul(state.Model["Wch"+ds], hiddenPrev)
		add67 := state.G.Add(h6, h7)
		add67bcds := state.G.Add(add67, state.Model["bc"+ds])
		cellWrite := state.G.Tanh(add67bcds)

		// compute new cell activation
		retainCell := state.G.Eltmul(forgetGate, cellPrev) // what do we keep from cell
		writeCell := state.G.Eltmul(inputGate, cellWrite)  // what do we write to cell
		cellD := state.G.Add(retainCell, writeCell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		tahncellD := state.G.Tanh(cellD)
		hiddenD := state.G.Eltmul(outputGate, tahncellD)

		hidden = append(hidden, hiddenD)
		cell = append(cell, cellD)

		// TODO: clear pointer leaks?
	}

	// one decoder to outputs at end
	whdlasthidden := state.G.Mul(state.Model["Whd"], hidden[len(hidden)-1])
	output := state.G.Add(whdlasthidden, state.Model["bd"])

	// return cell memory, hidden representation and output
	return CellMemory{
		Hidden: hidden,
		Cell:   cell,
		Output: output,
	}
}
