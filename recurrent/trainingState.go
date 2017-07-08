package recurrent

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
)

/*
TrainingState is the representation of the training data which gets saved or loaded
to disk between sessions.
*/
type TrainingState struct {
	Graph
	HiddenSizes []int
	// LetterSize is the size of letter embeddings
	LetterSize     int
	Model          Model
	Solver         Solver
	LetterToIndex  map[string]int
	IndexToLetter  map[int]string
	Vocab          []string
	PerplexityList []float64

	// the following do not need to be persisted between training sessions
	EpochSize     int
	InputSize     int
	OutputSize    int
	DataSentences []string `json:"-"`
	TickIterator  int
	hiddenPrevs   []*Mat
	cellPrevs     []*Mat
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
	fmt.Println("Found", len(state.Vocab), " distinct characters: ", state.Vocab)
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
ForwardLSTM does forward propagation for a single tick of LSTM. Will be called in a loop.

x is 1D column vector with observation
prev is a struct containing hidden and cell from previous iteration
*/
func (state *TrainingState) ForwardLSTM(hiddenSizes []int, x *Mat, prev *CellMemory) *CellMemory {

	// initialize when not yet initialized. we know there will always be hidden layers.
	if len(prev.Hidden) == 0 {
		// reset these
		state.hiddenPrevs = make([]*Mat, len(hiddenSizes))
		state.cellPrevs = make([]*Mat, len(hiddenSizes))
		for s := 0; s < len(hiddenSizes); s++ {
			state.hiddenPrevs[s] = NewMat(hiddenSizes[s], 1)
			state.cellPrevs[s] = NewMat(hiddenSizes[s], 1)
		}
	} else {
		state.hiddenPrevs = prev.Hidden
		state.cellPrevs = prev.Cell
	}

	var hidden []*Mat
	var cell []*Mat
	var inputVector *Mat
	var hiddenPrev *Mat
	var cellPrev *Mat

	for d := 0; d < len(hiddenSizes); d++ {

		if d == 0 {
			inputVector = x
		} else {
			inputVector = hidden[d-1]
		}
		hiddenPrev = state.hiddenPrevs[d]
		cellPrev = state.cellPrevs[d]

		// ds is the index but as a string
		ds := strconv.Itoa(d)

		// input gate
		h0 := state.Mul(state.Model["Wix"+ds], inputVector)
		h1 := state.Mul(state.Model["Wih"+ds], hiddenPrev)
		add1 := state.Add(h0, h1)
		add2 := state.Add(add1, state.Model["bi"+ds])
		inputGate := state.Sigmoid(add2)

		// forget gate
		h2 := state.Mul(state.Model["Wfx"+ds], inputVector)
		h3 := state.Mul(state.Model["Wfh"+ds], hiddenPrev)
		add3 := state.Add(h2, h3)
		add4 := state.Add(add3, state.Model["bf"+ds])
		forgetGate := state.Sigmoid(add4)

		// output gate
		h4 := state.Mul(state.Model["Wox"+ds], inputVector)
		h5 := state.Mul(state.Model["Woh"+ds], hiddenPrev)
		add45 := state.Add(h4, h5)
		add45bods := state.Add(add45, state.Model["bo"+ds])
		outputGate := state.Sigmoid(add45bods)

		// write operation on cells
		h6 := state.Mul(state.Model["Wcx"+ds], inputVector)
		h7 := state.Mul(state.Model["Wch"+ds], hiddenPrev)
		add67 := state.Add(h6, h7)
		add67bcds := state.Add(add67, state.Model["bc"+ds])
		cellWrite := state.Tanh(add67bcds)

		// compute new cell activation
		retainCell := state.Eltmul(forgetGate, cellPrev) // what do we keep from cell
		writeCell := state.Eltmul(inputGate, cellWrite)  // what do we write to cell
		cellD := state.Add(retainCell, writeCell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		tahncellD := state.Tanh(cellD)
		hiddenD := state.Eltmul(outputGate, tahncellD)

		hidden = append(hidden, hiddenD)
		cell = append(cell, cellD)

		// TODO: clear pointer leaks?
	}

	// one decoder to outputs at end
	whdlasthidden := state.Mul(state.Model["Whd"], hidden[len(hidden)-1])
	output := state.Add(whdlasthidden, state.Model["bd"])

	// return cell memory, hidden representation and output
	return &CellMemory{
		Hidden: hidden,
		Cell:   cell,
		Output: output,
	}
}

/*
StepSolver does a step.
Should model be a poiner? unable to loop over it if not. So we return it and then copy it back
onto the existing model.
*/
func (state *TrainingState) StepSolver(solver *Solver, stepSize float64, regc float64, clipval float64) SolverStats {
	// perform parameter update
	solverStats := SolverStats{}
	numClipped := 0.0
	numTot := 0.0

	var mux sync.Mutex
	var wg sync.WaitGroup

	for key, mod := range state.Model {
		solver.mux.Lock()
		_, hasKey := solver.StepCache[key]
		if !hasKey {
			solver.StepCache[key] = NewMat(mod.RowCount, mod.ColumnCount)
		}
		solver.mux.Unlock()

		f := 0
		n := len(mod.W)
		for ; f < n; f++ {
			wg.Add(1)
			// pass in the current iteration stuff to concurrently process
			// without referencing the wrong value
			go (func(k string, m *Mat, i int) {
				// rmsprop adaptive learning rate
				mdwi := m.DW[i]
				solver.mux.Lock()
				solver.StepCache[k].W[i] = solver.StepCache[k].W[i]*solver.DecayRate + (1.0-solver.DecayRate)*mdwi*mdwi
				solver.mux.Unlock()

				// gradient clip
				if mdwi > clipval {
					mdwi = clipval
					mux.Lock()
					numClipped++
					mux.Unlock()
				}
				if mdwi < -clipval {
					mdwi = -clipval
					mux.Lock()
					numClipped++
					mux.Unlock()
				}
				mux.Lock()
				numTot++
				mux.Unlock()

				// update (and regularize)
				solver.mux.Lock()
				kwi := solver.StepCache[k].W[i]
				solver.mux.Unlock()
				m.W[i] += -stepSize*mdwi/math.Sqrt(kwi+solver.SmoothEPS) - regc*m.W[i]
				m.DW[i] = 0 // reset gradients for next iteration

				wg.Done()
			})(key, mod, f)
		}
	}
	wg.Wait()

	solverStats["ratio_clipped"] = numClipped * 1.0 / numTot
	return solverStats
}

/*
PredictSentence creates a prediction based on the current training state. similar to cost function.
*/
func (state *TrainingState) PredictSentence(samplei bool, temperature float64, maxCharsGenerate int) (s string) {
	state.NeedsBackprop = false // temporary but do not lose functions
	var prev *CellMemory
	initial := &CellMemory{}
	prev = initial
	var lh *CellMemory

	for {
		// RNN tick
		var ixSource int
		if len(s) == 0 {
			ixSource = 0
		} else {
			letters := strings.Split(s, "")
			prevLetter := letters[len(s)-1]
			ixSource = state.LetterToIndex[prevLetter]
		}

		lh = state.ForwardLSTM(
			state.HiddenSizes,
			state.RowPluck(state.Model["Wil"], ixSource),
			prev,
		)
		prev = lh

		// sample predicted letter
		logrithmicProbabilities := lh.Output
		if temperature != 1.0 && samplei {
			// scale log probabilities by temperature and renormalize
			// if temperature is high, logrithmicProbabilities will go towards zero
			// and the softmax outputs will be more diffuse. if temperature is
			// very low, the softmax outputs will be more peaky
			q := 0
			nq := len(logrithmicProbabilities.W)
			for ; q < nq; q++ {
				logrithmicProbabilities.W[q] /= temperature
			}
		}

		probs := Softmax(logrithmicProbabilities)

		if samplei {
			ixSource = SampleArgmaxI(probs.W)
		} else {
			ixSource = ArgmaxI(probs.W)
		}

		if ixSource == 0 || ixSource == len(probs.W) {
			break // start or end token predicted, break out
		}
		if len(s) > maxCharsGenerate {
			break // something is wrong
		}

		letter := state.IndexToLetter[ixSource]
		s += letter
	}

	state.NeedsBackprop = true // temporary but do not lose functions
	return s
}
