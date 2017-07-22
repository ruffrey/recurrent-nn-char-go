package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"github.com/ruffrey/recurrent-nn-char-go/mat32"
)

/*
TrainingState is the representation of the training data which gets saved or loaded
to disk between sessions.
*/
type TrainingState struct {
	mat32.Graph `json:"-"`
	HiddenSizes    []int
	Model          Model
	Solver         Solver
	LetterToIndex  map[string]int
	IndexToLetter  map[int]string
	Vocab          []string
	PerplexityList []float64 `json:"-"`

	// the following do not need to be persisted between training sessions
	EpochSize     int
	lastSaveEpoch int
	InputSize     int
	OutputSize    int
	DataSentences []string `json:"-"`
	TickIterator  int `json:"-"`
	hiddenPrevs   []*mat32.Mat
	cellPrevs     []*mat32.Mat
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
		if len(ch) > 1 {
			fmt.Println("Dropping char due to size-bounds issue:", ch)
			continue
		}
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
	fmt.Println(len(state.Vocab), "distinct characters:\n ", state.Vocab)
}

/*
InitModel inits its own Model
*/
func (state *TrainingState) InitModel() {
	// letter embedding vectors
	tempModel := Model{}
	tempModel["Wil"] = mat32.RandMat(state.InputSize, sequenceLength, 0, 0.08)

	lstm := NewLSTMModel(sequenceLength, state.HiddenSizes, state.OutputSize)
	utilAddToModel(tempModel, lstm)

	state.Model = tempModel
}

func utilAddToModel(modelto Model, modelfrom Model) {
	for k := range modelfrom {
		// copy over the pointer but change the key to use the append
		modelto[k] = modelfrom[k]
	}
}

// worker processes intensive func presumably async then sends back the value
func worker(fn func() *mat32.Mat, out chan *mat32.Mat) {
	out <- fn()
}

var inputChan chan *mat32.Mat = make(chan *mat32.Mat)
var forgetChan chan *mat32.Mat = make(chan *mat32.Mat)
var outputChan chan *mat32.Mat = make(chan *mat32.Mat)
var writeChan chan *mat32.Mat = make(chan *mat32.Mat)

/*
ForwardLSTM does forward propagation for a single tick of LSTM. Will be called in a loop.

x is 1D column vector with observation
prev is a struct containing hidden and cell from previous iteration
*/
func (state *TrainingState) ForwardLSTM(hiddenSizes []int, x *mat32.Mat, prev *CellMemory) *CellMemory {

	// initialize when not yet initialized. we know there will always be hidden layers.
	if len(prev.Hidden) == 0 {
		// reset these
		state.hiddenPrevs = make([]*mat32.Mat, len(hiddenSizes))
		state.cellPrevs = make([]*mat32.Mat, len(hiddenSizes))
		for s := 0; s < len(hiddenSizes); s++ {
			state.hiddenPrevs[s] = mat32.NewMat(hiddenSizes[s], 1)
			state.cellPrevs[s] = mat32.NewMat(hiddenSizes[s], 1)
		}
	} else {
		state.hiddenPrevs = prev.Hidden
		state.cellPrevs = prev.Cell
	}

	var hidden []*mat32.Mat
	var cell []*mat32.Mat
	var inputVector *mat32.Mat
	var hiddenPrev *mat32.Mat
	var cellPrev *mat32.Mat

	// Parallizing this hot path is tricky because it
	// relies on the previous array value.
	for d := 0; d < len(hiddenSizes); d++ {
		var inputGate *mat32.Mat
		var forgetGate *mat32.Mat
		var outputGate *mat32.Mat
		var cellWrite *mat32.Mat

		if d == 0 {
			inputVector = x
		} else {
			inputVector = hidden[d-1]
		}
		hiddenPrev = state.hiddenPrevs[d]
		cellPrev = state.cellPrevs[d]

		// ds is the index but as a string
		ds := strconv.Itoa(d)

		// send 4 jobs to the worker, when 4 come back, done.

		// input gate
		go worker(func() *mat32.Mat {
			h0 := state.Mul(state.Model["Wix"+ds], inputVector)
			h1 := state.Mul(state.Model["Wih"+ds], hiddenPrev)
			add1 := state.Add(h0, h1)
			add2 := state.Add(add1, state.Model["bi"+ds])
			return state.Sigmoid(add2)
		}, inputChan)

		// forget gate
		go worker(func() *mat32.Mat {
			h2 := state.Mul(state.Model["Wfx"+ds], inputVector)
			h3 := state.Mul(state.Model["Wfh"+ds], hiddenPrev)
			add3 := state.Add(h2, h3)
			add4 := state.Add(add3, state.Model["bf"+ds])
			return state.Sigmoid(add4)
		}, forgetChan)

		// output gate
		go worker(func() *mat32.Mat {
			h4 := state.Mul(state.Model["Wox"+ds], inputVector)
			h5 := state.Mul(state.Model["Woh"+ds], hiddenPrev)
			add45 := state.Add(h4, h5)
			add45bods := state.Add(add45, state.Model["bo"+ds])
			return state.Sigmoid(add45bods)
		}, outputChan)

		// write operation on cells
		go worker(func() *mat32.Mat {
			h6 := state.Mul(state.Model["Wcx"+ds], inputVector)
			h7 := state.Mul(state.Model["Wch"+ds], hiddenPrev)
			add67 := state.Add(h6, h7)
			add67bcds := state.Add(add67, state.Model["bc"+ds])
			return state.Tanh(add67bcds)
		}, writeChan)

		gatesComplete := 0
		for {
			select {
			case inputGate = <-inputChan:
				gatesComplete++
				break
			case forgetGate = <-forgetChan:
				gatesComplete++
				break
			case outputGate = <-outputChan:
				gatesComplete++
				break
			case cellWrite = <-writeChan:
				gatesComplete++
				break
			}
			if gatesComplete >= 4 {
				break
			}
		}

		// compute new cell activation
		retainCell := state.Eltmul(forgetGate, cellPrev) // what do we keep from cell
		writeCell := state.Eltmul(inputGate, cellWrite)  // what do we write to cell
		cellD := state.Add(retainCell, writeCell)        // new cell contents

		// compute hidden state as gated, saturated cell activations
		tahncellD := state.Tanh(cellD)
		hiddenD := state.Eltmul(outputGate, tahncellD)

		hidden = append(hidden, hiddenD)
		cell = append(cell, cellD)
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
StepSolver does a param update on the model, increasing or decreasting the weights,
and clipping the derivative first if necessary.

Should model be a poiner? unable to loop over it if not. So we return it and then copy it back
onto the existing model.

stepSize is the learningRate
regc is regularization
*/
func (state *TrainingState) StepSolver(solver *Solver, stepSize float32, regc float32, clipval float32) {
	// perform parameter update
	var wg sync.WaitGroup

	// do this first loop to make sure everything exists
	// before doing the second loop in parallel
	for key, mod := range state.Model {
		_, hasKey := solver.StepCache[key]
		if !hasKey {
			solver.StepCache[key] = mat32.NewMat(mod.RowCount, mod.ColumnCount)
		}
	}

	for key, mod := range state.Model {
		wg.Add(1)
		// Pass in the current iteration stuff to concurrently process
		// without referencing the wrong value.
		// Having this lower down in the for loop nest did not seem to
		// speed things up, due to increased overhead of tracking
		// goroutines by the runtime.
		go (func(k string, m *mat32.Mat) {
			i := 0
			n := len(m.W)
			for ; i < n; i++ {
				// rmsprop adaptive learning rate
				mdwi := m.DW[i]
				solver.StepCache[k].W[i] = solver.StepCache[k].W[i]*solver.DecayRate + (1.0-solver.DecayRate)*mdwi*mdwi

				// gradient clip
				if mdwi > clipval {
					mdwi = clipval
				}
				if mdwi < -clipval {
					mdwi = -clipval
				}

				// update (and regularize)
				kwi := solver.StepCache[k].W[i]
				sqrtSumEPS := float32(math.Sqrt(float64(kwi + solver.SmoothEPS)))
				m.W[i] += -stepSize*mdwi/sqrtSumEPS - regc*m.W[i]
				m.DW[i] = 0 // reset gradients for next iteration
			}
			wg.Done()
		})(key, mod)
	}
	wg.Wait()
}

/*
PredictSentence creates a prediction based on the current training state. similar to cost function.
*/
func (state *TrainingState) PredictSentence(samplei bool, temperature float32, maxCharsGenerate int, seedString string) (s string) {
	state.NeedsBackprop = false // temporary but do not lose functions
	var prev *CellMemory
	initial := &CellMemory{}
	prev = initial
	var lh *CellMemory
	seedIndex := 0
	seed := strings.Split(seedString, "")

	for {
		// RNN tick
		var ixSource int
		if len(s) == 0 {
			ixSource = 0
		} else if seedIndex < len(seed) {
			prevIndex := len(s) - 1
			prevLetter := seed[prevIndex]
			ixSource = state.LetterToIndex[prevLetter]
			seedIndex++
		} else {
			letters := strings.Split(s, "")
			prevIndex := len(s) - 1
			prevLetter := letters[prevIndex]
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
	if seedString != "" {
		s = strings.Replace(s, seedString, "", 1)
	}
	return s
}
