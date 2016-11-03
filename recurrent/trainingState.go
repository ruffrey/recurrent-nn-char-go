package recurrent

import (
	"fmt"
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
func (state *TrainingState) ForwardIndex(G *Graph, ix int, prev *CellMemory) *CellMemory {
	modwil := state.Model["Wil"]
	x := G.RowPluck(&modwil, ix)
	// forward prop the sequence learner
	outputMemory := ForwardLSTM(G, state.Model, state.HiddenSizes, x, prev)

	prev = nil // avoid leaks
	G = nil    // avoid leaks
	prev = nil // avoid leaks

	return &outputMemory
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

	fmt.Println(state.LetterSize, state.HiddenSizes, state.OutputSize)
	lstm := InitLSTM(state.LetterSize, state.HiddenSizes, state.OutputSize)
	utilAddToModel(tempModel, lstm)

	state.Model = tempModel
}

func utilAddToModel(modelto Model, modelfrom Model) {
	for k := range modelfrom {
		// copy over the pointer but change the key to use the append
		modelto[k] = modelfrom[k]
	}
}
