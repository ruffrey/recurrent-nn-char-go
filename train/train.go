package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"recurrent/recurrent"
	"sort"
	"strings"
	"time"
)

// model parameters
// list of sizes of hidden layers
var hiddenSizes []int

// size of letter embeddings
var letterSize = 5

// optimization
// L2 regularization strength
const regc = 0.000001
const learning_rate = 0.01

// clip gradients at this value
const clipval = 3

/* */

// prediction params

// how peaky model predictions should be
const sample_softmax_temperature = 1.0

// max length of generated sentences
const max_chars_gen = 100

// various global var inits

var epoch_size int
var input_size int
var output_size int
var letterToIndex map[string]int
var indexToLetter map[int]string
var vocab []string
var data_sentences []string

// should be class because it needs memory for step caches
var solverecurrent recurrent.Solver

// unsure about these former accidental (?) globals
var logprobs recurrent.Mat

// var oldval;

var model recurrent.Model

func readFileContents(filename string) (string, error) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func initVocab(sents []string, count_threshold int) {
	// go over all characters and keep track of all unique ones seen
	txt := strings.Join(sents, "")

	// count up all characters
	d := make(map[string]int)
	i := 0
	n := len(txt)
	for ; i < n; i++ {
		txti := string(txt[i])
		if _, ok := d[txti]; ok {
			d[txti] += 1
		} else {
			d[txti] = 1
		}
	}

	// filter by count threshold and create pointers
	letterToIndex = make(map[string]int)
	indexToLetter = make(map[int]string)
	vocab = make([]string, 0)
	// NOTE: start at one because we will have START and END tokens!
	// that is, START token will be index 0 in model letter vectors
	// and END token will be index 0 in the next character softmax
	q := 1
	for ch := range d {
		if d[ch] >= count_threshold {
			// add character to vocab
			letterToIndex[ch] = q
			indexToLetter[q] = ch
			vocab = append(vocab, ch)
			q++
		}
	}

	// globals written: indexToLetter, letterToIndex, vocab (list), and:
	input_size = len(vocab)
	output_size = len(vocab)
	epoch_size = len(sents)
	fmt.Println("found", len(vocab), " distinct characters: ", vocab)
}

func utilAddToModel(modelto recurrent.Model, modelfrom recurrent.Model) {
	for k := range modelfrom {
		// copy over the pointer but change the key to use the append
		modelto[k] = modelfrom[k]
	}
}

func initModel() recurrent.Model {
	// letter embedding vectors
	tempModel := recurrent.Model{}
	tempModel["Wil"] = recurrent.RandMat(input_size, letterSize, 0, 0.08)

	lstm := recurrent.InitLSTM(letterSize, hiddenSizes, output_size)
	utilAddToModel(tempModel, lstm)

	return tempModel
}

/*
TrainingState is the representation of the training data which gets saved or loaded
to disk between sessions.
*/
type TrainingState struct {
	HiddenSizes   []int
	LetterSize    int
	Model         recurrent.Model
	Solver        recurrent.Solver
	LetterToIndex map[string]int
	IndexToLetter map[int]string
	Vocab         []string
}

func saveModel() {
	out := TrainingState{
		HiddenSizes: hiddenSizes,
		LetterSize:  letterSize,
	}

	out.Model = recurrent.Model{}
	for k := range model {
		out.Model[k] = model[k]
	}

	out.Solver = solverecurrent
	out.LetterToIndex = letterToIndex
	out.IndexToLetter = indexToLetter
	out.Vocab = vocab

	// TODO: finish json stuff and write it somewhere
}

func modelFromDisk() {
	// TODO; read from disk into TrainingState{}
}

func loadModel(j TrainingState) {
	hiddenSizes = j.HiddenSizes
	letterSize = j.LetterSize
	model = j.Model

	solverecurrent = j.Solver

	letterToIndex = j.LetterToIndex
	indexToLetter = j.IndexToLetter
	vocab = j.Vocab

	// reinit these
	perplexityList = make([]float64, 0)
	tick_iter = 0
}

func forwardIndex(G *recurrent.Graph, mod recurrent.Model, ix int, prev recurrent.CellMemory) recurrent.CellMemory {
	modwil := mod["Wil"]
	x := G.RowPluck(&modwil, ix)
	// forward prop the sequence learner
	out_struct := recurrent.ForwardLSTM(G, mod, hiddenSizes, x, prev)
	return out_struct
}

func predictSentence(mod recurrent.Model, samplei bool, temperature float64) (s string) {
	G := recurrent.NewGraph(false)
	prev := recurrent.CellMemory{}

	for {
		// RNN tick
		var ix int
		if len(s) == 0 {
			ix = 0
		} else {
			ix = letterToIndex[string(s[len(s)-1])]
		}
		lh := forwardIndex(&G, mod, ix, prev)
		prev = lh

		// sample predicted letter
		logprobs = lh.Output
		if temperature != 1.0 && samplei {
			// scale log probabilities by temperature and renormalize
			// if temperature is high, logprobs will go towards zero
			// and the softmax outputs will be more diffuse. if temperature is
			// very low, the softmax outputs will be more peaky
			q := 0
			nq := len(logprobs.W)
			for ; q < nq; q++ {
				logprobs.W[q] /= temperature
			}
		}

		probs := recurrent.Softmax(&logprobs)

		if samplei {
			ix = recurrent.Samplei(probs.W)
		} else {
			ix = recurrent.Maxi(probs.W)
		}

		if ix == 0 {
			break // END token predicted, break out
		}
		if len(s) > max_chars_gen {
			break // something is wrong
		}

		letter := indexToLetter[ix]
		s += letter
	}
	return s
}

/*
Cost represents the result of running the cost function.
*/
type Cost struct {
	G    recurrent.Graph
	ppl  float64
	cost float64
}

/**
 * costfun takes a model and a sentence and
 * calculates the loss. Also returns the Graph
 * object which can be used to do backprop
 */
func costfun(mod recurrent.Model, sent string) Cost {
	n := len(sent)
	G := recurrent.NewGraph(true)
	log2ppl := 0.0
	cost := 0.0
	prev := recurrent.CellMemory{}
	for i := -1; i < n; i++ {
		// start and end tokens are zeros

		// first step: start with START token
		var ix_source int
		if i == -1 {
			ix_source = 0
		} else {
			ix_source = letterToIndex[string(sent[i])]
		}
		// last step: end with END token
		var ix_target int
		if i == n-1 {
			ix_target = 0
		} else {
			ix_target = letterToIndex[string(sent[i+1])]
		}

		lh := forwardIndex(&G, mod, ix_source, prev)
		prev = lh

		// set gradients into logprobabilities
		logprobs = lh.Output                  // interpret output as logprobs
		probs := recurrent.Softmax(&logprobs) // compute the softmax probabilities

		log2ppl += -math.Log2(probs.W[ix_target]) // accumulate base 2 log prob and do smoothing
		cost += -math.Log2(probs.W[ix_target])

		// write gradients into log probabilities
		logprobs.DW = probs.W
		logprobs.DW[ix_target] -= 1
	}
	ppl := math.Pow(2, log2ppl/float64(n-1))
	return Cost{
		G,
		ppl,
		cost,
	}
}

func median(values []float64) float64 {
	sort.Float64s(values)
	lenValues := len(values)
	half := int(math.Floor(float64(lenValues / 2)))
	if math.Remainder(float64(lenValues), 2) != 0.0 {
		return values[half]
	}
	return (values[half-1] + values[half]) / 2.0
}

var perplexityList []float64
var tick_iter int

func tick() {
	defer tick() // loop it again
	// sample sentence fromd data
	sentix := recurrent.Randi(0, len(data_sentences))
	sent := data_sentences[sentix]

	t0 := time.Now().UnixNano() / 1000 // log start timestamp

	// evaluate cost func on a sentence
	cost_struct := costfun(model, sent)

	// use built up graph to compute backprop (set .dw fields in mats)
	cost_struct.G.Backward()
	// perform param update
	solverecurrent.Step(model, learning_rate, regc, clipval)

	t1 := time.Now().UnixNano() / 1000
	tick_time := t1 - t0

	perplexityList = append(perplexityList, cost_struct.ppl) // keep track of perplexity

	// evaluate now and then
	tick_iter++

	if math.Remainder(float64(tick_iter), 50) == 0 {
		pred := ""
		fmt.Println("---------------------")
		// draw samples
		for q := 0; q < 5; q++ {
			pred = predictSentence(model, true, sample_softmax_temperature)
			fmt.Println("prediction", pred)
		}

		epoch := (tick_iter / epoch_size)
		perplexity := cost_struct.ppl
		medianPerplexity := median(perplexityList)
		perplexityList = make([]float64, 0)

		fmt.Println("epoch=", epoch)
		fmt.Println("epoch_size", epoch_size)
		fmt.Println("perplexity", perplexity)
		fmt.Println("ticktime", tick_time, "ms")
		fmt.Println("medianPerplexity", medianPerplexity)
	}
}

// old gradCheck was here.

func main() {
	// Define the hidden layers
	hiddenSizes = make([]int, 2)
	hiddenSizes[0] = 20
	hiddenSizes[1] = 20

	epoch_size = -1
	input_size = -1
	output_size = -1

	perplexityList = make([]float64, 0)
	data_sentences = make([]string, 0)

	solverecurrent = recurrent.NewSolver() // reinit solver
	tick_iter = 0

	// process the input, filter out blanks
	input, err := readFileContents("input.txt")
	if err != nil {
		log.Fatal("Failed reading file input", err)
	}
	data_sentences_raw := strings.Split(input, "\n")

	for i := 0; i < len(data_sentences_raw); i++ {
		sent := data_sentences_raw[i] // .trim();
		if len(sent) > 0 {
			data_sentences = append(data_sentences, sent)
		}
	}

	initVocab(data_sentences, 1) // takes count threshold for characters
	model = initModel()

	// go!
	tick()
}
