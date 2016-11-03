package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	_ "net/http/pprof"
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
const learningRate = 0.01

// clip gradients at this value
const clipval = 3

/* */

// prediction params

// how peaky model predictions should be
const sampleSoftmaxTemperature = 1.0

// max length of generated sentences
const maxCharsGenerate = 100

// various global var inits

// should be class because it needs memory for step caches
var solverecurrent recurrent.Solver

// unsure about these former accidental (?) globals
var logprobs recurrent.Mat

func readFileContents(filename string) (string, error) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func predictSentence(state *recurrent.TrainingState, samplei bool, temperature float64) (s string) {
	G := recurrent.NewGraph(false)
	var prev *recurrent.CellMemory
	initial := recurrent.CellMemory{}
	prev = &initial

	for {
		// RNN tick
		var ix int
		if len(s) == 0 {
			ix = 0
		} else {
			ix = state.LetterToIndex[string(s[len(s)-1])]
		}
		lh := state.ForwardIndex(&G, ix, prev)
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
		if len(s) > maxCharsGenerate {
			break // something is wrong
		}

		letter := state.IndexToLetter[ix]
		s += letter
	}

	prev = nil

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
 *
 * this LEAKS
 */
func costfun(state *recurrent.TrainingState, sent string) Cost {
	n := len(sent)
	G := recurrent.NewGraph(true)
	var log2ppl float64
	var cost float64

	var ixSource int
	var ixTarget int
	var lh *recurrent.CellMemory
	var prev *recurrent.CellMemory
	initial := recurrent.CellMemory{}
	prev = &initial
	var probswixtarget float64
	var probs recurrent.Mat
	for i := -1; i < n; i++ {
		// start and end tokens are zeros

		// first step: start with START token
		if i == -1 {
			ixSource = 0
		} else {
			ixSource = state.LetterToIndex[string(sent[i])]
		}
		// last step: end with END token
		if i == n-1 {
			ixTarget = 0
		} else {
			ixTarget = state.LetterToIndex[string(sent[i+1])]
		}

		lh = state.ForwardIndex(&G, ixSource, prev)
		prev = lh

		// set gradients into logprobabilities
		logprobs = lh.Output                 // interpret output as logprobs
		probs = recurrent.Softmax(&logprobs) // compute the softmax probabilities

		probswixtarget = probs.W[ixTarget]
		// fmt.Println(i, n, ixTarget, probswixtarget, log2ppl)

		// the following line has a huge leak, apparently.
		log2ppl += -math.Log2(probswixtarget) // accumulate base 2 log prob and do smoothing
		cost += -math.Log2(probswixtarget)

		// write gradients into log probabilities
		logprobs.DW = probs.W
		logprobs.DW[ixTarget]--
	}

	ppl := math.Pow(2, log2ppl/float64(n-1))

	prev = nil // avoid leaks
	lh = nil   // avoid leaks

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

func tick(state *recurrent.TrainingState) {
	defer func() {
		tick(state)
	}()
	// sample sentence fromd data
	sentix := recurrent.Randi(0, len(state.DataSentences))
	sent := state.DataSentences[sentix]

	t0 := time.Now().UnixNano() / 1000000 // log start timestamp ms

	// evaluate cost func on a sentence
	costStruct := costfun(state, sent)

	// use built up graph to compute backprop (set .dw fields in mats)
	costStruct.G.Backward()
	// perform param update
	solverecurrent.Step(state.Model, learningRate, regc, clipval)

	t1 := time.Now().UnixNano() / 1000000 // ms
	tickTime := t1 - t0

	state.PerplexityList = append(state.PerplexityList, costStruct.ppl) // keep track of perplexity

	// evaluate now and then
	state.TickIterator++

	if math.Remainder(float64(state.TickIterator), 100) == 0 {
		pred := ""
		fmt.Println("---------------------")
		// draw samples
		for q := 0; q < 5; q++ {
			pred = predictSentence(state, true, sampleSoftmaxTemperature)
			fmt.Println("prediction", pred)
		}

		epoch := (state.TickIterator / state.EpochSize)
		perplexity := costStruct.ppl
		medianPerplexity := median(state.PerplexityList)
		state.PerplexityList = make([]float64, 0)

		fmt.Println("epoch=", epoch)
		fmt.Println("EpochSize", state.EpochSize)
		fmt.Println("perplexity", perplexity)
		fmt.Println("ticktime", tickTime, "ms")
		fmt.Println("medianPerplexity", medianPerplexity)
		// m := runtime.MemStats{}
		// runtime.ReadMemStats(&m)
		// fmt.Println(m)
	}
}

// old gradCheck was here.

func main() {
	// Define the hidden layers
	hiddenSizes = make([]int, 2)
	hiddenSizes[0] = 20
	hiddenSizes[1] = 20

	// this is where the training state is held in memory, not in global scope
	// most importantly, to prevent leaks.
	// (could also fetch from disk)
	state := recurrent.TrainingState{
		HiddenSizes: hiddenSizes,
		EpochSize:   -1,
		InputSize:   -1,
		OutputSize:  -1,
	}

	state.PerplexityList = make([]float64, 0)
	state.DataSentences = make([]string, 0)

	solverecurrent = recurrent.NewSolver() // reinit solver
	state.TickIterator = 0

	// process the input, filter out blanks
	input, err := readFileContents("input.txt")
	if err != nil {
		log.Fatal("Failed reading file input", err)
	}
	dataSentencesRaw := strings.Split(input, "\n")

	for i := 0; i < len(dataSentencesRaw); i++ {
		sent := dataSentencesRaw[i] // .trim();
		if len(sent) > 0 {
			state.DataSentences = append(state.DataSentences, sent)
		}
	}

	state.InitVocab(state.DataSentences, 1) // takes count threshold for characters
	state.InitModel()

	// checking memory leaks
	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	tick(&state)
}
