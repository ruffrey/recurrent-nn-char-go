package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	// "net/http"
	// _ "net/http/pprof"
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

func readFileContents(filename string) (string, error) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// create a prediction based on the current training state
func predictSentence(state *recurrent.TrainingState, samplei bool, temperature float64) (s string) {
	state.G = recurrent.NewGraph(false)
	prev := recurrent.CellMemory{}
	var lh recurrent.CellMemory

	for len(s) < maxCharsGenerate {
		// RNN tick
		var ix int
		if len(s) == 0 {
			ix = 0
		} else {
			ix = state.LetterToIndex[string(s[len(s)-1])]
		}

		lh = state.ForwardIndex(ix, prev)
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

		probs := recurrent.Softmax(&logrithmicProbabilities)

		if samplei {
			ix = recurrent.SampleArgmaxI(probs.W)
		} else {
			ix = recurrent.ArgmaxI(probs.W)
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

	state = nil

	return s
}

/*
Cost represents the result of running the cost function.
*/
type Cost struct {
	G    *recurrent.Graph
	ppl  float64
	cost float64
}

/**
 * costfun takes a model and a sentence and
 * calculates the loss. Also returns the Graph
 * object which can be used to do backprop
 */
func costfun(state *recurrent.TrainingState, sent string) Cost {
	n := len(sent)
	G := recurrent.NewGraph(true)
	var log2ppl float64
	var cost float64

	var ixSource int
	var ixTarget int
	var lh recurrent.CellMemory
	prev := recurrent.CellMemory{}
	var probswixtarget float64
	var probs recurrent.Mat

	// loop through each letter of the selected sentence
	for i := -1; i < n; i++ {
		// start and end tokens are zeros
		// fmt.Println("i=", i, ", n=", n)
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
		// TODO: this is never changing the value
		// fmt.Println(prev)
		lh = state.ForwardIndex(ixSource, prev)
		// fmt.Println(lh)
		prev = lh

		// set gradients into logprobabilities
		logrithmicProbabilities := lh.Output                // interpret output as logrithmicProbabilities
		probs = recurrent.Softmax(&logrithmicProbabilities) // compute the softmax probabilities

		probswixtarget = probs.W[ixTarget]

		// the following line has a huge leak, apparently.
		log2ppl += -math.Log2(probswixtarget) // accumulate base 2 log prob and do smoothing
		cost += -math.Log2(probswixtarget)

		// write gradients into log probabilities
		// TODO: this is not woring
		logrithmicProbabilities.DW = probs.W
		logrithmicProbabilities.DW[ixTarget]--
	}

	ppl := math.Pow(2, log2ppl/float64(n-1))

	return Cost{
		G:    &G,
		ppl:  ppl,
		cost: cost,
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
		if state.TickIterator < 2 {
			tick(state)
		}
	}()
	// sample sentence from data
	sentix := recurrent.Randi(0, len(state.DataSentences))
	sent := state.DataSentences[sentix]

	t0 := time.Now().UnixNano() / 1000000 // log start timestamp ms

	// evaluate cost func on a sentence
	// TODO: should be different before and after
	costStruct := costfun(state, sent)
	// use built up graph to compute backprop (set .DW fields in mats)
	fmt.Println(state.TickIterator, " -- BEFORE Backward:\n  ", state.Model)
	costStruct.G.Backward()
	fmt.Println(state.TickIterator, " -- AFTER Backward:\n  ", state.Model)

	// perform param update
	var solverStats recurrent.SolverStats
	state.Model, solverStats = solverecurrent.Step(state.Model, learningRate, regc, clipval)

	t1 := time.Now().UnixNano() / 1000000 // ms
	tickTime := t1 - t0

	// keep track of perplexity between printing progress
	state.PerplexityList = append(state.PerplexityList, costStruct.ppl)

	// evaluate now and then
	state.TickIterator++

	if math.Remainder(float64(state.TickIterator), 250) == 0 {
		pred := ""
		fmt.Println("---------------------")
		// draw samples
		for q := 0; q < 5; q++ {
			pred = predictSentence(state, true, sampleSoftmaxTemperature)
			fmt.Println("prediction", pred)
		}

		epoch := (float32(state.TickIterator) / float32(state.EpochSize))
		perplexity := costStruct.ppl
		medianPerplexity := median(state.PerplexityList)
		state.PerplexityList = make([]float64, 0)

		fmt.Println("epoch=", epoch)
		fmt.Println("EpochSize", state.EpochSize)
		fmt.Println("perplexity", perplexity)
		fmt.Println("ticktime", tickTime, "ms")
		fmt.Println("medianPerplexity", medianPerplexity)
		fmt.Println("solverStats", solverStats)
	}

}

// old gradCheck was here.

func main() {
	// Define the hidden layers
	hiddenSizes = make([]int, 2)
	hiddenSizes[0] = 2
	hiddenSizes[1] = 2

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

	solverecurrent = recurrent.NewSolver() // reinit solver
	state.TickIterator = 0

	// process the input, filter out blanks
	input, err := readFileContents("input.txt")
	if err != nil {
		log.Fatal("Failed reading file input", err)
	}

	state.DataSentences = strings.Split(input, "\n")
	state.InitVocab(state.DataSentences, 1) // takes count threshold for characters
	state.InitModel()
	// checking memory leaks
	// go func() {
	// 	log.Println(http.ListenAndServe("localhost:6060", nil))
	// }()

	tick(&state)
}
