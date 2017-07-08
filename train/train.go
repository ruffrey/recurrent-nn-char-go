package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"github.com/ruffrey/recurrent-nn-char-go/recurrent"
	"sort"
	"strings"
	"time"
	//"github.com/pkg/profile"
	"os"
	"encoding/json"
)

// model parameters
// list of sizes of hidden layers
var hiddenSizes []int

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

func writeFileContents(filename string, contents []byte) (error) {
	return ioutil.WriteFile(filename, contents, os.ModePerm)
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
	// sample sentence from data
	sentix := recurrent.Randi(0, len(state.DataSentences))
	sent := state.DataSentences[sentix]

	t0 := time.Now().UnixNano() / 1000000 // log start timestamp ms

	// evaluate cost func on a sentence
	// TODO: should be different before and after
	// fmt.Println(state.TickIterator, " -- BEFORE CostFunction:\n  ", state.Model)
	costStruct := state.CostFunction(sent)
	// fmt.Println(state.TickIterator, " -- AFTER CostFunction:\n  ", state.Model)
	// use built up graph to compute backprop (set .DW fields in mats)
	state.Backward()

	// perform param update
	solverStats := state.StepSolver(solverecurrent, learningRate, regc, clipval)

	// keep track of perplexity between printing progress
	state.PerplexityList = append(state.PerplexityList, costStruct.Ppl)

	// evaluate now and then
	state.TickIterator++

	epoch := float64(state.TickIterator) / float64(state.EpochSize)

	if math.Remainder(float64(state.TickIterator), 300) == 0 {
		t1 := time.Now().UnixNano() / 1000000 // ms
		tickTime := t1 - t0

		pred := ""
		fmt.Println("---------------------")
		// draw samples
		for q := 0; q < 5; q++ {
			pred = state.PredictSentence(true, sampleSoftmaxTemperature, maxCharsGenerate)
			fmt.Println(pred)
		}
		fmt.Println("---------------------")
		medianPerplexity := median(state.PerplexityList)
		state.PerplexityList = make([]float64, 0)

		fmt.Println("epoch=", epoch)
		fmt.Println("ticktime", tickTime, "ms")
		fmt.Println("medianPerplexity", medianPerplexity)
		fmt.Println("solverStats", solverStats)
	}

	if math.Remainder(epoch, 1) < 0.01 {
		fname := fmt.Sprintf("save-%f.json", epoch)
		fmt.Println("Saving progress", fname)
		jsonState, err := json.Marshal(state)
		if err != nil {
			fmt.Println("stringify err", err)
		} else {
			err = writeFileContents(fname, jsonState)
			if err != nil {
				fmt.Println("Save error", err, fname)
			}
		}
	}

	tick(state)
}

// old gradCheck was here.

func main() {
	//defer profile.Start(profile.MemProfile).Stop()
	//defer profile.Start(profile.CPUProfile).Stop()

	// Define the hidden layers
	hiddenSizes = make([]int, 5)
	hiddenSizes[0] = 30
	hiddenSizes[1] = 30
	hiddenSizes[2] = 30
	hiddenSizes[3] = 30
	hiddenSizes[4] = 30

	// this is where the training state is held in memory, not in global scope
	// most importantly, to prevent leaks.
	// (could also fetch from disk)
	state := &recurrent.TrainingState{
		LetterSize:  5,
		HiddenSizes: hiddenSizes,
		EpochSize:   -1,
		InputSize:   -1,
		OutputSize:  -1,
	}

	state.PerplexityList = make([]float64, 0)

	solverecurrent = recurrent.NewSolver() // reinit solver
	state.TickIterator = 0

	// process the input, filter out blanks
	input, err := readFileContents("apollo.txt")
	if err != nil {
		log.Fatal("Failed reading file input", err)
	}

	state.DataSentences = strings.Split(input, "\n")
	state.InitVocab(state.DataSentences, 1) // takes count threshold for characters
	state.InitModel()

	tick(state)
}
