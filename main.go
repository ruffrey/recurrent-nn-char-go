package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
	"encoding/json"
	"gopkg.in/urfave/cli.v1"
	"os"
	"github.com/getlantern/errors"
	"github.com/pkg/profile"
	"io/ioutil"
)

// model parameters
// list of sizes of hidden layers
var hiddenSizes []int

// optimization
// L2 regularization strength
var regc = 0.000001

const learningRate = 0.01

// clip gradients at this value
const clipval = 5.0

/* */

// prediction params

// how peaky model predictions should be
const sampleSoftmaxTemperature = 1.0

// max length of generated sentences
const maxCharsGenerate = 100

// various global var inits

// should be class because it needs memory for step caches
var solverecurrent *Solver

// old gradCheck was here.

func main() {
	app := cli.NewApp()
	app.Name = "ricur: A recurrent neural trainer for general text prediction."
	app.Commands = []cli.Command{
		{
			Name:  "train",
			Usage: "Train a neural network",
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "in",
					Usage: "File path to input text file (instead of `seed` text)",
				},
				cli.StringFlag{
					Name:  "seed",
					Usage: "Literal input text (instead of `in` file path)",
				},
				cli.StringFlag{
					Name:  "load",
					Usage: "Optional file path to load an existing model",
				},
				cli.StringFlag{
					Name:  "save",
					Value: "model.json",
					Usage: "File path to save the model",
				},
				cli.IntFlag{
					Name:  "depth",
					Value: 2,
					Usage: "For a new network, this is how many hidden layers deep it should be",
				},
				cli.IntFlag{
					Name:  "cells",
					Value: 25,
					Usage: "For a new network, this is how many neurons per hidden layer",
				},
			},
			Action: func(c *cli.Context) error {
				return training(
					c.String("seed"),
					c.String("in"),
					c.String("load"),
					c.String("save"),
					c.Int("depth"),
					c.Int("cells"),
				)
			},
		},
		{
			Name:  "sample",
			Usage: "Run and receive output from an existing neural network",
			Action: func(c *cli.Context) error {
				return nil
			},
		},
	}

	app.Run(os.Args)
}

func training(inputSeed string, inputFile string, loadFilepath string, saveFilepath string, depthLayers int, cellCount int) (err error) {
	// cpu profiling via PERF environment flag
	if profileWhich := os.Getenv("PERF"); profileWhich != "" {
		if profileWhich == "mem" {
			defer profile.Start(profile.MemProfile).Stop()
		} else if profileWhich == "cpu" {
			defer profile.Start(profile.CPUProfile).Stop()
		}
	}

	// this is where the training state is held in memory, not in global scope
	// most importantly, to prevent leaks.
	// (could also fetch from disk)
	var state *TrainingState
	if loadFilepath != "" {
		s, err := ioutil.ReadFile(loadFilepath)
		if err != nil {
			return err
		}
		state = &TrainingState{}
		err = json.Unmarshal(s, state)
		if err != nil {
			fmt.Println("state=", state)
			return err
		}
		hiddenSizes = state.HiddenSizes
		fmt.Println("Loaded network", hiddenSizes)
	} else {
		// new state
		// Define the hidden layers
		hiddenSizes = make([]int, depthLayers)
		for i := 0; i < depthLayers; i++ {
			hiddenSizes[i] = cellCount
		}
		state = &TrainingState{
			LetterSize:  5,
			HiddenSizes: hiddenSizes,
			EpochSize:   -1,
			InputSize:   -1,
			OutputSize:  -1,
		}
		fmt.Println("Created new network", hiddenSizes)
	}

	fmt.Println("HiddenSizes=", hiddenSizes)

	state.PerplexityList = make([]float64, 0)

	solverecurrent = NewSolver() // reinit solver
	state.TickIterator = 0

	// process the input, filter out blanks
	var input string
	if inputSeed != "" {
		input = inputSeed
	}
	if inputFile != "" {
		input, err = readFileContents(inputFile)
		if err != nil {
			return err
		}
	}
	if input == "" {
		return errors.New("Cannot proceed - input text is empty")
	}

	state.DataSentences = strings.Split(input, "\n")
	state.InitVocab(state.DataSentences, 1) // takes count threshold for characters

	if loadFilepath == "" {
		state.InitModel()
	}

	tick(state, saveFilepath)

	return err
}

func tick(state *TrainingState, saveFilepath string) {
	// sample sentence from data
	sentix := Randi(0, len(state.DataSentences))
	sent := state.DataSentences[sentix]

	t0 := time.Now().UnixNano() / 1000000 // log start timestamp ms

	// evaluate cost func on a sentence
	costStruct := state.CostFunction(sent)
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

	if math.Remainder(epoch, 1) == 0 {
		fmt.Println("Saving progress", saveFilepath)
		jsonState, err := json.Marshal(state)
		go (func() {
			if err != nil {
				fmt.Println("stringify err", err)
				return
			}
			err = writeFileContents(saveFilepath, jsonState)
			if err != nil {
				fmt.Println("Save error", err, saveFilepath)
			}
		})()
	}

	tick(state, saveFilepath)
}

func median(values []float64) (middleValue float64) {
	sort.Float64s(values)
	lenValues := len(values)
	halfway := int(math.Floor(float64(lenValues / 2)))
	middleValue = values[halfway]
	return middleValue
}
