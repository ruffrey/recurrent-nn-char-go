package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/getlantern/errors"
	"github.com/pkg/profile"
	"gopkg.in/urfave/cli.v1"
)

// model parameters

// optimization
/*
regc is L2 regularization strength, which reduces complexity,
or helps prevent overfitting.
original 0.000001
*/
var regc float32

/*
learningRate is how much to increase or decrease weights (AKA stepSize)
original 0.01
*/
var learningRate float32

/*
clipval is how high gradients (derivatives) can be maximum before
they are clipped to this value
original 5.0
*/
var clipval float32

/*
sequenceLength is how many characters to use when

(I think!)

Other definitions:
- number of timesteps to unroll for
- size of letter embeddings
- frame size
*/
var sequenceLength int

/* */

// prediction params

// how peaky model predictions should be
const sampleSoftmaxTemperature = 1.0

// max length of generated sentences
const maxCharsGenerate = 500

// various global var inits

// should be class because it needs memory for step caches
var solverecurrent *Solver

// old gradCheck was here.

func main() {
	app := cli.NewApp()
	app.Name = "ricur: A recurrent neural trainer for general text prediction."
	app.Author = "Jeff H. Parrish"
	app.Email = "jeffhparrish@gmail.com"
	app.Copyright = "Copyright (c) 2017 Jeff H. Parrish"
	app.Version = "0.1.0"
	app.Usage = ""
	app.Commands = []cli.Command{
		{
			Name:  "train",
			Usage: "Train a neural network",
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "in",
					Usage: "Path to input text `file` (instead of --seed)",
				},
				cli.StringFlag{
					Name:  "seed",
					Usage: "Literal input `text` (instead of `in` file path)",
				},
				cli.StringFlag{
					Name:  "load",
					Usage: "Optional `file` path to load an existing model",
				},
				cli.StringFlag{
					Name:  "save",
					Value: "models/model.json",
					Usage: "(default=models/model.json) `file` path to save the model",
				},
				cli.IntSliceFlag{
					Name:  "hidden",
					Value: &cli.IntSlice{30, 30, 30},
					Usage: "For a new network, this is a representation of hidden layer sizes, where each value in the list is a layer of `int` size. Example: --hidden={100,50,75}",
				},
				cli.Float64Flag{
					Name:  "learn",
					Value: 0.01,
					Usage: "(optional) Optimization param: `float32` , influences the amount of neuron weight changes",
				},
				cli.Float64Flag{
					Name:  "regc",
					Value: 0.000001,
					Usage: "(optional) Regularization: `float32` reduces complexity / overfitting",
				},
				cli.Float64Flag{
					Name:  "gradmax",
					Value: 5.0,
					Usage: "(optional) Gradient Clip: `float32` max value allowed for derivatives of weights before they are capped",
				},
				cli.Float64Flag{
					Name:  "seqlen",
					Value: 10,
					Usage: "(optional) Sequence Length: `int` frame size",
				},
			},
			Before: func(c *cli.Context) error {
				learningRate = float32(c.Float64("learn"))
				regc = float32(c.Float64("regc"))
				clipval = float32(c.Float64("gradmax"))
				sequenceLength = c.Int("seqlen")

				return nil
			},
			Action: func(c *cli.Context) error {
				hidden := c.IntSlice("hidden")
				if c.IsSet("hidden") {
					// cut of beginning default if user passed custom
					hidden = hidden[3:]
				}
				return training(
					c.String("seed"),
					c.String("in"),
					c.String("load"),
					c.String("save"),
					hidden,
				)
			},
		},
		{
			Name:  "sample",
			Usage: "Run and receive output from an existing neural network",
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "load",
					Usage: "`file` path to load an existing model",
				},
				cli.StringFlag{
					Name:  "seed",
					Usage: "Text to use in prediction",
				},
			},
			Action: func(c *cli.Context) error {
				loadFilepath := c.String("load")
				if loadFilepath == "" {
					return errors.New("Missing required filepath to model: --load")
				}
				s, err := ioutil.ReadFile(loadFilepath)
				if err != nil {
					return err
				}
				state := &TrainingState{}
				err = json.Unmarshal(s, state)
				if err != nil {
					fmt.Println("state=", state)
					return err
				}

				sentences := strings.Split(c.String("seed"), "\n")
				solver := NewSolver()
				for i := 0; i < len(sentences); i++ {
					// load up the gradients before prediction
					fmt.Println("--", sentences[i], "--")
					state.CostFunction(sentences[i])
					state.Backward()
					state.StepSolver(solver, learningRate, regc, clipval)
					pred := state.PredictSentence(true, sampleSoftmaxTemperature, maxCharsGenerate, sentences[i])
					fmt.Println(pred)
				}

				return nil
			},
		},
	}

	app.Run(os.Args)
}

func training(inputSeed string, inputFile string, loadFilepath string, saveFilepath string, defaultHiddenLayers []int) (err error) {
	// cpu profiling via PERF environment flag
	if profileWhich := os.Getenv("PERF"); profileWhich != "" {
		if profileWhich == "mem" {
			defer profile.Start(profile.MemProfile).Stop()
		} else if profileWhich == "cpu" {
			defer profile.Start(profile.CPUProfile).Stop()
		}
	}
	fmt.Println("Optimization params:")
	fmt.Println("  learn rate=", learningRate)
	fmt.Println("  regularization=", regc)
	fmt.Println("  gradient clip=", clipval)
	fmt.Println("  sequence length=", sequenceLength)

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
		fmt.Println("Loaded network\n ", state.HiddenSizes)
	} else {
		// new state
		// Define the hidden layers
		if len(defaultHiddenLayers) == 0 {
			return errors.New("Cannot create a new network that is empty")
		}
		state = &TrainingState{
			HiddenSizes: defaultHiddenLayers,
			EpochSize:   -1,
			InputSize:   -1,
			OutputSize:  -1,
		}
		fmt.Println("Created new network\n ", state.HiddenSizes)
	}

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

	if loadFilepath == "" {
		state.InitVocab(state.DataSentences, 1) // takes count threshold for characters
	}
	state.EpochSize = len(state.DataSentences)
	if loadFilepath == "" {
		state.InitModel()
	}

	for {
		tick(state, saveFilepath)
	}

	return err
}

func tick(state *TrainingState, saveFilepath string) {
	// sample sentence from data
	sentix := randi(0, len(state.DataSentences))
	sent := state.DataSentences[sentix]

	t0 := time.Now().UnixNano() / 1000000 // log start timestamp ms

	// evaluate cost func on a sentence
	costStruct := state.CostFunction(sent)
	// use built up graph to compute backprop (set .DW fields in mats)
	state.Backward()

	// perform param update
	state.StepSolver(solverecurrent, learningRate, regc, clipval)

	// keep track of perplexity between printing progress
	state.PerplexityList = append(state.PerplexityList, costStruct.Ppl)

	// evaluate now and then
	state.TickIterator++

	epoch := float64(state.TickIterator) / float64(state.EpochSize)

	if math.Remainder(float64(state.TickIterator), 250) == 0 {
		t1 := time.Now().UnixNano() / 1000000 // ms
		tickTime := t1 - t0

		pred := ""
		fmt.Println("---------------------")
		// draw samples
		for q := 0; q < 2; q++ {
			pred = state.PredictSentence(true, sampleSoftmaxTemperature, maxCharsGenerate, "")
			fmt.Println(pred)
		}
		fmt.Println("---------------------")
		medianPerplexity := median(state.PerplexityList)
		state.PerplexityList = make([]float64, 0)

		fmt.Println("epoch=", epoch)
		fmt.Println("ticktime", tickTime, "ms")
		fmt.Println("medianPerplexity", medianPerplexity)

		iepoch := int(epoch)
		isNewEpoch := iepoch != 0 && iepoch > state.lastSaveEpoch
		if isNewEpoch {
			state.lastSaveEpoch = iepoch
			go saveState(state, saveFilepath)
		}
	}
}

func saveState(state *TrainingState, saveFilepath string) {
	fmt.Println("Saving progress...", saveFilepath)
	jsonState, err := json.Marshal(state)
	if err != nil {
		fmt.Println("stringify err", err)
		return
	}
	err = writeFileContents(saveFilepath, jsonState)
	if err != nil {
		fmt.Println("Save error", err, saveFilepath)
	} else {
		fmt.Println("  ok - ", saveFilepath)
	}
}

func median(values []float64) (middleValue float64) {
	sort.Float64s(values)
	lenValues := len(values)
	halfway := int(math.Floor(float64(lenValues / 2)))
	middleValue = values[halfway]
	return middleValue
}

/*
randi makes random integers between two integers
*/
func randi(low int, hi int) int {
	a := float64(low)
	b := float64(hi)
	return int(math.Floor(rand.Float64()*(b-a) + a))
}
