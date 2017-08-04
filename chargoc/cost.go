package main

import (
	"math"
	"strings"

	"github.com/ruffrey/recurrent-nn-char-go/cat32"
)

/*
Cost represents the result of running the cost function.
*/
type Cost struct {
	Ppl  float64
	Cost float64
}

/*
CostFunction takes a model and a sentence and calculates the loss.
*/
func (state *TrainingState) CostFunction(sent string) Cost {
	letters := strings.Split(sent, "")
	n := len(letters)
	state.ResetBackprop(true)
	var log2ppl float64
	var cost float64

	var ixSource int
	var ixTarget int
	var prev *CellMemory
	initial := CellMemory{}
	prev = &initial
	var probs *cat32.Mat

	// loop through each letter of the selected sentence
	for i := -1; i < n; i++ {
		// start and end tokens are zeros
		// fmt.Println("i=", i, ", n=", n)
		// first step: start with START token
		if i == -1 {
			ixSource = 0
		} else {
			ixSource = state.LetterToIndex[letters[i]]
		}
		// last step: end with END token
		if i == n-1 {
			ixTarget = 0
		} else {
			ixTarget = state.LetterToIndex[letters[i+1]]
		}
		// formerly ForwardIndex. Forward propagate the sequence learner.
		lh := state.ForwardLSTM(
			state.HiddenSizes,
			state.RowPluck(state.Model["Wil"], ixSource),
			prev,
		)

		// set gradients into logprobabilities
		// interpret output as logrithmicProbabilities
		probs = cat32.Softmax(lh.Output) // compute the softmax probabilities

		// all done? END?
		if (len(probs.W) - 1) < ixTarget {
			break
		}
		log2ppl += -math.Log2(float64(probs.Value(ixTarget))) // accumulate base 2 log prob and do smoothing
		cost += -math.Log(float64(probs.Value(ixTarget)))

		// write gradients into log probabilities
		lh.Output.DW = probs.W
		lh.Output.DW[ixTarget] = cat32.Subf32x4(lh.Output.DW[ixTarget], cat32.F32_1)

		prev = lh
	}

	exponent := log2ppl / float64(n-1)
	ppl := math.Pow(2, exponent)

	return Cost{
		Ppl:  ppl,
		Cost: cost,
	}
}
