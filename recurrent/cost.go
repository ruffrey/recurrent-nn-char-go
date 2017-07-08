package recurrent

import (
	"math"
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
	n := len(sent)
	state.ResetBackprop(true)
	var log2ppl float64
	var cost float64

	var ixSource int
	var ixTarget int
	var prev *CellMemory
	initial := CellMemory{}
	prev = &initial
	var probs Mat

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
		// formerly ForwardIndex. Forward propagate the sequence learner.
		lh := state.ForwardLSTM(
			state.HiddenSizes,
			state.RowPluck(state.Model["Wil"], ixSource),
			prev,
		)
		prev = lh

		// set gradients into logprobabilities
		// interpret output as logrithmicProbabilities
		probs = Softmax(prev.Output) // compute the softmax probabilities

		// all done? END?
		if (len(probs.W) - 1) < ixTarget {
			break
		}
		log2ppl += -math.Log2(probs.W[ixTarget]) // accumulate base 2 log prob and do smoothing
		cost += -math.Log(probs.W[ixTarget])

		// write gradients into log probabilities
		// TODO: this is not working
		// fmt.Println("before: ", (*prev).Output)
		prev.Output.DW = probs.W
		prev.Output.DW[ixTarget]--
		// fmt.Println("after: ", (*prev).Output)
	}

	ppl := math.Pow(2, log2ppl/float64(n-1))

	return Cost{
		Ppl:  ppl,
		Cost: cost,
	}
}
