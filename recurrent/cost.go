package recurrent

import "math"

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
	prev := CellMemory{}
	var probswixtarget float64
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
		// TODO: this is never changing the value, and seems to be the crux of the matter
		// fmt.Println("before: ", prev)
		prev = state.ForwardIndex(ixSource, prev)
		// fmt.Println("after: ", prev)

		// set gradients into logprobabilities
		logrithmicProbabilities := prev.Output    // interpret output as logrithmicProbabilities
		probs = Softmax(&logrithmicProbabilities) // compute the softmax probabilities

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
		Ppl:  ppl,
		Cost: cost,
	}
}
