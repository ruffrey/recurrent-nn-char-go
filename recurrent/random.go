package recurrent

import (
	"math"
	"math/rand"
)

/*
Randf makes random numbers
*/
func Randf(a float64, b float64) float64 {
	return rand.Float64()*(b-a) + a
}

/*
Randi makes random integers between two integers
*/
func Randi(low int, hi int) int {
	a := float64(low)
	b := float64(hi)
	return int(math.Floor(rand.Float64()*(b-a) + a))
}
