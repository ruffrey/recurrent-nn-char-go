package recurrent

import (
	"math"
	"math/rand"
	"time"
)

var randSource = rand.NewSource(time.Now().UnixNano())
var r = rand.New(randSource)

/*
Randf makes random numbers
*/
func Randf(a float64, b float64) float64 {
	return r.Float64()*(b-a) + a
}

/*
Randi makes random integers
*/
func Randi(a float64, b float64) float64 {
	return math.Floor(r.Float64()*(b-a) + a)
}
