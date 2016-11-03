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
Randi makes random integers between two integers
*/
func Randi(low int, hi int) int {
	a := float64(low)
	b := float64(hi)
	return int(math.Floor(r.Float64()*(b-a) + a))
}
