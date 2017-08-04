package cat32

import (
	"math/rand"
)

/*
Randf makes random numbers
*/
func Randf(a float32, b float32) float32 {
	return rand.Float32()*(b-a) + a
}
