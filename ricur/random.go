package main

import (
	"math"
	"math/rand"
)

/*
Randf makes random numbers
*/
func Randf(a float32, b float32) float32 {
	return rand.Float32()*(b-a) + a
}

/*
Randi makes random integers between two integers
*/
func Randi(low int, hi int) int {
	a := float64(low)
	b := float64(hi)
	return int(math.Floor(rand.Float64()*(b-a) + a))
}
