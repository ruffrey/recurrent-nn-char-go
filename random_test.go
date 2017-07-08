package main

import (
	"math"
	"testing"
)

func TestRandf(t *testing.T) {
	t.Run("Randf() produces a random number between two others", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			r := Randf(1, 6)
			if r < 1 || r > 6 {
				t.Fail()
			}
		}
	})
	t.Run("Randi() produces a random int between two others", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			r := Randi(4, 8)
			rem := math.Remainder(float64(r), 1)

			if r < 1 || r > 8 {
				t.Fail()
			}
			if rem != 0 {
				t.Fail()
			}
		}
	})
}
