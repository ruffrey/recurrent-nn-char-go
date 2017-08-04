package main

import (
	"github.com/bjwbell/gensimd/simd"
	"github.com/ruffrey/recurrent-nn-char-go/cat32"
)

/*
Solver is a solver
*/
type Solver struct {
	DecayRate simd.F32x4
	SmoothEPS simd.F32x4
	StepCache map[string]*cat32.Mat
}

/*
NewSolver instantiates a Solver
*/
func NewSolver() *Solver {
	s := &Solver{
		// easier for math to have them in this format
		DecayRate: simd.F32x4{0.999, 0.999, 0.999, 0.999},
		SmoothEPS: simd.F32x4{1e-8, 1e-8, 1e-8, 1e-8},
		StepCache: make(map[string]*cat32.Mat),
	}
	return s
}
