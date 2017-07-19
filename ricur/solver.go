package main

import "github.com/ruffrey/recurrent-nn-char-go/mat32"

/*
Solver is a solver
*/
type Solver struct {
	DecayRate float32
	SmoothEPS float32
	StepCache map[string]*mat32.Mat
}

/*
NewSolver instantiates a Solver
*/
func NewSolver() *Solver {
	s := &Solver{
		DecayRate: 0.999,
		SmoothEPS: 1e-8,
		StepCache: make(map[string]*mat32.Mat),
	}
	return s
}
