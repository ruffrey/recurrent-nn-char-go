package main

import "github.com/ruffrey/recurrent-nn-char-go/mat8"

/*
Solver is a solver
*/
type Solver struct {
	DecayRate float32
	SmoothEPS float32
	StepCache map[string]*mat8.Mat
}

/*
NewSolver instantiates a Solver
*/
func NewSolver() *Solver {
	s := &Solver{
		DecayRate: 0.999,
		SmoothEPS: 1e-8,
		StepCache: make(map[string]*mat8.Mat),
	}
	return s
}
