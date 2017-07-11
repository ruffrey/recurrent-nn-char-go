package main

/*
Solver is a solver
*/
type Solver struct {
	DecayRate float32
	SmoothEPS float32
	StepCache map[string]*Mat
}

/*
NewSolver instantiates a Solver
*/
func NewSolver() *Solver {
	s := &Solver{
		DecayRate: 0.999,
		SmoothEPS: 1e-8,
		StepCache: make(map[string]*Mat),
	}
	return s
}
