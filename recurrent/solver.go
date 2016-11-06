package recurrent

/*
Solver is a solver
*/
type Solver struct {
	DecayRate float64
	SmoothEPS float64
	StepCache map[string]Mat
}

/*
SolverStats is the result of running the solver.
*/
type SolverStats map[string]float64

/*
NewSolver instantiates a Solver
*/
func NewSolver() Solver {
	s := Solver{
		DecayRate: 0.999,
		SmoothEPS: 1e-8,
		StepCache: make(map[string]Mat),
	}
	return s
}
