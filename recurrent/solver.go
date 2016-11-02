package recurrent

import "math"

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

/*
Step does a step.
Should model be a poiner? unable to loop over it if not.
*/
func (solver *Solver) Step(model Model, stepSize float64, regc float64, clipval float64) SolverStats {
	// perform parameter update
	solverStats := SolverStats{}
	numClipped := 0.0
	numTot := 0.0

	for k, m := range model {
		_, hasKey := solver.StepCache[k]
		if !hasKey {
			solver.StepCache[k] = NewMat(m.RowCount, m.ColumnCount)
		}
		s := solver.StepCache[k]
		i := 0
		n := len(m.W)
		for ; i < n; i++ {
			// rmsprop adaptive learning rate
			mdwi := m.DW[i]
			s.W[i] = s.W[i]*solver.DecayRate + (1.0-solver.DecayRate)*mdwi*mdwi

			// gradient clip
			if mdwi > clipval {
				mdwi = clipval
				numClipped++
			}
			if mdwi < -clipval {
				mdwi = -clipval
				numClipped++
			}
			numTot++

			// update (and regularize)
			m.W[i] += -stepSize*mdwi/math.Sqrt(s.W[i]+solver.SmoothEPS) - regc*m.W[i]
			m.DW[i] = 0 // reset gradients for next iteration
		}
	}
	solverStats["ratio_clipped"] = numClipped * 1.0 / numTot
	return solverStats
}
