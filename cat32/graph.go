package cat32

import (
	"math"
	"sync"

	"github.com/bjwbell/gensimd/simd"
)

var F32_1 = simd.F32x4{1.0, 1.0, 1.0, 1.0}

type backprop func()

/*
Graph is the neural network graph.
*/
type Graph struct {
	NeedsBackprop bool
	Backprop      []backprop // holds backprop functions
	bpMux         sync.Mutex // modifying backprop array
}

/*
ResetBackprop instantiates a new Graph
*/
func (g *Graph) ResetBackprop(needsBackprop bool) {
	g.NeedsBackprop = needsBackprop
	g.Backprop = make([]backprop, 0)
}

/*
AddBackprop adds the backpropagation function `f` to the end of the Backrop list.
*/
func (g *Graph) AddBackprop(f func()) {
	g.bpMux.Lock()
	g.Backprop = append(g.Backprop, f)
	g.bpMux.Unlock()
}

/*
Backward runs all backpropagation functions, in order.
*/
func (g *Graph) Backward() {
	var wg sync.WaitGroup
	totalBackprops := len(g.Backprop)
	i := totalBackprops - 1
	for i >= 0 {
		fn := g.Backprop[i]
		wasCreatedInGoneGoroutine := fn == nil
		if wasCreatedInGoneGoroutine {
			// Introduced in commit: 7807054d4594a08bdee0412a5782346c579a9b94
			// Once we started matrix math in parallel during the forward
			// pass, and some of the results were thrown out, we ended
			// up with callback functions getting cleaned up, I think.
			// Probably because the goroutine had been cleaned up, and
			// the result was only used temporarily inside the goroutine.
			// This is probably a Go runtime bug, mixed in with an unnecessary
			// callback - nonetheless easily worked around.
			wg.Done()
			return
		}
		fn()
		wg.Done()
	}
	g.Backprop = nil
}

/*
RowPluck plucks a row of m with index `ix` and returns it as col vector.
*/
func (g *Graph) RowPluck(m *Mat, ix int) (out *Mat) {
	Assert(ix >= 0 && ix < m.RowCount, "RowPluck invalid number of rows")

	d := m.ColumnCount
	n := d
	out = NewMat(d, 1)
	out.W = m.W[ix*d : ix*d+d]

	if g.NeedsBackprop {
		backpropRowPluck := func() {
			for j := 0; j < n; j++ {
				m.DW[d*ix+j] = AddF32x4(m.DW[d*ix+j], out.DW[j])
			}
		}
		g.AddBackprop(backpropRowPluck)
	}

	return out
}

/*
Tanh does tanh nonlinearity
*/
func (g *Graph) Tanh(m *Mat) *Mat {
	out := NewMat(m.RowCount, m.ColumnCount)
	n := len(m.W)
	f := 0
	for ix := 0; ix < n; ix++ {
		for f = 0; f < 4; f++ {
			out.W[ix][f] = float32(math.Tanh(float64(m.W[ix][f])))
		}
	}

	if g.NeedsBackprop {
		backpropTahn := func() {
			for i := 0; i < n; i++ {
				// grad for z = tanh(x) is (1 - z^2)
				mwi := out.W[i]
				m.DW[i] = AddF32x4(m.DW[i], SubF32x4(F32_1, SubF32x4(MulF32x4(mwi, mwi), out.DW[i])))
			}
		}
		g.AddBackprop(backpropTahn)
	}
	return out
}

/*
Sigmoid does sigmoid things.
*/
func (g *Graph) Sigmoid(m *Mat) *Mat {
	// sigmoid nonlinearity
	out := NewMat(m.RowCount, m.ColumnCount)
	n := len(m.W)
	var exps simd.F32x4
	for ix := 0; ix < n; ix++ {
		exps = simd.F32x4{
			float32(math.Exp(-float64(m.W[ix][0]))),
			float32(math.Exp(-float64(m.W[ix][1]))),
			float32(math.Exp(-float64(m.W[ix][2]))),
			float32(math.Exp(-float64(m.W[ix][3]))),
		}
		out.W[ix] = DivF32x4(F32_1, AddF32x4(F32_1, exps))
	}

	if g.NeedsBackprop {
		backpropSigmoid := func() {
			for i := 0; i < n; i++ {
				// grad for z = tanh(x) is (1 - z^2)
				mwi := out.W[i]
				m.DW[i] = AddF32x4(m.DW[i], MulF32x4(mwi, MulF32x4(SubF32x4(F32_1, mwi), out.DW[i])))
			}
		}
		g.AddBackprop(backpropSigmoid)
	}

	return out
}

/*
Mul multiplies two matrices
*/
func (g *Graph) Mul(m1 *Mat, m2 *Mat) *Mat {
	Assert(m1.ColumnCount == m2.RowCount, "matmul dimensions misaligned")

	n := m1.RowCount
	d := m2.ColumnCount
	out := NewMat(n, d)

	/* original */
	for row := 0; row < m1.RowCount/4; row++ { // loop over rows of m1
		for col := 0; col < m2.ColumnCount; col++ { // loop over cols of m2
			cellSum := simd.F32x4{0, 0, 0, 0}
			for colCell := 0; colCell < m1.ColumnCount; colCell++ { // dot product loop
				cellSum = AddF32x4(cellSum, MulF32x4(m1.W[m1.ColumnCount*row+colCell], m2.W[m2.ColumnCount*colCell+col]))
			}
			out.W[d*row+col] = cellSum
		}
	}

	if g.NeedsBackprop {
		backpropMul := func() {
			for i := 0; i < m1.RowCount; i++ { // loop over rows of m1
				// runtime overhead of parallelizing this with goroutines
				// makes it slower - each backprop func has its own goroutine
				// already.
				for j := 0; j < m2.ColumnCount; j++ { // loop over cols of m2
					for k := 0; k < m1.ColumnCount; k++ { // dot product loop
						// reach back through `out` pointer
						b := out.DW[m2.ColumnCount*i+j]
						m1dw := m1.DW[m1.ColumnCount*i+k]
						m2dw := m2.DW[m2.ColumnCount*k+j]
						m1.DW[m1.ColumnCount*i+k] = AddF32x4(m1dw, MulF32x4(m2.W[m2.ColumnCount*k+j], b))
						m2.DW[m2.ColumnCount*k+j] = AddF32x4(m2dw, MulF32x4(m1.W[m1.ColumnCount*i+k], b))
					}
				}
			}
		}
		g.AddBackprop(backpropMul)
	}
	return out
}

/*
Add adds two matrices
*/
func (g *Graph) Add(m1 *Mat, m2 *Mat) *Mat {
	Assert(len(m1.W) == len(m2.W), "Cannot add arrays")

	out := NewMat(m1.RowCount, m1.ColumnCount)
	ix := 0
	n := len(m1.W)

	for ; ix < n; ix++ {
		out.W[ix] = AddF32x4(m1.W[ix], m2.W[ix])
	}
	if g.NeedsBackprop {
		backpropAdd := func() {
			last := len(m1.W)
			for i := 0; i < last; i++ {
				m1.DW[i] = AddF32x4(m1.DW[i], out.DW[i])
				m2.DW[i] = AddF32x4(m2.DW[i], out.DW[i])
			}
		}
		g.AddBackprop(backpropAdd)
	}
	return out
}

/*
Eltmul does element-wise multiplication
*/
func (g *Graph) Eltmul(m1 *Mat, m2 *Mat) *Mat {
	Assert(len(m1.W) == len(m2.W), "Cannot Eltmul")

	out := NewMat(m1.RowCount, m1.ColumnCount)
	ix := 0
	n := len(m1.W)
	for ; ix < n; ix++ {
		out.W[ix] = MulF32x4(m1.W[ix], m2.W[ix])
	}
	if g.NeedsBackprop {
		backpropEtlmul := func() {
			last := len(m1.W)
			for i := 0; i < last; i++ {
				m1.DW[i] = AddF32x4(m1.DW[i], MulF32x4(m2.W[i], out.DW[i]))
				m2.DW[i] = AddF32x4(m2.DW[i], MulF32x4(m1.W[i], out.DW[i]))
			}
		}
		g.AddBackprop(backpropEtlmul)
	}

	return out
}
