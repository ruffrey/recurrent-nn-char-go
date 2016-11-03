package recurrent

import "math"

type backprop func()

/*
Graph is the neural network graph.
*/
type Graph struct {
	NeedsBackprop bool
	Backprop      []backprop // holds backprop functions
}

/*
NewGraph instantiates a new Graph
*/
func NewGraph(needsBackprop bool) Graph {
	g := Graph{
		NeedsBackprop: needsBackprop,
		Backprop:      make([]backprop, 0),
	}

	return g
}

/*
AddBackprop adds the backpropagation function `f` to the end of the Backrop list.
*/
func (g *Graph) AddBackprop(f func()) {
	g.Backprop = append(g.Backprop, f)
}

/*
Backward runs all backpropagation functions, in order.
*/
func (g *Graph) Backward() {
	for _, fn := range g.Backprop {
		fn()
	}
	g.Backprop = nil
}

/*
RowPluck plucks a row of m with index `ix` and returns it as col vector.
*/
func (g *Graph) RowPluck(m *Mat, ix int) Mat {
	Assert(ix >= 0 && ix < m.RowCount, "RowPluck invalid number of rows")

	d := m.ColumnCount
	n := d
	out := NewMat(d, 1)

	for i := 0; i < n; i++ {
		out.W[i] = m.W[d*ix+i]
	} // copy over the data

	if g.NeedsBackprop {
		backpropRowPluck := func() {
			for j := 0; j < n; j++ {
				m.DW[d*ix+j] += out.DW[j]
			}
		}
		g.AddBackprop(backpropRowPluck)
	} else {
		m = nil // avoid leaks
	}
	return out
}

/*
Tanh does tanh nonlinearity
*/
func (g *Graph) Tanh(m *Mat) Mat {
	out := NewMat(m.RowCount, m.ColumnCount)
	n := len(m.W)
	for ix := 0; ix < n; ix++ {
		out.W[ix] = math.Tanh(m.W[ix])
	}

	if g.NeedsBackprop {
		backpropTahn := func() {
			for i := 0; i < n; i++ {
				// grad for z = tanh(x) is (1 - z^2)
				mwi := out.W[i]
				m.DW[i] += (1.0 - mwi*mwi) * out.DW[i]
			}
			m = nil // avoid leaks
		}
		g.AddBackprop(backpropTahn)
	} else {
		m = nil // avoid leaks
	}
	return out
}

/*
Sigmoid does sigmoid things.
*/
func (g *Graph) Sigmoid(m *Mat) Mat {
	// sigmoid nonlinearity
	out := NewMat(m.RowCount, m.ColumnCount)
	n := len(m.W)
	for ix := 0; ix < n; ix++ {
		out.W[ix] = 1.0 / (1 + math.Exp(-m.W[ix]))
	}

	if g.NeedsBackprop {
		backpropSigmoid := func() {
			for i := 0; i < n; i++ {
				// grad for z = tanh(x) is (1 - z^2)
				mwi := out.W[i]
				m.DW[i] += mwi * (1.0 - mwi) * out.DW[i]
			}
			m = nil // avoid leaks
		}
		g.AddBackprop(backpropSigmoid)
	} else {
		m = nil // avoid leaks
	}

	return out
}

/*
Relu does something
*/
func (g *Graph) Relu(m *Mat) Mat {
	out := NewMat(m.RowCount, m.ColumnCount)
	n := len(m.W)
	for ix := 0; ix < n; ix++ {
		out.W[ix] = math.Max(0, m.W[ix]) // relu
	}
	if g.NeedsBackprop {
		backpropRelu := func() {
			for i := 0; i < n; i++ {
				if m.W[i] > 0 {
					m.DW[i] += out.DW[i]
				}
			}
			m = nil // avoid leaks
		}
		g.AddBackprop(backpropRelu)
	} else {
		m = nil // avoid leaks
	}

	return out
}

/*
Mul multiplies two matrices
*/
func (g *Graph) Mul(m1 *Mat, m2 *Mat) Mat {
	Assert(m1.ColumnCount == m2.RowCount, "matmul dimensions misaligned")

	n := m1.RowCount
	d := m2.ColumnCount
	out := NewMat(n, d)

	/* original */
	for row := 0; row < m1.RowCount; row++ { // loop over rows of m1
		for col := 0; col < m2.ColumnCount; col++ { // loop over cols of m2
			cellSum := 0.0
			for colCell := 0; colCell < m1.ColumnCount; colCell++ { // dot product loop
				cellSum += m1.W[m1.ColumnCount*row+colCell] * m2.W[m2.ColumnCount*colCell+col]
			}
			out.W[d*row+col] = cellSum
		}
	}

	if g.NeedsBackprop {
		// it is important to not share scope variables from above, as much as possible.
		backpropMul := func() {
			b := 0.0
			for i := 0; i < m1.RowCount; i++ { // loop over rows of m1
				for j := 0; j < m2.ColumnCount; j++ { // loop over cols of m2
					for k := 0; k < m1.ColumnCount; k++ { // dot product loop
						b = out.DW[m2.ColumnCount*i+j]
						m1.DW[m1.ColumnCount*i+k] += m2.W[m2.ColumnCount*k+j] * b
						m2.DW[m2.ColumnCount*k+j] += m1.W[m1.ColumnCount*i+k] * b
					}
				}
			}
			m1 = nil // avoid leaks
			m2 = nil // avoid leaks
		}
		g.AddBackprop(backpropMul)
	} else {
		m1 = nil // avoid leaks
		m2 = nil // avoid leaks
	}
	return out
}

/*
Add adds two matrices
*/
func (g *Graph) Add(m1 *Mat, m2 *Mat) Mat {
	Assert(len(m1.W) == len(m2.W), "Cannot add arrays")

	out := NewMat(m1.RowCount, m1.ColumnCount)
	ix := 0
	n := len(m1.W)

	for ; ix < n; ix++ {
		out.W[ix] = m1.W[ix] + m2.W[ix]
	}
	if g.NeedsBackprop {
		backpropAdd := func() {
			last := len(m1.W)
			for i := 0; i < last; i++ {
				m1.DW[i] += out.DW[i]
				m2.DW[i] += out.DW[i]
			}
			m1 = nil // avoid leaks
			m2 = nil // avoid leaks
		}
		g.AddBackprop(backpropAdd)
	} else {
		m1 = nil // avoid leaks
		m2 = nil // avoid leaks
	}
	return out
}

/*
Eltmul does something with multiplication
*/
func (g *Graph) Eltmul(m1 *Mat, m2 *Mat) Mat {
	Assert(len(m1.W) == len(m2.W), "Cannot Eltmul")

	out := NewMat(m1.RowCount, m1.ColumnCount)
	ix := 0
	n := len(m1.W)
	for ; ix < n; ix++ {
		out.W[ix] = m1.W[ix] * m2.W[ix]
	}
	if g.NeedsBackprop {
		backpropEtlmul := func() {
			last := len(m1.W)
			for i := 0; i < last; i++ {
				m1.DW[i] += m2.W[i] * out.DW[i]
				m2.DW[i] += m1.W[i] * out.DW[i]
			}
			m1 = nil // avoid leaks
			m2 = nil // avoid leaks
		}
		g.AddBackprop(backpropEtlmul)
	} else {
		m1 = nil // avoid leaks
		m2 = nil // avoid leaks
	}

	return out
}
