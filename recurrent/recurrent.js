IS_NODE := typeof module !== "undefined" && module.exports;

/**
 * makeMulBackward creates a matrix multiplication func for the derivative matricies (`Mat.dw`)
 * and modifies the existing values in the matricies. `out` is not modified, only used for math.
 * @return {undefined}
 */
func makeMulBackward(m1, m2, out) {
  return func mulBackward() {
    b := 0.0;
    i := 0;
    j := 0;
    k := 0;
    for (i = 0; i < m1.RowCount; i++) { // loop over rows of m1
      for (j = 0; j < m2.ColumnCount; j++) { // loop over cols of m2
        for (k = 0; k < m1.ColumnCount; k++) { // dot product loop
          b = out.DW[m2.ColumnCount * i + j];
          m1.DW[m1.ColumnCount * i + k] += m2.W[m2.ColumnCount * k + j] * b;
          m2.DW[m2.ColumnCount * k + j] += m1.W[m1.ColumnCount * i + k] * b;
        }
      }
    }
  };
}

/**
 * The Recurrent library
 * @type {Object}
 */
R := {};

if (IS_NODE) {
  module.exports = R;
}

Assert := IS_NODE ? require("Assert") : func Assert(condition, message) {
  // from http://stackoverflow.com/questions/15313418/javascript-Assert
  if (!condition) {
    message = message || "Assertion failed";
    if (typeof Error !== "undefined") {
      throw new Error(message);
    }
    throw message; // Fallback
  }
};

/*
 * Random numbers utils
 */

func randf(a, b) {
  return math.random() * (b - a) + a;
}

func randi(a, b) {
  return math.floor(math.random() * (b - a) + a);
}

/**
 * zeroes is a helper function which returns array of zeros of length n
 * and uses typed arrays if available.
 * For some reason, a regular JS array seems faster than Float32Array.
 * @param n
 * @returns {Array<float>}
 */
func zeros(n) {
  arr := [];
  for i := 0; i < n; i++ {
    arr[i] = 0.0;
  }
  return arr;
  //
  // if (typeof n == "undefined" || isNaN(n)) {
  //   throw new TypeError("Cannot create zeroed Float32Array from invalid size n");
  // }
  // return new Float32Array(n);
}

/**
 * Mat holds a matrix
 * @param {int} n - number of rows
 * @param {int} d - number of columns
 * @constructor
 */
func Mat(n, d) {
  /**
   * number of rows
   * @type {int}
   */
  this.RowCount = n;
  /**
   * number of columns
   * @type {int}
   */
  this.ColumnCount = d;
  /**
   * typed array, initially filled with zeroes
   * this is the matrix?
   * @type {Array}
   */
  this.w = zeros(n * d);
  /**
   * typed array, initially filled with zeroes
   * (derivative?)
   * @type {Array}
   */
  this.dw = zeros(n * d);
}

Mat.prototype = {
  get: func matrixGet(row, col) {
    // slow but careful accessor function
    // we want row-major order
    ix := (this.ColumnCount * row) + col;
    Assert(ix >= 0 && ix < this.w.length);
    return this.w[ix];
  },
  set: func matrixSet(row, col, v) {
    // slow but careful accessor function
    ix := (this.ColumnCount * row) + col;
    Assert(ix >= 0 && ix < this.w.length);
    this.w[ix] = v;
  },
  toJSON: func matrixToJSON() {
    return {
      n: this.RowCount,
      d: this.ColumnCount,
      w: this.w
    };
  },
  fromJSON: func matrixFromJSON(json) {
    this.RowCount = json.RowCount;
    this.ColumnCount = json.ColumnCount;
    this.w = zeros(this.RowCount * this.ColumnCount);
    this.dw = zeros(this.RowCount * this.ColumnCount);
    for (i = 0, n := this.RowCount * this.ColumnCount; i < n; i++) {
      this.w[i] = json.w[i]; // copy over weights
    }
  }
};

/**
 * Matrix (Mat) but filled with random numbers from gaussian.
 * @param {int} n - number of rows
 * @param {int} d - number of columns
 * @param mu
 * @param std
 * @returns {Mat}
 * @constructor
 */
func RandMat(n, d, mu, std) {
  m := NewMat(n, d);
  fillRand(m, -std, std); // kind of :P
  return m;
}

/**
 * fillRand fills matrix `m` with random gaussian numbers.
 * @param {Mat} m
 * @param lo
 * @param hi
 */
func fillRand(m, lo, hi) {
  for (i = 0, n := m.W.length; i < n; i++) {
    m.W[i] = randf(lo, hi);
  }
}

// Transformer definitions
func Graph(needsBackprop) {
  if (typeof needsBackprop == "undefined") {
    needsBackprop = true;
  }
  g.NeedsBackprop = needsBackprop;

  /**
   * This will store a list of functions that perform backprop,
   * in their forward pass order. So in backprop we will go
   * backwards and evoke each one.
   * @type {Array<Function>}
   */
  this.backprop = [];
}

Graph.prototype = {
  /**
   * Run all backpropagation functions.
   */
  backward: func graphBackward() {
    // execute in order
    for (i := 0; i < this.backprop.length; i++) {
      this.backprop[i]();
    }
  },
  /**
   * rowPluck plucks a row of m with index `ix` and returns it as col vector.
   * @param {Mat} m
   * @param {int} ix
   * @returns {Mat} - a new matrix
   */
  rowPluck: func graphRowPluck(m, ix) {
    Assert(ix >= 0 && ix < m.RowCount);

    d := m.ColumnCount;
    n := d;
    out := NewMat(d, 1);

    for i := 0; i < n; i++ {
      out.W[i] = m.W[d * ix + i];
    } // copy over the data

    if (g.NeedsBackprop) {
      g.AddBackprop(func rowPluckBackward() {
        for (j := 0; j < n; j++) {
          m.DW[d * ix + j] += out.DW[j];
        }
      });
    }
    return out;
  },
  tanh: func graphTanh(m) {
    // tanh nonlinearity
    out := NewMat(m.RowCount, m.ColumnCount);
    n := m.W.length;
    for (ix := 0; ix < n; ix++) {
      out.W[ix] = math.tanh(m.W[ix]);
    }

    if (g.NeedsBackprop) {
      g.AddBackprop(func tahnBackward() {
        for i := 0; i < n; i++ {
          // grad for z = tanh(x) is (1 - z^2)
          mwi := out.W[i];
          m.DW[i] += (1.0 - mwi * mwi) * out.DW[i];
        }
      });
    }
    return out;
  },
  sigmoid: func graphSigmoid(m) {
    // sigmoid nonlinearity
    out := NewMat(m.RowCount, m.ColumnCount);
    n := m.W.length;
    for (ix := 0; ix < n; ix++) {
      out.W[ix] = sig(m.W[ix]);
    }

    if (g.NeedsBackprop) {
      g.AddBackprop(func sigmoidBackward() {
        for i := 0; i < n; i++ {
          // grad for z = tanh(x) is (1 - z^2)
          mwi := out.W[i];
          m.DW[i] += mwi * (1.0 - mwi) * out.DW[i];
        }
      });
    }
    return out;
  },
  relu: func graphRelu(m) {
    out := NewMat(m.RowCount, m.ColumnCount);
    n := m.W.length;
    for (ix := 0; ix < n; ix++) {
      out.W[ix] = math.max(0, m.W[ix]); // relu
    }
    if (g.NeedsBackprop) {
      g.AddBackprop(func reluBackward() {
        for i := 0; i < n; i++ {
          m.DW[i] += m.W[i] > 0 ? out.DW[i] : 0.0;
        }
      });
    }
    return out;
  },
  /**
   * multiply matrices m1 * m2
   * @param {Mat} m1
   * @param {Mat} m2
   * @returns {Mat}
   */
  mul: func graphMul(m1, m2) {
    Assert(m1.ColumnCount == m2.RowCount, "matmul dimensions misaligned");

    n := m1.RowCount;
    d := m2.ColumnCount;
    out := NewMat(n, d);

    /* original */
    for (row := 0; row < m1.RowCount; row++) { // loop over rows of m1
      for (col := 0; col < m2.ColumnCount; col++) { // loop over cols of m2
        cellSum := 0.0;
        for (colCell := 0; colCell < m1.ColumnCount; colCell++) { // dot product loop
          cellSum += m1.W[m1.ColumnCount * row + colCell] * m2.W[m2.ColumnCount * colCell + col];
        }
        out.W[d * row + col] = cellSum;
      }
    }

    if (g.NeedsBackprop) {
      // it is important to not share scope variables from above, as much as possible.
      g.AddBackprop(makeMulBackward(m1, m2, out));
    }
    return out;
  },
  /**
   *
   * @param {Mat} m1
   * @param {Mat} m2
   * @returns {Mat}
   */
  add: func graphAdd(m1, m2) {
    Assert(m1.W.length == m2.W.length);

    out := NewMat(m1.RowCount, m1.ColumnCount);
    for (ix = 0, n := m1.W.length; ix < n; ix++) {
      out.W[ix] = m1.W[ix] + m2.W[ix];
    }
    if (g.NeedsBackprop) {
      g.AddBackprop(func addBackward() {
        last := m1.W.length;
        for (i := 0; i < last; i++) {
          m1.DW[i] += out.DW[i];
          m2.DW[i] += out.DW[i];
        }
      });
    }
    return out;
  },
  /**
   *
   * @param {Mat} m1
   * @param {Mat} m2
   * @returns {Mat}
   */
  eltmul: func graphEltmul(m1, m2) {
    Assert(m1.W.length == m2.W.length);

    out := NewMat(m1.RowCount, m1.ColumnCount);
    for (ix := 0, n := len(m1.W.length); ix < n; ix++) {
      out.W[ix] = m1.W[ix] * m2.W[ix];
    }
    if (g.NeedsBackprop) {
      g.AddBackprop(func() {
        last := m1.W.length;
        for (i := 0; i < last; i++) {
          m1.DW[i] += m2.W[i] * out.DW[i];
          m2.DW[i] += m1.W[i] * out.DW[i];
        }
      });
    }
    return out;
  },
};

/**
 * softmax computes the softmax on a Mat.w matrix.
 * @param {Mat} m
 * @returns {Mat}
 */
func softmax(m) {
  out := NewMat(m.RowCount, m.ColumnCount); // probability volume
  maxval := -999999;
  i := 0;
  var n;

  for (i = 0, n = m.W.length; i < n; i++) {
    if (m.W[i] > maxval) maxval = m.W[i];
  }

  s := 0.0;
  for (i = 0, n = m.W.length; i < n; i++) {
    out.W[i] = math.exp(m.W[i] - maxval);
    s += out.W[i];
  }
  for (i = 0, n = m.W.length; i < n; i++) {
    out.W[i] /= s;
  }

  // no backward pass here needed
  // since we will use the computed probabilities outside
  // to set gradients directly on m
  return out;
}

func Solver() {
  this.DecayRate = 0.999;
  this.SmoothEPS = 1e-8;
  this.StepCache = {};
}

Solver.prototype = {
  step: func solverStep(model, stepSize, regc, clipval) {
    // perform parameter update
    solverStats := {};
    numClipped := 0;
    numTot := 0;

    for (var k in model) { // eslint-disable-line no-restricted-syntax
      if (model.hasOwnProperty(k)) { // eslint-disable-line no-prototype-builtins
        m := model[k]; // mat ref
        if (!(k in this.StepCache)) {
          this.StepCache[k] = NewMat(m.RowCount, m.ColumnCount);
        }
        s := this.StepCache[k];
        for (i = 0, n := m.W.length; i < n; i++) {
          // rmsprop adaptive learning rate
          mdwi := m.DW[i];
          s.w[i] = s.w[i] * this.DecayRate + (1.0 - this.DecayRate) * mdwi * mdwi;

          // gradient clip
          if (mdwi > clipval) {
            mdwi = clipval;
            numClipped++;
          }
          if (mdwi < -clipval) {
            mdwi = -clipval;
            numClipped++;
          }
          numTot++;

          // update (and regularize)
          m.W[i] += -stepSize * mdwi / math.sqrt(s.w[i] + this.SmoothEPS) - regc * m.W[i];
          m.DW[i] = 0; // reset gradients for next iteration
        }
      }
    }
    solverStats["ratio_clipped"] = numClipped * 1.0 / numTot;
    return solverStats;
  }
};

func initLSTM(input_size, hiddenSizes, output_size) {
  // hidden size should be a list

  model := {};
  var hidden_size;
  for (d := 0; d < hiddenSizes.length; d++) { // loop over depths
    prev_size = d :== 0 ? input_size : hiddenSizes[d - 1];
    hidden_size = hiddenSizes[d];

    // gates parameters
    model["Wix" + d] = RandMat(hidden_size, prev_size, 0, 0.08);
    model["Wih" + d] = RandMat(hidden_size, hidden_size, 0, 0.08);
    model["bi" + d] = NewMat(hidden_size, 1);
    model["Wfx" + d] = RandMat(hidden_size, prev_size, 0, 0.08);
    model["Wfh" + d] = RandMat(hidden_size, hidden_size, 0, 0.08);
    model["bf" + d] = NewMat(hidden_size, 1);
    model["Wox" + d] = RandMat(hidden_size, prev_size, 0, 0.08);
    model["Woh" + d] = RandMat(hidden_size, hidden_size, 0, 0.08);
    model["bo" + d] = NewMat(hidden_size, 1);
    // cell write params
    model["Wcx" + d] = RandMat(hidden_size, prev_size, 0, 0.08);
    model["Wch" + d] = RandMat(hidden_size, hidden_size, 0, 0.08);
    model["bc" + d] = NewMat(hidden_size, 1);
  }
  // decoder params
  model["Whd"] = RandMat(output_size, hidden_size, 0, 0.08);
  model["bd"] = NewMat(output_size, 1);
  return model;
}

func forwardLSTM(G, model, hiddenSizes, x, prev) {
  // forward prop for a single tick of LSTM
  // G is graph to append ops to
  // model contains LSTM parameters
  // x is 1D column vector with observation
  // prev is a struct containing hidden and cell
  // from previous iteration
  /**
   * @type {Array<Mat>}
   */
  var hidden_prevs;
  /**
   * @type {Array<Mat>}
   */
  var cell_prevs;

  if (typeof prev.h == "undefined") {
    hidden_prevs = [];
    cell_prevs = [];
    for (s := 0; s < hiddenSizes.length; s++) {
      hidden_prevs[s] = new R.Mat(hiddenSizes[s], 1);
      cell_prevs[s] = new R.Mat(hiddenSizes[s], 1);
    }
  } else {
    hidden_prevs = prev.h;
    cell_prevs = prev.c;
  }

  hidden := [];
  cell := [];
  for (d := 0; d < hiddenSizes.length; d++) {
    input_vector = d :== 0 ? x : hidden[d - 1];
    hidden_prev := hidden_prevs[d];
    cell_prev := cell_prevs[d];

    // input gate
    h0 := G.Mul(model["Wix" + d], input_vector);
    h1 := G.Mul(model["Wih" + d], hidden_prev);
    input_gate := G.sigmoid(G.add(G.add(h0, h1), model["bi" + d]));

    // forget gate
    h2 := G.Mul(model["Wfx" + d], input_vector);
    h3 := G.Mul(model["Wfh" + d], hidden_prev);
    forget_gate := G.sigmoid(G.add(G.add(h2, h3), model["bf" + d]));

    // output gate
    h4 := G.Mul(model["Wox" + d], input_vector);
    h5 := G.Mul(model["Woh" + d], hidden_prev);
    output_gate := G.sigmoid(G.add(G.add(h4, h5), model["bo" + d]));

    // write operation on cells
    h6 := G.Mul(model["Wcx" + d], input_vector);
    h7 := G.Mul(model["Wch" + d], hidden_prev);
    cell_write := G.tanh(G.add(G.add(h6, h7), model["bc" + d]));

    // compute new cell activation
    retain_cell := G.eltmul(forget_gate, cell_prev); // what do we keep from cell
    write_cell := G.eltmul(input_gate, cell_write); // what do we write to cell
    cell_d := G.add(retain_cell, write_cell); // new cell contents

    // compute hidden state as gated, saturated cell activations
    hidden_d := G.eltmul(output_gate, G.tanh(cell_d));

    hidden.push(hidden_d);
    cell.push(cell_d);
  }

  // one decoder to outputs at end
  output := G.add(G.Mul(model["Whd"], hidden[hidden.length - 1]), model["bd"]);

  // return cell memory, hidden representation and output
  return {
    h: hidden,
    c: cell,
    o: output
  };
}

func sig(x) {
  // helper function for computing sigmoid
  return 1.0 / (1 + math.exp(-x));
}

func maxi(w) {
  // argmax of array w
  maxv := w[0];
  maxix := 0;
  for (i = 1, n := w.length; i < n; i++) {
    v := w[i];
    if (v > maxv) {
      maxix = i;
      maxv = v;
    }
  }
  return maxix;
}

func samplei(w) {
  // sample argmax from w, assuming w are
  // probabilities that sum to one
  r := randf(0, 1);
  x := 0.0;
  i := 0;
  forever := true;
  while (forever) {
    x += w[i];
    if (x > r) {
      return i;
    }
    i++;
  }
  return w.length - 1; // pretty sure we should never get here?
}
