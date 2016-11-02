const R = require('../src/recurrent');
const fs = require('fs');

const debug = require('debug')('recurrent');

// model parameters
var generator = 'lstm'; // can be 'rnn' or 'lstm'
var hiddenSizes = [20, 20]; // list of sizes of hidden layers
var letterSize = 5; // size of letter embeddings

// optimization
var regc = 0.000001; // L2 regularization strength
var learning_rate = 0.01; // learning rate
var clipval = 3; // clip gradients at this value

/* */

// prediction params
var sample_softmax_temperature = 1.0; // how peaky model predictions should be
var max_chars_gen = 100; // max length of generated sentences

// various global var inits
var epoch_size = -1;
var input_size = -1;
var output_size = -1;
var letterToIndex = {};
var indexToLetter = {};
var vocab = [];
var data_sents = [];
var solver = new R.Solver(); // should be class because it needs memory for step caches

// unsure about these former accidental (?) globals
var logprobs;
var oldval;

var model = {};

function initVocab(sents, count_threshold) {
  // go over all characters and keep track of all unique ones seen
  var txt = sents.join(''); // concat all

  // count up all characters
  var d = {};
  for (var i = 0, n = txt.length; i < n; i++) {
    var txti = txt[i];
    if (txti in d) {
      d[txti] += 1;
    } else {
      d[txti] = 1;
    }
  }

  // filter by count threshold and create pointers
  letterToIndex = {};
  indexToLetter = {};
  vocab = [];
  // NOTE: start at one because we will have START and END tokens!
  // that is, START token will be index 0 in model letter vectors
  // and END token will be index 0 in the next character softmax
  var q = 1;
  for (var ch in d) {
    if (d.hasOwnProperty(ch)) {
      if (d[ch] >= count_threshold) {
        // add character to vocab
        letterToIndex[ch] = q;
        indexToLetter[q] = ch;
        vocab.push(ch);
        q++;
      }
    }
  }

  // globals written: indexToLetter, letterToIndex, vocab (list), and:
  input_size = vocab.length + 1;
  output_size = vocab.length + 1;
  epoch_size = sents.length;
  debug('found ' + vocab.length + ' distinct characters: ' + vocab.join(''));
}

function utilAddToModel(modelto, modelfrom) {
  for (var k in modelfrom) {
    if (modelfrom.hasOwnProperty(k)) {
      // copy over the pointer but change the key to use the append
      modelto[k] = modelfrom[k];
    }
  }
}

function initModel() {
  // letter embedding vectors
  var tempModel = {};
  tempModel['Wil'] = new R.RandMat(input_size, letterSize, 0, 0.08);

  if (generator === 'rnn') {
    var rnn = R.initRNN(letterSize, hiddenSizes, output_size);
    utilAddToModel(tempModel, rnn);
  } else {
    var lstm = R.initLSTM(letterSize, hiddenSizes, output_size);
    utilAddToModel(tempModel, lstm);
  }

  return tempModel;
}

function reinit() {
  // note: reinit writes global vars
  solver = new R.Solver(); // reinit solver

  perplexityList = [];
  tick_iter = 0;

  // process the input, filter out blanks
  var data_sents_raw = fs.readFileSync(`${__dirname}/input.txt`, { encoding: 'utf8' }).split('\n');
  data_sents = [];
  for (var i = 0; i < data_sents_raw.length; i++) {
    var sent = data_sents_raw[i].trim();
    if (sent.length > 0) {
      data_sents.push(sent);
    }
  }

  initVocab(data_sents, 1); // takes count threshold for characters
  model = initModel();
}

function saveModel() { // eslint-disable-line
  var out = {};
  out['hiddenSizes'] = hiddenSizes;
  out['generator'] = generator;
  out['letterSize'] = letterSize;
  var model_out = {};
  Object.keys(model).forEach((k) => {
    model_out[k] = model[k].toJSON();
  });

  out['model'] = model_out;
  const solver_out = {};
  solver_out['DecayRate'] = solver.DecayRate;
  solver_out['SmoothEPS'] = solver.SmoothEPS;
  const StepCache_out = {};
  Object.keys(solver.StepCache).forEach((k) => {
    StepCache_out[k] = solver.StepCache[k].toJSON();
  });
  solver_out['StepCache'] = StepCache_out;
  out['solver'] = solver_out;
  out['letterToIndex'] = letterToIndex;
  out['indexToLetter'] = indexToLetter;
  out['vocab'] = vocab;
  // $("#tio").val(JSON.stringify(out));
}

function loadModel(j) { // eslint-disable-line
  hiddenSizes = j.hiddenSizes;
  generator = j.generator;
  letterSize = j.letterSize;
  model = {};
  Object.keys(j.model).forEach((k) => {
    const matjson = j.model[k];
    model[k] = new R.Mat(1, 1);
    model[k].fromJSON(matjson);
  });

  solver = new R.Solver(); // have to reinit the solver since model changed
  solver.DecayRate = j.solver.DecayRate;
  solver.SmoothEPS = j.solver.SmoothEPS;
  solver.StepCache = {};
  Object.keys(j.solver.StepCache).forEach((k) => {
    const matjson = j.solver.StepCache[k];
    solver.StepCache[k] = new R.Mat(1, 1);
    solver.StepCache[k].fromJSON(matjson);
  });

  letterToIndex = j['letterToIndex'];
  indexToLetter = j['indexToLetter'];
  vocab = j['vocab'];

  // reinit these
  perplexityList = [];
  tick_iter = 0;
}

function forwardIndex(G, mod, ix, prev) {
  var x = G.rowPluck(mod['Wil'], ix);
  // forward prop the sequence learner
  var out_struct;
  if (generator === 'rnn') {
    out_struct = R.forwardRNN(G, mod, hiddenSizes, x, prev);
  } else {
    out_struct = R.forwardLSTM(G, mod, hiddenSizes, x, prev);
  }
  return out_struct;
}

function predictSentence(mod, samplei, temperature) {
  if (typeof samplei === 'undefined') {
    samplei = false;
  }
  if (typeof temperature === 'undefined') {
    temperature = 1.0;
  }

  const G = new R.Graph(false);
  let s = '';
  let prev = {};
  const forever = true;
  while (forever) {
    // RNN tick
    let ix = s.length === 0 ? 0 : letterToIndex[s[s.length - 1]];
    const lh = forwardIndex(G, mod, ix, prev);
    prev = lh;

    // sample predicted letter
    logprobs = lh.o;
    if (temperature !== 1.0 && samplei) {
      // scale log probabilities by temperature and renormalize
      // if temperature is high, logprobs will go towards zero
      // and the softmax outputs will be more diffuse. if temperature is
      // very low, the softmax outputs will be more peaky
      for (var q = 0, nq = logprobs.w.length; q < nq; q++) {
        logprobs.w[q] /= temperature;
      }
    }

    const probs = R.softmax(logprobs);

    if (samplei) {
      ix = R.Samplei(probs.w);
    } else {
      ix = R.Maxi(probs.w);
    }

    if (ix === 0) break; // END token predicted, break out
    if (s.length > max_chars_gen) {
      break;
    } // something is wrong

    var letter = indexToLetter[ix];
    s += letter;
  }
  return s;
}

/**
 * takes a model and a sentence and
 * calculates the loss. Also returns the Graph
 * object which can be used to do backprop
 */
function costfun(mod, sent) {
  const n = sent.length;
  const G = new R.Graph();
  let log2ppl = 0.0;
  let cost = 0.0;
  let prev = {};
  for (let i = -1; i < n; i++) {
    // start and end tokens are zeros
    const ix_source = i === -1 ? 0 : letterToIndex[sent[i]]; // first step: start with START token
    const ix_target = i === n - 1 ? 0 : letterToIndex[sent[i + 1]]; // last step: end with END token

    const lh = forwardIndex(G, mod, ix_source, prev);
    prev = lh;

    // set gradients into logprobabilities
    logprobs = lh.o; // interpret output as logprobs
    const probs = R.softmax(logprobs); // compute the softmax probabilities

    log2ppl += -Math.log2(probs.w[ix_target]); // accumulate base 2 log prob and do smoothing
    cost += -Math.log(probs.w[ix_target]);

    // write gradients into log probabilities
    logprobs.dw = probs.w;
    logprobs.dw[ix_target] -= 1;
  }
  var ppl = Math.pow(2, log2ppl / (n - 1));
  return {
    G,
    ppl,
    cost
  };
}

function median(values) {
  values.sort((a, b) => a - b);
  var half = Math.floor(values.length / 2);
  if (values.length % 2) {
    return values[half];
  }
  return (values[half - 1] + values[half]) / 2.0;
}

var perplexityList = [];
var tick_iter = 0;

function tick() {
  // sample sentence fromd data
  var sentix = R.randi(0, data_sents.length);
  var sent = data_sents[sentix];

  var t0 = +new Date(); // log start timestamp

  // evaluate cost function on a sentence
  var cost_struct = costfun(model, sent);

  // use built up graph to compute backprop (set .dw fields in mats)
  cost_struct.G.backward();
  // perform param update
  solver.step(model, learning_rate, regc, clipval);

  var t1 = +new Date();
  var tick_time = t1 - t0;

  perplexityList.push(cost_struct.ppl); // keep track of perplexity

  // evaluate now and then
  tick_iter += 1;

  if (tick_iter % 50 === 0) {
    var pred = '';
    debug('---------------------');
    // draw samples
    for (var q = 0; q < 5; q++) {
      pred = predictSentence(model, true, sample_softmax_temperature);
      debug('prediction', pred);
    }

    const epoch = (tick_iter / epoch_size).toFixed(2);
    const perplexity = cost_struct.ppl.toFixed(2);
    const ticktime = tick_time.toFixed(1) + 'ms';
    const medianPerplexity = median(perplexityList);
    perplexityList = [];

    debug('epoch=', epoch);
    debug('epoch_size', epoch_size);
    debug('perplexity', perplexity);
    debug('ticktime', ticktime);
    debug('medianPerplexity', medianPerplexity);
  }

  process.nextTick(tick);
}

// function gradCheck() { // eslint-disable-line
//   var mod = initModel();
//   var sent = '^test sentence$';
//   var cost_struct = costfun(mod, sent);
//   cost_struct.G.backward();
//   var eps = 0.000001;
//
//   for (var k in mod) {
//     if (mod.hasOwnProperty(k)) {
//       var m = mod[k]; // mat ref
//
//       for (var i = 0, n = m.w.length; i < n; i++) {
//         oldval = m.w[i];
//         m.w[i] = oldval + eps;
//         var c0 = costfun(mod, sent);
//         m.w[i] = oldval - eps;
//         var c1 = costfun(mod, sent);
//         m.w[i] = oldval;
//
//         var gnum = (c0.cost - c1.cost) / (2 * eps);
//         var ganal = m.dw[i];
//         var relerr = (gnum - ganal) / (Math.abs(gnum) + Math.abs(ganal));
//         if (relerr > 1e-1) {
//           debug(k + ': numeric: ' + gnum + ', analytic: ' + ganal + ', err: ' + relerr);
//         }
//       }
//     }
//   }
// }

// the following starts stuff.
reinit();
tick();
