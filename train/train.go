package train

// model parameters
const hiddenSizes = [20, 20]; // list of sizes of hidden layers
const letterSize = 5; // size of letter embeddings

// optimization
const regc = 0.000001; // L2 regularization strength
const learning_rate = 0.01; // learning rate
const clipval = 3; // clip gradients at this value

/* */

// prediction params
const sample_softmax_temperature = 1.0; // how peaky model predictions should be
const max_chars_gen = 100; // max length of generated sentences

// constious global const inits
const epoch_size = -1;
const input_size = -1;
const output_size = -1;
const letterToIndex = {};
const indexToLetter = {};
const vocab = [];
const data_sents = [];
const solver = new R.Solver(); // should be class because it needs memory for step caches

// unsure about these former accidental (?) globals
var logprobs;
var oldval;

const model = {};
