# TODO

x perplexity does not change between sessions
- no output comes from predictions
    - the pointers do not appear to work even close to how they work in JS
    - during backprop functions, need to be able to assign back the updated values to the matrix.
    - change the math functions to accept the model key pointing to the Mat, instead of the Mat.
        - have the model on hand and pass it into the backprop funcs.
        - this will make backprop work.
    - or, pass a function or something to be used during the backprop
