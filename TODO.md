# TODO

x perplexity does not change between sessions
- no output comes from predictions
    - the pointers do not appear to work even close to how they work in JS
    - during backprop functions, need to be able to assign back the updated values to the matrix.
    - change the math functions to accept the model key pointing to the Mat, instead of the Mat.
        - have the model on hand and pass it into the backprop funcs.
        - this will make backprop work.
    - oddly, this works:
        - https://play.golang.org/p/eb2UbhYXzJ
        - It is changing an array, by passing around a matrix and modifying it
    - It seems the Backward / Backpropagation does not work. it's like the pointers no longer
        point to the current model.
