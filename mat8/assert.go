package mat8

/*
Assert ensures our code is not breaking down and halts the program.
*/
func Assert(assertion bool, msg string) {
	if !assertion {
		panic(msg)
	}
}
