package mat8

import (
	"math/rand"
)

/*
Randf makes random numbers
*/
func Randf(a int8, b int8) int8 {
	ia := int(a)
	ib := int(b)
	return int8(rand.Int()*(ib-ia) + ia)
}
