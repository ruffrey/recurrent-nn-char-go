// +build amd64,gc

package cat32

import "github.com/bjwbell/gensimd/simd"

//go:generate $GOPATH/bin/gensimd -fn "Addf32x4, Subf32x4, Mulf32x4, Divf32x4" -outfn "Addf32x4, Subf32x4, Mulf32x4, Divf32x4" -f "simd_math_amd64.go" -o "simd_math_amd64.s"

func Addf32x4(x, y simd.F32x4) simd.F32x4
func Subf32x4(x, y simd.F32x4) simd.F32x4
func Mulf32x4(x, y simd.F32x4) simd.F32x4
func Divf32x4(x, y simd.F32x4) simd.F32x4
