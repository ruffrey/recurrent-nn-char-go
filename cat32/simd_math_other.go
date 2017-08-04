// +build !amd64,gc

package cat32

import "github.com/bjwbell/gensimd/simd"

func Addf32x4(x, y simd.F32x4) simd.F32x4 { return simd.AddF32x4(x, y) }
func Subf32x4(x, y simd.F32x4) simd.F32x4 { return simd.SubF32x4(x, y) }
func Mulf32x4(x, y simd.F32x4) simd.F32x4 { return simd.MulF32x4(x, y) }
func Divf32x4(x, y simd.F32x4) simd.F32x4 { return simd.Divf32x4(x, y) }
