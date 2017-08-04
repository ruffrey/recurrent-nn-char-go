// +build !amd64,gc

package cat32

import "github.com/bjwbell/gensimd/simd"

func AddF32x4(x, y simd.F32x4) simd.F32x4 { return simd.AddF32x4(x, y) }
func SubF32x4(x, y simd.F32x4) simd.F32x4 { return simd.SubF32x4(x, y) }
func MulF32x4(x, y simd.F32x4) simd.F32x4 { return simd.MulF32x4(x, y) }
func DivF32x4(x, y simd.F32x4) simd.F32x4 { return simd.DivF32x4(x, y) }
