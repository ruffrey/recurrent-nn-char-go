// +build amd64,gc

package cat32

import "github.com/bjwbell/gensimd/simd"

func AddF32x4(x, y simd.F32x4) simd.F32x4
func SubF32x4(x, y simd.F32x4) simd.F32x4
func MulF32x4(x, y simd.F32x4) simd.F32x4
func DivF32x4(x, y simd.F32x4) simd.F32x4
