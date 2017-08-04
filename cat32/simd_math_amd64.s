// +build amd64 !noasm !appengine

#include "textflag.h"

TEXT ·AddF32x4(SB),$24-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)
block0:
        MOVUPS       y+16(FP), X15
        MOVUPS       x+0(FP), X14
        ADDPS        X15, X14
        MOVUPS       X14, ret0+32(FP)
        RET

TEXT ·SubF32x4(SB),$24-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)
block0:
        MOVUPS       y+16(FP), X15
        MOVUPS       x+0(FP), X14
        SUBPS        X15, X14
        MOVUPS       X14, ret0+32(FP)
        RET

TEXT ·MulF32x4(SB),$24-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)
block0:
        MOVUPS       y+16(FP), X15
        MOVUPS       x+0(FP), X14
        MULPS        X15, X14
        MOVUPS       X14, ret0+32(FP)
        RET

TEXT ·DivF32x4(SB),$24-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)
block0:
        MOVUPS       y+16(FP), X15
        MOVUPS       x+0(FP), X14
        DIVPS        X15, X14
        MOVUPS       X14, ret0+32(FP)
        RET
