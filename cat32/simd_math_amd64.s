// +build amd64 !noasm !appengine

#include "textflag.h"

TEXT ·Addf32x4(SB),$8-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)

TEXT ·Subf32x4(SB),$8-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)

TEXT ·Mulf32x4(SB),$8-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)

TEXT ·Divf32x4(SB),$8-48
        MOVQ         $0, ret0+32(FP)
        MOVQ         $0, ret0+40(FP)

