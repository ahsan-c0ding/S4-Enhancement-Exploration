# Optimized Math.s for M4
# Math library for M4, single-precision float (f32)
# Every function takes its input in fa0 and return its result in fa0.

.global my_exp
.global my_sin
.global my_cos
.global my_tanh
.global my_sqrt
.global my_pow
.global my_log

# my_exp(x) (computes e^x):
# We use a range reduction + polynomial approximation strategy to compute e^x efficiently
# Working:
# 1) Reject bad inputs such as x < -88 return 0.0, x > 88 return + infinity
# 2) Rewrite x as: x + n*ln(2) + r, where n is a while number and r is a tiny leftover
#    Now: e^x = 2^n * e^r   (split into two easier pieces)
# 3) Approximate e^r with a small poly:
#    P(r) = 1 + r + (r^2)/2 + (t^3)/6 
#    This works well becasue r is small
# 4) Build 2^n by putting n directly into exponent bits of f32
#
# Rresult = P9r) * 2^n  
my_exp:
    # Bounds checking
    li t0, 0xC2B00000
    fmv.w.x ft0, t0
    flt.s t1, fa0, ft0
    beqz t1, .Lexp_check_pos
    fmv.w.x fa0, zero
    ret
.Lexp_check_pos:
    li t0, 0x42B00000
    fmv.w.x ft0, t0
    flt.s t1, ft0, fa0
    beqz t1, .Lexp_math
    li t0, 0x7F800000
    fmv.w.x fa0, t0
    ret

.Lexp_math:
    # n = round(x / ln2)
    li t0, 0x3FB8AA3B
    fmv.w.x ft0, t0
    fmul.s ft1, fa0, ft0
    fcvt.w.s a0, ft1, rne
    fcvt.s.w ft2, a0

    # r = x - n * ln2
    li t0, 0x3F317218
    fmv.w.x ft3, t0
    fmul.s ft3, ft2, ft3
    fsub.s ft4, fa0, ft3

    # P(r) = 1 + r(1 + r(0.5 + r(0.166667 + r*0.0416667)))
    li t0, 0x3D2AAAAB
    fmv.w.x ft5, t0
    fmul.s ft5, ft5, ft4
    li t0, 0x3E2AAAAB
    fmv.w.x ft6, t0
    fadd.s ft5, ft5, ft6
    fmul.s ft5, ft5, ft4
    li t0, 0x3F000000
    fmv.w.x ft6, t0
    fadd.s ft5, ft5, ft6
    fmul.s ft5, ft5, ft4
    li t0, 0x3F800000
    fmv.w.x ft6, t0
    fadd.s ft5, ft5, ft6
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6

    # 2^n using exponent injection
    addi a0, a0, 127          # bias exponent
    slli a0, a0, 23           # shift to exponent field
    fmv.w.x ft6, a0              # ft6 = 2^n exactly

    fmul.s fa0, ft5, ft6        # result = P(r) * 2^n
    ret

# my_sin(x) (computes sin(x))
#
# Working:
# 1) Fold x into (-π, π] by dividing by 2π, taking the integer part,
#    and subtracting it back.
#
# 2) Run the Taylor series term by term:
#    sin(x) = x - x³/6 + x⁵/120 - ...
#    Each new term = previous term * (-x²) / ((n+1)*(n+2))
#    Stop when the current term is smaller than machine epsilon (~1e-7).
#    This adapts to the input, the small x converges in very few steps.
my_sin:
    li t0, 0x40490FDB
    fmv.w.x ft0, t0
    fadd.s ft1, ft0, ft0

    fdiv.s ft2, fa0, ft1
    fcvt.w.s a0, ft2, rtz
    fcvt.s.w ft2, a0                

    fmul.s ft2, ft2, ft1
    fsub.s fa0, fa0, ft2

    flt.s t0, ft0, fa0
    beqz t0, .Lsin_check_neg
    fsub.s fa0, fa0, ft1

.Lsin_check_neg:
    fneg.s ft2, ft0
    flt.s t0, fa0, ft2
    beqz t0, .Lsin_tay_init
    fadd.s fa0, fa0, ft1

.Lsin_tay_init:
    fmv.w.x ft5,  zero
    fmv.s ft4, fa0 
    fmul.s ft3, fa0, fa0

    li t0, 0x33D6BF95
    fmv.w.x ft6, t0
    li t1, 1

.Lsin_loop:
    li t0, 50
    bgt t1, t0, .Lsin_done

    fadd.s  ft5,  ft5, ft4
    addi t2, t1, 1
    addi t3, t1, 2
    mul t0, t2, t3
    fcvt.s.w ft2, t0

    fmul.s ft4, ft4, ft3
    fneg.s ft4, ft4
    fdiv.s ft4, ft4, ft2

    fabs.s ft2,  ft4
    flt.s t0, ft2, ft6
    bnez t0, .Lsin_done

    addi t1, t1, 2
    j .Lsin_loop

.Lsin_done:
    fmv.s fa0, ft5
    ret

# my_cos(x) — computes cos(x)
#
# Same structure as my_sin with two differences:
# 1) Series starts at 1.0 (x^0 term) instead of x
# 2) Loop steps through even powers instead of odd
#    cos(x) = 1 - x²/2 + x⁴/24 - ...
my_cos:
    li t0, 0x40490FDB
    fmv.w.x ft0, t0
    fadd.s ft1, ft0, ft0

    fdiv.s ft2, fa0, ft1
    fcvt.w.s a0, ft2, rtz
    fcvt.s.w ft2, a0
    fmul.s ft2, ft2, ft1
    fsub.s fa0, fa0, ft2

    flt.s t0, ft0, fa0
    beqz t0, .Lcos_check_neg
    fsub.s fa0, fa0, ft1

.Lcos_check_neg:
    fneg.s ft2, ft0
    flt.s t0, fa0, ft2
    beqz t0, .Lcos_tay_init
    fadd.s fa0, fa0, ft1

.Lcos_tay_init:
    fmv.w.x ft5, zero
    li t0, 0x3F800000
    fmv.w.x ft4, t0
    fmul.s ft3, fa0, fa0

    li t0, 0x33D6BF95
    fmv.w.x ft6, t0
    li t1, 0

.Lcos_loop:
    li t0, 50
    bgt t1, t0, .Lcos_done

    fadd.s  ft5,  ft5, ft4
    addi t2, t1, 1
    addi t3, t1, 2
    mul t0, t2, t3
    fcvt.s.w ft2, t0

    fmul.s ft4, ft4, ft3
    fneg.s ft4, ft4
    fdiv.s ft4, ft4, ft2

    fabs.s ft2, ft4
    flt.s t0, ft2, ft6
    bnez t0, .Lcos_done

    addi t1, t1, 2
    j .Lcos_loop

.Lcos_done:
    fmv.s fa0, ft5
    ret


# my_tanh(x) (computes tanh(x))
#
# Working:
# 1) It uses rational approximation: tanh(x) ≈ numerator/denominator
#    where both are polynomials in x^2 exploiting tanh(-x) = -tanh(x)).
#    Numumerator= x * (a0 + a1*x^2 + a2*x^4 + a3*x^6)
#    Denominator= b0 + b1*x^2 + b2*x^4 + b3*x^
#
#    Coefficients were fit by least squares over [-7, 7] 
#
# Reason for clamp +-7:
#   tanh saturates near +-1 for large inputs, but the polynomial would diverge.
#   By clamping x to [-7, 7] keeps it in the fitted range. At x=7 the rational
#   gives ~0.99999, which is close enough to the true value of 0.9999983.
#   Clamping uses fmax/fmin — no branches, no pipeline stalls.
my_tanh:
    # Clamp x to [-7, 7] using two arithmetic ops (no branches)
    li t0, 0x40E00000
    fmv.w.x ft0, t0
    fneg.s ft4, ft0
    fmax.s ft1, fa0, ft4        # ft1 = max(x, -7.0)   (clamp low side)
    fmin.s ft1, ft1, ft0        # ft1 = min(ft1, 7.0)  (clamp high side) = xc

    fmul.s ft0, ft1, ft1        # ft0 = xc^2  (Horner variable, reuses ft0)

    # Numerator: xc * (a0 + xc^2*(a1 + xc^2*(a2 + xc^2*a3))) 
    li t0, 0x36622111
    fmv.w.x ft2, t0
    fmul.s ft2, ft2, ft0        # ft2 = a3*xc^2
    li t0, 0x3B0BACC8
    fmv.w.x ft5, t0
    fadd.s ft2, ft2, ft5        # ft2 = a2 + a3*xc^2
    fmul.s ft2, ft2, ft0        # ft2 = (a2 + a3*xc^2)*xc^2
    li t0, 0x3DF80989
    fmv.w.x ft5, t0
    fadd.s ft2, ft2, ft5        # ft2 = a1 + xc^2*(a2 + a3*xc^2)
    fmul.s ft2, ft2, ft0        # ft2 = xc^2*(a1 + ...)
    li t0, 0x3F7FFA71
    fmv.w.x ft5, t0
    fadd.s ft2, ft2, ft5        # ft2 = a0 + xc^2*(a1 + ...)   = poly(xc^2)
    fmul.s ft2, ft1, ft2        # ft2 = xc * poly(xc^2)         = Num

    #  Denominator: b0 + xc^2*(b1 + xc^2*(b2 + xc^2*b3)),  b0 = 1.0 
    li t0, 0x39063D79
    fmv.w.x ft3, t0
    fmul.s ft3, ft3, ft0        # ft3 = b3*xc^2
    li t0, 0x3CA68C6E
    fmv.w.x ft5, t0
    fadd.s ft3, ft3, ft5        # ft3 = b2 + b3*xc^2
    fmul.s ft3, ft3, ft0        # ft3 = (b2 + b3*xc^2)*xc^2
    li t0, 0x3EE89A1E
    fmv.w.x ft5, t0
    fadd.s ft3, ft3, ft5        # ft3 = b1 + xc^2*(b2 + b3*xc^2)
    fmul.s ft3, ft3, ft0        # ft3 = xc^2*(b1 + ...)
    li t0, 0x3F800000
    fmv.w.x ft5, t0
    fadd.s ft3, ft3, ft5        # ft3 = 1.0 + xc^2*(b1 + ...)   = Den

    fdiv.s fa0, ft2, ft3        # tanh(x) ~ Num / Den
    ret

# my_sqrt(x)
# Uses built-in optimized sqrt in RVV, replaces our Newton loop implementation.
my_sqrt:
    fsqrt.s fa0, fa0
    ret


# my_log(x) (computes ln(x))
#
# Working:
# 1) Return -inf for x ≤ 0 (ln is undefined there)
#
# 2) Every f32 can be written as  x = m * 2^e
#    m is between 1.0 and 2.0, and e an integer.
#    We read e and m directly from the f32 bit pattern.
#
# 3) ln(x) = e*ln(2) + ln(m)
#    e*ln(2) is just a multiply.
#    ln(m) is approximated over [1, 2) with a short polynomial:
#      let f = m - 1  (so f is in [0, 1))
#      ln(1+f) ≈ f*(1 - f/2 + f²/3 - f³/4)
my_log:
    fmv.w.x ft0, zero
    fle.s t1, fa0, ft0
    beqz t1, .Llog_math
    li t0, 0xFF800000       # return -inf for x <= 0
    fmv.w.x fa0, t0
    ret

.Llog_math:
    fmv.x.w a0, fa0
    # Extract integer exponent (e)
    srli a1, a0, 23
    andi a1, a1, 0xFF
    addi a1, a1, -127         # a1 = e
    
    # Isolate mantissa fraction (f) to reconstruct m in [1, 2)
    li t2, 0x007FFFFF
    and a0, a0, t2
    li t2, 0x3F800000       # exponent for 1.0
    or a0, a0, t2
    fmv.w.x ft0, a0              # ft0 = m = 1 + f
    
    # Let f_val = m - 1.0
    fmv.w.x ft1, t2              # 1.0
    fsub.s ft2, ft0, ft1        # ft2 = f_val
    
    # ln(1+f) ≈ f - f^2/2 + f^3/3 - f^4/4
    # Horner: f * (1.0 + f * (-0.5 + f * (0.33333 + f * -0.25)))
    li t0, 0xBE800000
    fmv.w.x ft3, t0
    fmul.s ft3, ft3, ft2        # -0.25 * f
    li t0, 0x3EAAAAAB
    fmv.w.x ft4, t0
    fadd.s ft3, ft3, ft4        # 0.333333 - 0.25f
    fmul.s ft3, ft3, ft2        # f * (...)
    li t0, 0xBF000000
    fmv.w.x ft4, t0
    fadd.s ft3, ft3, ft4        # -0.5 + ...
    fmul.s ft3, ft3, ft2        # f * (...)
    fadd.s ft3, ft3, ft1        # 1.0 + ...
    fmul.s ft3, ft3, ft2        # ft3 = ln(m)
    
    # Calculate e * ln2
    fcvt.s.w ft4, a1             # float(e)
    li t0, 0x3F317218       # 0.693147
    fmv.w.x ft5, t0
    fmul.s ft4, ft4, ft5        # e * ln(2)
    fadd.s fa0, ft3, ft4        # Result = e*ln2 + ln(m)
    ret

# my_pow(x, y) (computes x^y)
#
# Working:
# Uses the identity:  x^y = e^(y * ln(x))
# So we just call my_log then my_exp with y multiplied in between it.
my_pow:
    addi sp, sp, -16
    sw ra, 12(sp)
    fsw fs1, 8(sp)

    fmv.w.x ft0, zero
    fle.s t1, fa0, ft0
    beqz t1, .Lpow_math
    fmv.w.x fa0, zero            # return 0 if x <= 0 (simplified)
    j .Lpow_done

.Lpow_math:
    fmv.s fs1, fa1             # save y
    call my_log               # fa0 = ln(x)
    fmul.s fa0, fa0, fs1        # fa0 = y * ln(x)
    call my_exp               # fa0 = exp(y * ln(x))

.Lpow_done:
    flw fs1, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    ret


#  M4 VECTOR BLOCKS
#  Don't touch these yet. We'll use them later when we refactor 
#  nn.s to pass array pointers directly instead of looping floats.

.global v_my_tanh

# v_my_tanh
# a0 = input array pointer (*x)
# a1 = output array pointer (*y)
# a2 = number of elements (N)
v_my_tanh:
.L_v_tanh_loop:
    vsetvli t0, a2, e32, m8, ta, ma    # Configure vector lengths for 32-bit floats
    vle32.v v8, (a0)                   # Load vector of x

    # Calculate x^2
    vfmul.vv v16, v8, v8

    # Numerator: x * (1.0 + 0.10001*x^2)
    li t1, 0x3DCCCCD0             # 0.10001
    fmv.w.x ft0, t1
    vfmv.v.f v24, ft0
    vfmacc.vv v24, v16, v24            # v24 = 0.10001 * x^2
    li t1, 0x3F800000             # 1.0
    fmv.w.x ft1, t1
    vfadd.vf v24, v24, ft1             # v24 = 1.0 + 0.10001*x^2
    vfmul.vv v24, v8, v24              # Numerator complete

    # Denominator: 1.0 + x^2 * (0.43301 + 0.009999*x^2)
    li t1, 0x3C23D69A             # 0.009999
    fmv.w.x ft2, t1
    vfmv.v.f v0, ft2
    vfmul.vv v0, v16, v0               # 0.009999 * x^2
    li t1, 0x3EDDA740             # 0.43301
    fmv.w.x ft3, t1
    vfadd.vf v0, v0, ft3               # + 0.43301
    vfmul.vv v0, v16, v0               # x^2 * (...)
    vfadd.vf v0, v0, ft1               # Denominator complete

    # Divide and store
    vfdiv.vv v8, v24, v0               # v8 = Num / Den
    vse32.v v8, (a1)                   # Store result

    # Bump pointers
    slli t1, t0, 2                  # t1 = elements processed * 4 bytes
    add a0, a0, t1                 # advance input pointer
    add a1, a1, t1                 # advance output pointer
    sub a2, a2, t0                 # subtract elements processed
    bnez a2, .L_v_tanh_loop         # loop if N > 0
    ret
