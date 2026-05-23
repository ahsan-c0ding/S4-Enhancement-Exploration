# Optimized Math.s for M4

.global my_exp
.global my_sin
.global my_cos
.global my_tanh
.global my_sqrt
.global my_pow
.global my_log

# _____________________________________________________________________________
# my_exp(x)
# First we apply bounds checking. If x is less than -88.0, the result underflows 
# single-precision float limits and returns 0.0. If x exceeds 88.0, it overflows 
# to positive infinity (+inf). 
#
# For valid inputs, we perform a range reduction to decompose x into:
#   x = n * ln(2) + r, where n is an integer and |r| <= ln(2)/2 = 0.34657
#
# This allows us to compute the exponential function using power-of-two scaling:
#   e^x = e^(n*ln(2) + r) = (e^ln(2))^n * e^r = 2^n * e^r
#
# Horner's polynomial P(r) = 1 + r + 0.5r^2 + 0.166667r^3 + 0.0416667r^4
#
# We approximate the remainder e^r on the tiny interval [-0.34657, 0.34657] using 
# a 4th-degree Horner polynomial, P(r). Then, we reconstruct 2^n instantly by shifting 
# the integer n directly into the exponent bitfield, avoiding adding a loop.
# _____________________________________________________________________________
my_exp:
    # Bounds checking
    li      t0, 0xC2B00000       # -88.0
    fmv.w.x ft0, t0
    flt.s   t1, fa0, ft0
    beqz    t1, .Lexp_check_pos
    fmv.w.x fa0, zero            # return 0.0 if x < -88
    ret
.Lexp_check_pos:
    li      t0, 0x42B00000       # 88.0
    fmv.w.x ft0, t0
    flt.s   t1, ft0, fa0
    beqz    t1, .Lexp_math
    li      t0, 0x7F800000       # +inf
    fmv.w.x fa0, t0              # return +inf if x > 88
    ret

.Lexp_math:
    # n = round(x / ln2)
    li      t0, 0x3FB8AA3B       # 1.442695 (1/ln2)
    fmv.w.x ft0, t0
    fmul.s  ft1, fa0, ft0
    fcvt.w.s a0, ft1, rne        # Integer n
    fcvt.s.w ft2, a0             # Float n

    # r = x - n * ln2
    li      t0, 0x3F317218       # 0.693147 (ln2)
    fmv.w.x ft3, t0
    fmul.s  ft3, ft2, ft3        # n * ln2
    fsub.s  ft4, fa0, ft3        # r

    # P(r) = 1 + r(1 + r(0.5 + r(0.166667 + r*0.0416667)))
    li      t0, 0x3D2AAAAB       # 0.0416667
    fmv.w.x ft5, t0
    fmul.s  ft5, ft5, ft4        # r * 0.0416667
    li      t0, 0x3E2AAAAB       # 0.166667
    fmv.w.x ft6, t0
    fadd.s  ft5, ft5, ft6        # 0.166667 + r*0.0416667
    fmul.s  ft5, ft5, ft4        # r * (...)
    li      t0, 0x3F000000       # 0.5
    fmv.w.x ft6, t0
    fadd.s  ft5, ft5, ft6        # 0.5 + ...
    fmul.s  ft5, ft5, ft4        # r * (...)
    li      t0, 0x3F800000       # 1.0
    fmv.w.x ft6, t0
    fadd.s  ft5, ft5, ft6        # 1.0 + ...
    fmul.s  ft5, ft5, ft4        # r * (...)
    fadd.s  ft5, ft5, ft6        # P(r) = 1.0 + ...

    # 2^n using exponent injection
    addi    a0, a0, 127          # bias exponent
    slli    a0, a0, 23           # shift to exponent field
    fmv.w.x ft6, a0              # ft6 = 2^n exactly

    fmul.s  fa0, ft5, ft6        # result = P(r) * 2^n
    ret


# _____________________________________________________________________________
# my_sin(x)
# First we reduce x into range (-π, π] because Taylor series only works well
# for small values. We do this by removing multiples of 2π and then adjusting
# if it goes outside ±π.
#
# After this we use Taylor series:
#   sin(x) = x - x^3/3! + x^5/5! - ...
#
# Instead of computing powers and factorials again and again, we reuse the
# previous term:
#   term = -term * x^2 / ((i+1)(i+2))
#
# Loop runs until term becomes very small (1e-7) or max iterations have been reached.
# _____________________________________________________________________________

    .section .text
    .globl my_sin
    .globl my_cos
    .align 2

# _________________________________________________________________________________
# float my_sin(float x)
# Register map
# ┌────────┬──────────────────────────────────────────────────────────────────┐
# │  fa0   │ input x, holds reduced x throughout then overwritten with result │
# │  ft0   │ pi  (3.1415927f)                                                 │
# │  ft1   │ two_pi  (2 * pi)                                                 │
# │  ft2   │ scratch float  (quotient-float, -pi, divisor-float, |term|)      │
# │  ft3   │ x_sq (x*x after reduction it is constant through the Taylor loop)│
# │  ft4   │ term  (running Taylor term, sin: starts as x)                    │
# │  ft5   │ result  (Taylor accumulator this is returned in fa0)             │
# │  ft6   │ epsilon  (1e-7f convergence threshold)                           │
# │  a0    │ integer quotient from fcvt.w.s / divisor integer                 │
# │  t0    │ integer scratch  (bit-patterns, comparison results, divisor)     │
# │  t1    │ loop counter i  (sin: starts at 1 then step 2)                   │
# │  t2    │ i + 1  (first factor of denominator term)                        │
# │  t3    │ i + 2  (second factor of denominator term)                       │
# └────────┴──────────────────────────────────────────────────────────────────┘
# __________________________________________________________________________________
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

    #    Taylor series initialisation
    #    sin(x) = x  –  x³/3!  +  x⁵/5!  –...
    #    result = 0;  term = x;  x_sq = x*x
    #    Loop: i = 1, 3, 5...  (i += 2 each step)
.Lsin_tay_init:
    fmv.w.x ft5,  zero
    fmv.s ft4, fa0 
    fmul.s ft3, fa0, fa0

    li t0, 0x33D6BF95
    fmv.w.x ft6, t0

    li t1, 1

    #  The main Taylor loop
.Lsin_loop:
    li t0, 50
    bgt t1, t0, .Lsin_done

    fadd.s  ft5,  ft5, ft4

    # denominator: (i+1) * (i+2)
    addi t2, t1, 1
    addi t3, t1, 2
    mul t0, t2, t3
    fcvt.s.w ft2, t0

    # term = -term * x_sq / ((i+1)*(i+2))
    fmul.s ft4, ft4, ft3
    fneg.s ft4, ft4
    fdiv.s ft4, ft4, ft2

    # Convergence check: |term| < epsilon
    fabs.s ft2,  ft4
    flt.s t0, ft2, ft6
    bnez t0, .Lsin_done

    addi t1, t1, 2
    j .Lsin_loop

.Lsin_done:
    fmv.s fa0, ft5
    ret

# _______________________________________________________________________________
# my_cos(x)
# For cos same idea is used as sin, first reduce x into (-π, π] for accuracy.
# Then use Taylor series:
#   cos(x) = 1 - x^2/2! + x^4/4! - ...
#
# Difference from sin is that it starts from 1 instead of x and loop 
# starts at i = 0
#
# The term update has same idea:
#   term = -term * x^2 / ((i+1)(i+2))
# and stop when term is very small or iterations exceed limit.
# _____________________________________________________________________________

# ______________________________________________________________________________
# float my_cos(float x)
# Register map  (similar to my_sin except ft4 and t1 init)
# ┌────────┬──────────────────────────────────────────────────────────────────┐
# │  ft4   │ term  (cos: starts at 1.0f, NOT x)                               │
# │  t1    │ loop counter i  (cos: starts at 0, step 2)                       │
# └────────┴──────────────────────────────────────────────────────────────────┘
# ______________________________________________________________________________
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

# _________________________________________________________________________________
# my_tanh(x)
# First we perform boundary checks. If the absolute value of x is greater than 
# 4.0, tanh(x) asymptotically approaches ±1.0 within single-precision limits. 
# We clip the value early to bypass core calculations entirely for large inputs.
#
# For values within [-4.0, 4.0], we use a [3/3] Padé Approximant:
#   tanh(x) = x * (1 + 0.10001*x^2) / (1 + 0.43301*x^2 + 0.009999*x^4)
#
# Instead of calculating expensive exponential terms or relying on dynamic loops 
# like a Taylor series, this method uses a rational function of polynomials.
# It minimizes the worst-case error uniformly across the target interval.
# _________________________________________________________________________________
my_tanh:
    # Check boundaries: if |x| > 4.0, tanh(x) is basically +-1.0
    li      t0, 0x40800000       # 4.0 in hex
    fmv.w.x ft0, t0
    fabs.s  ft1, fa0
    flt.s   t1, ft0, ft1         # t1 = 1 if |x| > 4.0
    beqz    t1, .Ltanh_math

    # Return sign(x) * 1.0
    li      t0, 0x3F800000       # 1.0
    fmv.w.x ft0, t0
    fsgnj.s fa0, ft0, fa0        # inject sign of fa0(x) into t0(1.0)
    ret

.Ltanh_math:
    fmul.s  ft0, fa0, fa0        # ft0 = x^2
    
    # Numerator: x * (1.0 + 0.10001 * x^2)
    li      t0, 0x3DCCCCD0       # 0.10001000
    fmv.w.x ft1, t0
    fmul.s  ft1, ft1, ft0        # 0.10001 * x^2
    li      t0, 0x3F800000       # 1.0
    fmv.w.x ft2, t0
    fadd.s  ft1, ft1, ft2        # 1.0 + 0.10001*x^2
    fmul.s  ft3, fa0, ft1        # Num = x * (1 + 0.10001*x^2)

    # Denominator: 1.0 + x^2 * (0.43301 + 0.009999 * x^2)
    li      t0, 0x3C23D69A       # 0.009999
    fmv.w.x ft1, t0
    fmul.s  ft1, ft1, ft0        # 0.009999 * x^2
    li      t0, 0x3EDDA740       # 0.43301
    fmv.w.x ft4, t0
    fadd.s  ft1, ft1, ft4        # 0.43301 + 0.009999*x^2
    fmul.s  ft1, ft1, ft0        # x^2 * (0.43301 + 0.009999*x^2)
    fadd.s  ft4, ft2, ft1        # Den = 1.0 + ...

    fdiv.s  fa0, ft3, ft4        # tanh(x) = Num / Den
    ret

# _________________________________________________________________
# my_sqrt(x)
# Uses built-in optimized sqrt in RVV, replaces our Newton loop implementation.
# _________________________________________________________________
my_sqrt:
    fsqrt.s fa0, fa0
    ret


# ___________________________________________________________
# my_pow(x, y) uses:
#   x^y = e^(y * ln(x))
# If x <= 0 then return 0 (not handled here).
#
# Steps:
# 1 Compute ln(x) using my_log
# 2 Multiply with y
# 3 Pass result to my_exp
# ___________________________________________________________
my_pow:
    addi sp, sp, -16
    sw ra, 12(sp)
    fsw fs1, 8(sp)

    li t0, 0
    fcvt.s.w ft0, t0
    fle.s t1, fa0, ft0
    bne t1, zero, pow_zero_exit

    fmv.s fs1, fa1

    call my_log

    fmul.s fa0, fa0, fs1
    call my_exp
    j pow_finish

pow_zero_exit:
    li t0, 0
    fcvt.s.w fa0, t0

pow_finish:
    flw fs1, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
    ret

# ____________________________________________________________
# my_log(x) solves:
#   e^y = x
# to find ln(x).
# If x <= 0 then return -inf.
#
# Start with guess:
#   y = x - 1
# Then we use Newton update:
#   y = y - (e^y - x) / e^y
# it stops when error is very small (1e-7) or iterations finish.
# _____________________________________________________________
my_log:
    addi sp, sp, -32
    sw ra, 28(sp)
    fsw fs0, 24(sp)
    fsw fs1, 20(sp)
    sw s0, 16(sp)

    li t0, 0
    fcvt.s.w ft0, t0
    fle.s t1, fa0, ft0
    beq t1, zero, log_positive

    li t0, 0xFF800000
    fmv.w.x fa0, t0
    j log_exit

log_positive:
    fmv.s fs0, fa0
    li t0, 1
    fcvt.s.w ft0, t0
    fsub.s fs1, fs0, ft0
    li s0, 20

log_loop:
    # Compute e^y
    fmv.s fa0, fs1
    call my_exp
    fmv.s ft2, fa0

    fsub.s ft3, ft2, fs0

    fdiv.s ft4, ft3, ft2
    fsub.s fs1, fs1, ft4

    fabs.s ft5, ft3
    li t2, 0x33D6BF95
    fmv.w.x ft6, t2
    flt.s t3, ft5, ft6

    addi s0, s0, -1
    beq s0, zero, log_done
    beq t3, zero, log_loop

log_done:
    fmv.s fa0, fs1

log_exit:
    lw ra, 28(sp)
    flw fs0, 24(sp)
    flw fs1, 20(sp)
    lw s0, 16(sp)
    addi sp, sp, 32
    ret
