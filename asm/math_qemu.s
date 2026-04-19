# Math.c implementation in RISC-V assembly

.global my_exp
.global my_sin
.global my_cos
.global my_tanh
.global my_sqrt
.global my_pow
.global my_log

# ____________________________________________________________
# my_exp(x)
# This function computes e^x using the Taylor series:
#       e^x = 1 + x + x^2/2! + x^3/3! + ...
# Approach:
#   If x is negative, we avoid by directly computing the series
#   as it converges slower for neg values.
#   Instead we use:
#       e^x = 1 / e^(-x)
#   Now we recursively compute e^(-x) and take reciprocal of it.
#   For positive x:
#   We iteratively build each term using the relation:
#       term = term * x / i
#   By doing this we are able to avoid recomputation of power 
#   and factorials from scratch.
#   We keep adding terms to the result until the term becomes
#   very small (|term| < 1e-7), or we hit a safe iteration cap
# ________________________________________________________________

my_exp:
    addi sp, sp, -16
    sw ra, 12(sp)
    fsw fs0, 8(sp)

    fmv.s fs0, fa0
    li t0, 0
    fcvt.s.w ft1, t0

    flt.s t1, fs0, ft1
    beq  t1, zero, exp_positive
    # Handle negative: compute 1 / exp(-x)
    fneg.s fa0, fs0
    call my_exp

    li t0, 1
    fcvt.s.w ft1, t0
    fdiv.s fa0, ft1, fa0
    j exp_exit

exp_positive:
    li t0, 1
    fcvt.s.w ft1, t0
    fmv.s ft2, ft1
    li t0, 1

exp_loop:
    # term = term * x / i
    fmul.s ft2, ft2, fs0
    fcvt.s.w ft4, t0
    fdiv.s ft2, ft2, ft4

    fadd.s ft1, ft1, ft2

    # Convergence check |term| < 1e-7
    li t1, 0x33D6BF95
    fmv.w.x ft5, t1
    fabs.s ft6, ft2
    flt.s t2, ft6, ft5

    addi t0, t0, 1
    li t3, 50
    bge t0, t3, exp_done
    beq t2, zero, exp_loop

exp_done:
    fmv.s fa0, ft1

exp_exit:
    flw fs0, 8(sp)
    lw ra, 12(sp)
    addi sp, sp, 16
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

# ____________________________________________________________
# my_tanh(x)
# using identity:
#   tanh(x) = (e^x - e^-x) / (e^x + e^-x)
# If |x| is large (>= 10) the tanh basically becomes ±1
# so we directly return 1 or -1 to save time.
#
# Otherwise we compute e^x and e^-x using my_exp
# Then apply the formula above to get the result.
# ____________________________________________________________
my_tanh:
    addi sp, sp, -32
    sw ra, 28(sp)
    fsw fs0, 24(sp)
    fsw fs1, 20(sp)

    fmv.s fs0, fa0

    li t0, 0x41200000
    fmv.w.x ft0, t0
    fabs.s ft1, fs0
    flt.s t1, ft1, ft0
    bne t1, zero, tanh_math

    li t0, 1
    fcvt.s.w fa0, t0

    li t0, 0
    fcvt.s.w ft2, t0

    flt.s t1, fs0, ft2
    beq t1, zero, tanh_exit
    fneg.s fa0, fa0
    j tanh_exit

tanh_math:
    # Compute e^x
    fmv.s fa0, fs0
    call my_exp
    fmv.s fs1, fa0

    # Compute e^-x
    fneg.s fa0, fs0
    call my_exp

    # (e^x - e^-x) / (e^x + e^-x)
    fsub.s ft1, fs1, fa0
    fadd.s ft2, fs1, fa0
    fdiv.s fa0, ft1, ft2

tanh_exit:
    lw ra, 28(sp)
    flw fs0, 24(sp)
    flw fs1, 20(sp)
    addi sp, sp, 32
    ret

# ____________________________________________________________
# my_sqrt(x) uses Newton’s method:
#   guess = (guess + x/guess) / 2
# If x = 0 then return 0.
#
# Initial guess if x >= 1 then start with x otherwise if x < 1 
# then start with 1
# Keep updating guess until difference becomes very small (1e-7).
# _____________________________________________________________-
my_sqrt:
    li t0, 0
    fcvt.s.w ft0, t0
    feq.s t1, fa0, ft0
    bne t1, zero, sqrt_zero

    fmv.s ft0, fa0

    li t0, 1
    fcvt.s.w ft1, t0
    flt.s t1, fa0, ft1
    beq t1, zero, sqrt_loop
    fmv.s ft0, ft1

sqrt_loop:
    # Newton iteration: guess = (guess + x/guess) / 2
    fdiv.s ft2, fa0, ft0
    fadd.s ft2, ft0, ft2
    li t0, 2
    fcvt.s.w ft3, t0
    fdiv.s ft2, ft2, ft3

    fsub.s ft3, ft2, ft0
    fabs.s ft3, ft3
    li t0, 0x33D6BF95
    fmv.w.x ft4, t0
    flt.s t1, ft3, ft4

    fmv.s ft0, ft2

    beq t1, zero, sqrt_loop

sqrt_done:
    fmv.s fa0, ft0
    ret

sqrt_zero:
    fmv.s fa0, ft0
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
