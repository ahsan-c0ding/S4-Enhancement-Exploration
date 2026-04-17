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
    li   t0, 0
    fcvt.s.w ft1, t0

    flt.s t1, fs0, ft1
    beq  t1, zero, exp_positive
    
    fneg.s fa0, fs0
    call my_exp

    li   t0, 1
    fcvt.s.w ft1, t0
    fdiv.s fa0, ft1, fa0
    j    exp_exit

exp_positive:
    li   t0, 1
    fcvt.s.w ft1, t0
    fmv.s ft2, ft1
    li   t0, 1

exp_loop:
    fmul.s ft2, ft2, fs0
    fcvt.s.w ft4, t0
    fdiv.s ft2, ft2, ft4

    fadd.s ft1, ft1, ft2

    li   t1, 0x33D6BF95
    fmv.w.x ft5, t1
    fabs.s ft6, ft2
    flt.s t2, ft6, ft5

    addi t0, t0, 1
    li   t3, 50
    bge  t0, t3, exp_done
    beq  t2, zero, exp_loop

exp_done:
    fmv.s fa0, ft1

exp_exit:
    flw  fs0, 8(sp)
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret

# ______________________________________________________________
# my_sin(x)
# Computes sin(x) using Taylor series:
#   sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
# Approach:
#   Start with first term = x
#   Each next term is derived from the previous one using:
#     term = -term * x^2 / ((2n)(2n+1)) by doing this we are able 
#     avoid computing powers and factorials separately.
#   Next we accumulate terms into the result in each iteration.
# ______________________________________________________________
my_sin:
    fmv.s ft0, fa0

    li t0, 0
    fcvt.s.w ft1, t0
    fmv.s ft2, ft0

    li t0, 1
    fmul.s ft3, ft0, ft0

sin_loop:
    fadd.s ft1, ft1, ft2

    fneg.s ft2, ft2
    fmul.s ft2, ft2, ft3

    slli t1, t0, 1
    addi t2, t1, 1

    fcvt.s.w ft4, t1
    fcvt.s.w ft5, t2
    fmul.s ft4, ft4, ft5

    fdiv.s ft2, ft2, ft4

    addi t0, t0, 1

    li t3, 10
    bge t0, t3, sin_done

    j sin_loop

sin_done:
    fmv.s fa0, ft1
    ret

# ______________________________________________________________
# my_cos(x)
# Computes cos(x) using Taylor series:
#   cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...
# Approach:
#   Start with term = 1
#   Each next term is computed using:
#     term = -term * x^2 / ((2k+1)(2k+2))
#   We get this pattern by expanding factorial terms incrementally.
#   Next we add each term to the result as we go.
# ______________________________________________________________
my_cos:
    fmv.s ft0, fa0

    li t0, 1
    fcvt.s.w ft2, t0
 
    li t0, 0
    fcvt.s.w ft1, t0

    fmul.s ft3, ft0, ft0

    li t0, 0

cos_loop:
    fadd.s ft1, ft1, ft2

    fneg.s ft2, ft2
    fmul.s ft2, ft2, ft3

    # denominator = (2k+1)(2k+2)
    slli t1, t0, 1
    addi t2, t1, 1
    addi t3, t1, 2

    fcvt.s.w ft4, t2
    fcvt.s.w ft5, t3
    fmul.s ft4, ft4, ft5

    fdiv.s ft2, ft2, ft4

    addi t0, t0, 1
    li t4, 20
    bge t0, t4, cos_done

    j cos_loop

cos_done:
    fmv.s fa0, ft1
    ret

# ____________________________________________________________
# my_tanh(x)
# Computes tanh(x) using identity:
#   tanh(x) = (e^x - e^-x) / (e^x + e^-x)
# Approach:
#   First handle large values of x such that for |x| >= 10, 
#   tanh(x) ≈ ±1 (it saturates), so we directly return +1 or -1 
#   depending on sign.
#   Otherwise, we compute e^x using my_exp, compute e^-x using my_exp
#   and apply formula above
#   We store intermediate values (like e^x) to avoid recomputation.
# _____________________________________________________________
my_tanh:
    addi sp, sp, -32
    sw   ra, 28(sp)
    fsw  fs0, 24(sp)
    fsw  fs1, 20(sp)

    fmv.s fs0, fa0

    li   t0, 0x41200000
    fmv.w.x ft0, t0
    fabs.s ft1, fs0
    flt.s t1, ft1, ft0
    bne  t1, zero, tanh_math

    li   t0, 1
    fcvt.s.w fa0, t0

    li   t0, 0
    fcvt.s.w ft2, t0

    flt.s t1, fs0, ft2
    beq  t1, zero, tanh_exit
    fneg.s fa0, fa0
    j    tanh_exit

tanh_math:
    # Compute e^x
    fmv.s fa0, fs0
    call  my_exp
    fmv.s fs1, fa0

    # Compute e^-x
    fneg.s fa0, fs0
    call  my_exp

    # (e^x - e^-x) / (e^x + e^-x)
    fsub.s ft1, fs1, fa0
    fadd.s ft2, fs1, fa0
    fdiv.s fa0, ft1, ft2

tanh_exit:
    lw   ra, 28(sp)
    flw  fs0, 24(sp)
    flw  fs1, 20(sp)
    addi sp, sp, 32
    ret

# ___________________________________________________________
# my_sqrt(x)
# Computes sqrt(x) using Newton's Method:
#   guess = (guess + x/guess) / 2
# Approach:
#   If x = 0 we return 0 immediately.
#   Choose an initial guess such that if x >= 1 we start with x
#   else if x < 1 we start with 1 this id for stability
#   Iteratively improve guess using Newton update formula.
#   We stop when difference between consecutive guesses
#   becomes very small (< 1e-7).
# ______________________________________________________________
my_sqrt:
    # Handle x = 0
    li t0, 0
    fcvt.s.w ft0, t0
    feq.s t1, fa0, ft0
    bne t1, zero, sqrt_zero

    # Initial guess
    fmv.s ft0, fa0

    # For x < 1 start with guess = 1
    li t0, 1
    fcvt.s.w ft1, t0
    flt.s t1, fa0, ft1
    beq t1, zero, sqrt_loop
    fmv.s ft0, ft1

sqrt_loop:
    # Newton iteration guess = (guess + x/guess) / 2
    fdiv.s ft2, fa0, ft0
    fadd.s ft2, ft0, ft2
    li t0, 2
    fcvt.s.w ft3, t0
    fdiv.s ft2, ft2, ft3

    # Check convergence
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

# ____________________________________________________________
# my_pow(x, y)
# Computes x^y using identity:
#   x^y = e^(y * ln(x))
# Approach:
#   First check if x <= 0 this implementation does not handle
#   those cases, so it simply returns 0.
#   Otherwise we compute ln(x) using my_log, multiply by y,
#   and then pass result to my_exp to compute final value
#   We are bale to store y safely because function calls
#   overwrite registers.
# ____________________________________________________________
my_pow:
    addi sp, sp, -16
    sw   ra, 12(sp)
    fsw  fs1, 8(sp)

    # If x <= 0, return 0
    li   t0, 0
    fcvt.s.w ft0, t0
    fle.s t1, fa0, ft0
    bne  t1, zero, pow_zero_exit

    fmv.s fs1, fa1

    # Compute ln(x)
    call my_log

    fmul.s fa0, fa0, fs1
    call my_exp
    j    pow_finish

pow_zero_exit:
    li   t0, 0
    fcvt.s.w fa0, t0

pow_finish:
    flw  fs1, 8(sp)
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret

# ___________________________________________________________
# my_log(x)
# Computes ln(x) using Newton's Method on equation:
#   e^y = x
# Approach:
#   We want to solve for y such that e^y = x.
#   So, newton update formula becomes:
#     y = y - (e^y - x) / e^y
#   Start with initial guess:
#     y = x - 1  (simple approximation)
#   Then we iteratively refine y using above formula.
#   The stopping condition:
#     |e^y - x| < 1e-7 OR max iterations reached
#   If x <= 0 then log is undefined and return -infinity.
# _____________________________________________________________
my_log:
    addi sp, sp, -32
    sw   ra, 28(sp)
    fsw  fs0, 24(sp)
    fsw  fs1, 20(sp)
    sw   s0, 16(sp)

    # Check x > 0
    li   t0, 0
    fcvt.s.w ft0, t0
    fle.s t1, fa0, ft0
    beq  t1, zero, log_positive

    li   t0, 0xFF800000
    fmv.w.x fa0, t0
    j    log_exit

log_positive:
    fmv.s fs0, fa0
    # Initial guess: y = x - 1
    li   t0, 1
    fcvt.s.w ft0, t0
    fsub.s fs1, fs0, ft0
    li   s0, 20          # Max 20 iterations

log_loop:
    # Compute e^y
    fmv.s fa0, fs1
    call  my_exp
    fmv.s ft2, fa0

    # diff = e^y - x
    fsub.s ft3, ft2, fs0

    # y = y - (diff / e^y)
    fdiv.s ft4, ft3, ft2
    fsub.s fs1, fs1, ft4

    # Convergence check: |diff| < 1e-7
    fabs.s ft5, ft3
    li   t2, 0x33D6BF95
    fmv.w.x ft6, t2
    flt.s t3, ft5, ft6

    addi s0, s0, -1
    beq  s0, zero, log_done
    beq  t3, zero, log_loop

log_done:
    fmv.s fa0, fs1

log_exit:
    lw   ra, 28(sp)
    flw  fs0, 24(sp)
    flw  fs1, 20(sp)
    lw   s0, 16(sp)
    addi sp, sp, 32
    ret
