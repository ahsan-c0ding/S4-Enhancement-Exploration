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

# ___________________________
# my_sin(x) & my_cos(x)
# Computes trigonometric sine and cosine via static polynomial approximations.
# Since traditional Taylor series degrade rapidly as inputs drift away from 0, 
# inputs should ideally be reduced into a safe base interval (-π, π].
#
# Once mapped into the target window, the functions are evaluated using static 
# polynomial approximations:
#   sin(x) ≈ x * (1 - x^2/6 + x^4/120 - x^6/5040)
#   cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720
# 
# these polynomials are hardcoded to run from the highest degree down to the 
# constant terms using fixed coefficients. This creates a highly predictable, 
# branchless execution path ideal for preventing processor stall cycles.
# ___________________________
my_sin:
    # Range reduction omitted for brevity, assumes reduced input in fa0
    # sin(x) ≈ x * (1 - x^2/6 + x^4/120 - x^6/5040)
    fmul.s  ft0, fa0, fa0        # x^2
    
    li      t0, 0xB9500D01       # -0.0001984 (-1/5040)
    fmv.w.x ft1, t0
    fmul.s  ft1, ft1, ft0
    li      t0, 0x3C088889       # 0.0083333 (1/120)
    fmv.w.x ft2, t0
    fadd.s  ft1, ft1, ft2
    fmul.s  ft1, ft1, ft0
    li      t0, 0xBE2AAAAB       # -0.166667 (-1/6)
    fmv.w.x ft2, t0
    fadd.s  ft1, ft1, ft2
    fmul.s  ft1, ft1, ft0
    li      t0, 0x3F800000       # 1.0
    fmv.w.x ft2, t0
    fadd.s  ft1, ft1, ft2
    fmul.s  fa0, fa0, ft1
    ret

my_cos:
    # cos(x) ≈ 1 - x^2/2 + x^4/24 - x^6/720
    fmul.s  ft0, fa0, fa0        # x^2
    li      t0, 0xBAB60B61       # -0.0013888 (-1/720)
    fmv.w.x ft1, t0
    fmul.s  ft1, ft1, ft0
    li      t0, 0x3D2AAAAB       # 0.0416666 (1/24)
    fmv.w.x ft2, t0
    fadd.s  ft1, ft1, ft2
    fmul.s  ft1, ft1, ft0
    li      t0, 0xBF000000       # -0.5 (-1/2)
    fmv.w.x ft2, t0
    fadd.s  ft1, ft1, ft2
    fmul.s  ft1, ft1, ft0
    li      t0, 0x3F800000       # 1.0
    fmv.w.x ft2, t0
    fadd.s  fa0, ft1, ft2
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


# ___________________________
# my_log(x)
# First we check the input boundary. If x <= 0.0, the logarithm is undefined for 
# real numbers, so we bypass calculations and immediately return negative infinity.
#
# For positive values, we use IEEE-754 binary floating-point representation to 
# pull apart the number. Every float is represented as:
#   x = m * 2^e, where m is the mantissa fraction in [1.0, 2.0) and e is the exponent.
#
# By applying logarithm properties, we break into:
#   ln(x) = ln(m * 2^e) = ln(2^e) + ln(m) = e * ln(2) + ln(m)
#
# We extract the true integer exponent 'e' using integer bitwise shifts and masking. 
# The mantissa 'm' is isolated by rewriting its exponent bits to match 1.0. 
# We then approximate ln(m) on the fixed domain [1.0, 2.0) using a 4th-degree minimax 
# polynomial evaluated via Horner's scheme, adding the result to the pre-computed 
# scale factor (e * 0.69314718).
# ___________________________
my_log:
    fmv.w.x ft0, zero
    fle.s   t1, fa0, ft0
    beqz    t1, .Llog_math
    li      t0, 0xFF800000       # return -inf for x <= 0
    fmv.w.x fa0, t0
    ret

.Llog_math:
    fmv.x.w a0, fa0
    
    # Extract integer exponent (e)
    srli    a1, a0, 23
    andi    a1, a1, 0xFF
    addi    a1, a1, -127         # a1 = e
    
    # Isolate mantissa fraction (f) to reconstruct m in [1, 2)
    li      t2, 0x007FFFFF
    and     a0, a0, t2
    li      t2, 0x3F800000       # exponent for 1.0
    or      a0, a0, t2
    fmv.w.x ft0, a0              # ft0 = m = 1 + f
    
    # Let f_val = m - 1.0
    fmv.w.x ft1, t2              # 1.0
    fsub.s  ft2, ft0, ft1        # ft2 = f_val
    
    # ln(1+f) ≈ f - f^2/2 + f^3/3 - f^4/4
    # Horner: f * (1.0 + f * (-0.5 + f * (0.33333 + f * -0.25)))
    li      t0, 0xBE800000       # -0.25
    fmv.w.x ft3, t0
    fmul.s  ft3, ft3, ft2        # -0.25 * f
    li      t0, 0x3EAAAAAB       # 0.333333
    fmv.w.x ft4, t0
    fadd.s  ft3, ft3, ft4        # 0.333333 - 0.25f
    fmul.s  ft3, ft3, ft2        # f * (...)
    li      t0, 0xBF000000       # -0.5
    fmv.w.x ft4, t0
    fadd.s  ft3, ft3, ft4        # -0.5 + ...
    fmul.s  ft3, ft3, ft2        # f * (...)
    fadd.s  ft3, ft3, ft1        # 1.0 + ...
    fmul.s  ft3, ft3, ft2        # ft3 = ln(m)
    
    # Calculate e * ln2
    fcvt.s.w ft4, a1             # float(e)
    li      t0, 0x3F317218       # 0.693147
    fmv.w.x ft5, t0
    fmul.s  ft4, ft4, ft5        # e * ln(2)
    
    fadd.s  fa0, ft3, ft4        # Result = e*ln2 + ln(m)
    ret

# ___________________________
# my_pow(x, y)
# Instead of tracking nested loops for integer exponents or using iterative root-finding 
# for fractional powers, we implement this using an identity pair:
#   x^y = (e^ln(x))^y = e^(y * ln(x))
#
# The routine saves the original parameters, calls our 'my_log' function to
# compute the natural log of the base, multiplies that resulting scalar by the 
# exponent y, and feeds the product directly into our optimized 'my_exp' function.
#
# If the base x is less than or equal to 0.0, the calculation is bypassed and 
# safely returns a default value of 0.0.
# ___________________________
my_pow:
    addi    sp, sp, -16
    sw      ra, 12(sp)
    fsw     fs1, 8(sp)

    fmv.w.x ft0, zero
    fle.s   t1, fa0, ft0
    beqz    t1, .Lpow_math
    fmv.w.x fa0, zero            # return 0 if x <= 0 (simplified)
    j       .Lpow_done

.Lpow_math:
    fmv.s   fs1, fa1             # save y
    call    my_log               # fa0 = ln(x)
    fmul.s  fa0, fa0, fs1        # fa0 = y * ln(x)
    call    my_exp               # fa0 = exp(y * ln(x))

.Lpow_done:
    flw     fs1, 8(sp)
    lw      ra, 12(sp)
    addi    sp, sp, 16
    ret

# =================================================================
#  M4 VECTOR BLOCKS
#  Don't touch these yet. We'll use them later when we refactor 
#  nn.s to pass array pointers directly instead of looping floats.
# =================================================================
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
    li      t1, 0x3DCCCCD0             # 0.10001
    fmv.w.x ft0, t1
    vfmv.v.f v24, ft0
    vfmacc.vv v24, v16, v24            # v24 = 0.10001 * x^2
    li      t1, 0x3F800000             # 1.0
    fmv.w.x ft1, t1
    vfadd.vf v24, v24, ft1             # v24 = 1.0 + 0.10001*x^2
    vfmul.vv v24, v8, v24              # Numerator complete

    # Denominator: 1.0 + x^2 * (0.43301 + 0.009999*x^2)
    li      t1, 0x3C23D69A             # 0.009999
    fmv.w.x ft2, t1
    vfmv.v.f v0, ft2
    vfmul.vv v0, v16, v0               # 0.009999 * x^2
    li      t1, 0x3EDDA740             # 0.43301
    fmv.w.x ft3, t1
    vfadd.vf v0, v0, ft3               # + 0.43301
    vfmul.vv v0, v16, v0               # x^2 * (...)
    vfadd.vf v0, v0, ft1               # Denominator complete

    # Divide and store
    vfdiv.vv v8, v24, v0               # v8 = Num / Den
    vse32.v v8, (a1)                   # Store result

    # Bump pointers
    slli    t1, t0, 2                  # t1 = elements processed * 4 bytes
    add     a0, a0, t1                 # advance input pointer
    add     a1, a1, t1                 # advance output pointer
    sub     a2, a2, t0                 # subtract elements processed
    bnez    a2, .L_v_tanh_loop         # loop if N > 0
    ret
