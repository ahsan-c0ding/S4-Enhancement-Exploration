# This is the risc-v assembly implementation of math.c
# All function use single-precision floation point (float)
# Arguments in fa0, fa1 and return value in fa0

.global my_exp
.global my_sin
.global my_cos
.global my_tanh
.global my_sqrt
.global my_pow

# my_exp(x) - Exponential using Taylor series
# Input fa0 = x, return fa0 = e^x

my_exp:
    addi sp, sp, -16         # Allocate 16 bytes on stack
    sw ra, 12(sp)	     # Save return address
    fsw fs0, 8(sp)	     # Save fs0

    fmv.s fs0, fa0       # Move x to fs0 (callee-saved, survives 'call')
    li   t0, 0
    fcvt.s.w ft1, t0     # ft1 = 0.0

    flt.s t1, fs0, ft1   # if x < 0
    beq  t1, zero, exp_positive
    # Handle negative: compute 1 / exp(-x)
    fneg.s fa0, fs0      # fa0 = -x (prep argument for recursion)
    call my_exp          # fa0 = exp(-x)

    li   t0, 1
    fcvt.s.w ft1, t0     # ft1 = 1.0
    fdiv.s fa0, ft1, fa0 # fa0 = 1 / exp(-x)
    j    exp_exit        # Jump to exp exit

exp_positive:
    li   t0, 1
    fcvt.s.w ft1, t0     # ft1 = result (starts at 1.0)
    fmv.s ft2, ft1       # ft2 = current term (starts at 1.0)
    li   t0, 1           # t0 = i (counter)

exp_loop:
# term = term * x / i
    fmul.s ft2, ft2, fs0 # term = term * x
    fcvt.s.w ft4, t0     # ft4 = i
    fdiv.s ft2, ft2, ft4 # term = term / i

    fadd.s ft1, ft1, ft2 # result = result + term

    # Convergence check (|term| < 1e-7)
    li   t1, 0x33D6BF95  # 1e-7 hex
    fmv.w.x ft5, t1
    fabs.s ft6, ft2
    flt.s t2, ft6, ft5

    addi t0, t0, 1       # i++
    li   t3, 50          # Match 50 iterations
    bge  t0, t3, exp_done
    beq  t2, zero, exp_loop

exp_done:
    fmv.s fa0, ft1       # Move result to return register

exp_exit:
    flw  fs0, 8(sp)      # Restore fs0
    lw   ra, 12(sp)      # Restore Return Address
    addi sp, sp, 16      # Deallocate stack
    ret

# ============================================================
# my_sin(x) - Sine using Taylor series
# sin x = x - x^3/3! + x^5/5! - ...
# Argument: fa0 = x (radians)
# Return: fa0 = sin(x)
# ============================================================
my_sin:
    # Reduce x to [-π, π]
    li t0, 0x40490FDB        # π = 3.141592653589793
    fmv.w.x ft0, t0
    fmv.s ft1, fa0           # ft1 = x

    # Range reduction - simple version (for now)
    # We'll keep it simple for initial implementation

    fmv.s ft0, fa0           # ft0 = x
    li t0, 1
    fcvt.s.w ft1, t0         # ft1 = 1.0
    fmv.s ft2, ft0           # ft2 = term = x
    fmv.s ft3, ft1           # ft3 = result = 0
    li t0, 1                 # i = 1
    fmv.s ft4, ft0           # x_sq = x * x
    fmul.s ft4, ft4, ft4

sin_loop:
    # Add term to result
    fadd.s ft3, ft3, ft2

    # Next term: term = -term * x^2 / ((i+1)*(i+2))
    fneg.s ft2, ft2          # -term
    fmul.s ft2, ft2, ft4     # * x^2

    # Calculate denominator = (i+1)*(i+2)
    addi t0, t0, 1
    fcvt.s.w ft5, t0
    addi t0, t0, 1
    fcvt.s.w ft6, t0
    fmul.s ft5, ft5, ft6
    fdiv.s ft2, ft2, ft5

    # Check convergence
    fabs.s ft6, ft2
    li t1, 0x33D6BF95        # 1e-7
    fmv.w.x ft7, t1
    flt.s t2, ft6, ft7

    # i += 2
    addi t0, t0, 1
    li t1, 15
    bge t0, t1, sin_done

    beq t2, zero, sin_loop

sin_done:
    fmv.s fa0, ft3
    ret

# ============================================================
# my_cos(x) - Cosine using Taylor series
# cos x = 1 - x^2/2! + x^4/4! - ...
# Argument: fa0 = x (radians)
# Return: fa0 = cos(x)
# ============================================================
my_cos:
    fmv.s ft0, fa0
    li t0, 1
    fcvt.s.w ft1, t0         # ft1 = result = 1.0
    fmv.s ft2, ft1           # ft2 = term = 1.0
    fmul.s ft3, ft0, ft0     # ft3 = x^2
    li t0, 0                 # i = 0

cos_loop:
    # Next term: term = -term * x^2 / ((i+1)*(i+2))
    fneg.s ft2, ft2
    fmul.s ft2, ft2, ft3

    addi t0, t0, 1
    fcvt.s.w ft4, t0
    addi t0, t0, 1
    fcvt.s.w ft5, t0
    fmul.s ft4, ft4, ft5
    fdiv.s ft2, ft2, ft4

    # Add term to result
    fadd.s ft1, ft1, ft2

    # Check convergence
    fabs.s ft6, ft2
    li t1, 0x33D6BF95        # 1e-7
    fmv.w.x ft7, t1
    flt.s t2, ft6, ft7

    # i += 2
    addi t0, t0, 1
    li t1, 15
    bge t0, t1, cos_done

    beq t2, zero, cos_loop

cos_done:
    fmv.s fa0, ft1
    ret

# ============================================================
# my_tanh(x) - Hyperbolic tangent using: tanh(x) = (e^x - e^-x)/(e^x + e^-x)
# Argument: fa0 = x
# Return: fa0 = tanh(x)
# ============================================================
my_tanh:
    addi sp, sp, -32
    sw   ra, 28(sp)
    fsw  fs0, 24(sp)     # To store x
    fsw  fs1, 20(sp)     # To store e^x

    fmv.s fs0, fa0       # Save input x to fs0

    # Large x Check (Clamping)
    li   t0, 0x41200000  # 10.0
    fmv.w.x ft0, t0
    fabs.s ft1, fs0
    flt.s t1, ft1, ft0   # if |x| < 10, proceed to math
    bne  t1, zero, tanh_math

    # If |x| >= 10, return sign(x) * 1.0
    li   t0, 1
    fcvt.s.w fa0, t0
    flt.s t1, fs0, zero  # if x < 0
    beq  t1, zero, tanh_exit
    fneg.s fa0, fa0      # make it -1.0
    j    tanh_exit

tanh_math:
    # Compute e^x
    fmv.s fa0, fs0
    call  my_exp
    fmv.s fs1, fa0       # fs1 = e^x

    # Compute e^-x
    fneg.s fa0, fs0      # arg = -x
    call  my_exp         # fa0 = e^-x

    # (e^x - e^-x) / (e^x + e^-x)
    fsub.s ft1, fs1, fa0 # numerator
    fadd.s ft2, fs1, fa0 # denominator
    fdiv.s fa0, ft1, ft2 # result

tanh_exit:
    lw   ra, 28(sp)
    flw  fs0, 24(sp)
    flw  fs1, 20(sp)
    addi sp, sp, 32
    ret

# ============================================================
# my_sqrt(x) - Square root using Newton's method
# Argument: fa0 = x
# Return: fa0 = sqrt(x)
# ============================================================
my_sqrt:
    # Handle x = 0
    li t0, 0
    fcvt.s.w ft0, t0
    feq.s t1, fa0, ft0
    bne t1, zero, sqrt_zero

    # Initial guess
    fmv.s ft0, fa0           # ft0 = x

    # For x < 1, start with guess = 1
    li t0, 1
    fcvt.s.w ft1, t0         # ft1 = 1.0
    flt.s t1, fa0, ft1       # if x < 1
    beq t1, zero, sqrt_loop
    fmv.s ft0, ft1           # guess = 1.0

sqrt_loop:
    # Newton iteration: guess = (guess + x/guess) / 2
    fdiv.s ft2, fa0, ft0     # ft2 = x / guess
    fadd.s ft2, ft0, ft2     # ft2 = guess + x/guess
    li t0, 2
    fcvt.s.w ft3, t0
    fdiv.s ft2, ft2, ft3     # ft2 = (guess + x/guess)/2

    # Check convergence
    fsub.s ft3, ft2, ft0
    fabs.s ft3, ft3
    li t0, 0x33D6BF95        # 1e-7
    fmv.w.x ft4, t0
    flt.s t1, ft3, ft4

    fmv.s ft0, ft2           # guess = new guess

    beq t1, zero, sqrt_loop

sqrt_done:
    fmv.s fa0, ft0
    ret

sqrt_zero:
    fmv.s fa0, ft0
    ret

# ============================================================
# my_pow(x, y) - Power function: x^y = e^(y * ln(x))
# Arguments: fa0 = x, fa1 = y
# Return: fa0 = x^y
# ============================================================
my_pow:
    addi sp, sp, -16
    sw   ra, 12(sp)
    fsw  fs1, 8(sp)      # Use fs1 to keep 'y' safe during calls

    # If x <= 0, return 0
    li   t0, 0
    fcvt.s.w ft0, t0
    fle.s t1, fa0, ft0
    bne  t1, zero, pow_zero_exit

    fmv.s fs1, fa1       # Save y into fs1 (will survive call to my_log)

    # Compute ln(x)
    # fa0 already contains x
    call my_log          # fa0 = ln(x)

    fmul.s fa0, fa0, fs1 # fa0 = ln(x) * y
    call my_exp          # fa0 = exp(y * ln(x))
    j    pow_finish

pow_zero_exit:
    li   t0, 0
    fcvt.s.w fa0, t0

pow_finish:
    flw  fs1, 8(sp)
    lw   ra, 12(sp)
    addi sp, sp, 16
    ret

# ============================================================
# my_log(x) - Natural log using Newton's method
# Solve e^y = x for y
# Argument: fa0 = x
# Return: fa0 = ln(x)
# ============================================================
my_log:
    addi sp, sp, -32     # Space for ra, fs0 (x), fs1 (y), s0 (counter)
    sw   ra, 28(sp)
    fsw  fs0, 24(sp)     # We'll store original 'x' here
    fsw  fs1, 20(sp)     # We'll store current guess 'y' here
    sw   s0, 16(sp)      # Iteration counter

    # Check x > 0
    li   t0, 0
    fcvt.s.w ft0, t0
    fle.s t1, fa0, ft0
    beq  t1, zero, log_positive

    # x <= 0: return -inf (0xFF800000)
    li   t0, 0xFF800000
    fmv.w.x fa0, t0
    j    log_exit

log_positive:
    fmv.s fs0, fa0       # fs0 = x
    # Initial guess: y = x - 1
    li   t0, 1
    fcvt.s.w ft0, t0
    fsub.s fs1, fs0, ft0 # fs1 = y = x - 1
    li   s0, 20          # Max 20 iterations

log_loop:
    # Compute e^y
    fmv.s fa0, fs1       # Arg for my_exp
    call  my_exp         # fa0 = e^y
    fmv.s ft2, fa0       # ft2 = e^y

    # diff = e^y - x
    fsub.s ft3, ft2, fs0 # ft3 = diff

    # y = y - (diff / e^y)
    fdiv.s ft4, ft3, ft2 # ft4 = diff / e^y
    fsub.s fs1, fs1, ft4 # y = y - ft4

    # Convergence check: |diff| < 1e-7
    fabs.s ft5, ft3
    li   t2, 0x33D6BF95  # 1e-7
    fmv.w.x ft6, t2
    flt.s t3, ft5, ft6

    addi s0, s0, -1      # counter--
    beq  s0, zero, log_done
    beq  t3, zero, log_loop

log_done:
    fmv.s fa0, fs1       # Return y

log_exit:
    lw   ra, 28(sp)
    flw  fs0, 24(sp)
    flw  fs1, 20(sp)
    lw   s0, 16(sp)
    addi sp, sp, 32
    ret
