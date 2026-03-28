# Arguments: fa0=ar, fa1=ai, fa2=br, fa3=bi
# Returns: fa0=out_r, fa1=out_i
complex_mul:
    fmul.s ft0, fa0, fa2    # ar * br
    fmul.s ft1, fa1, fa3    # ai * bi
    fsub.s ft4, ft0, ft1    # result_real = ar*br - ai*bi

    fmul.s ft2, fa0, fa3    # ar * bi
    fmul.s ft3, fa1, fa2    # ai * br
    fadd.s ft5, ft2, ft3    # result_imag = ar*bi + ai*br

    fmv.s  fa0, ft4
    fmv.s  fa1, ft5
    ret

# Arguments: a0=input_ptr, a1=output_ptr, a2=indices_ptr
# Constants: SEQ_LEN=4096, IN_CHANNELS=1
hilbert_scan:
    li   t0, 0              # d = 0
    li   t1, 4096           # SEQ_LEN

hilbert_loop:
    # flat_idx = hilbert_indices[d]
    slli t2, t0, 2          # offset = d * 4
    add  t3, a2, t2         # addr of indices[d]
    lw   t4, 0(t3)          # t4 = flat_idx

    # Simple Safety Check: if (flat_idx < 0 || flat_idx >= 4096) flat_idx = 0
    blt  t4, zero, h_reset
    li   t5, 4096
    blt  t4, t5, h_access
h_reset:
    li   t4, 0

h_access:
    # input[flat_idx]
    slli t5, t4, 2          # offset = flat_idx * 4
    add  t6, a0, t5         # addr of input[flat_idx]
    flw  ft0, 0(t6)         # Load pixel

    # output[d] = pixel
    add  t6, a1, t2         # addr of output[d]
    fsw  ft0, 0(t6)

    addi t0, t0, 1          # d++
    blt  t0, t1, hilbert_loop
    ret
# Arguments: a0=input_ptr (4096x64), a1=output_ptr (64)
# Logic: Copy input[4095][0...63] to output
take_last_timestamp:
    li   t0, 4095           # Last index
    li   t1, 64             # D_MODEL
    mul  t0, t0, t1         # Row offset (in elements)
    slli t0, t0, 2          # Row offset (in bytes)
    add  a0, a0, t0         # Point a0 to the start of the last row

    li   t2, 0              # loop counter j
timestamp_loop:
    slli t3, t2, 2          # j * 4
    add  t4, a0, t3         # input[4095][j]
    flw  ft0, 0(t4)

    add  t5, a1, t3         # output[j]
    fsw  ft0, 0(t5)

    addi t2, t2, 1
    blt  t2, t1, timestamp_loop
    ret

.section .text
.global linear

# Generic Linear Layer (Matrix-Matrix Multiplication + Bias)
linear:
    # We use t-registers (caller-saved) as we don't call other functions here.
    # t0: i (outer loop counter - batch_size)
    # t1: j (middle loop counter - out_features)
    # t4: k (inner loop counter - in_features)

    li t0, 0                    # i = 0
outer_loop:
    bge t0, a4, end_linear      # if i >= batch_size, exit

    li t1, 0                    # j = 0
middle_loop:
    bge t1, a6, next_i          # if j >= out_features, next row

    # float acc = bias[j]
    slli t2, t1, 2              # j * 4 bytes
    add  t3, a3, t2             # addr of bias[j]
    flw  fa0, 0(t3)             # fa0 = acc

    li t4, 0                    # k = 0
inner_loop:
    bge t4, a5, store_result    # if k >= in_features, store the accumulation

    # Load input[i * in_features + k]
    mul  t2, t0, a5             # i * in_features
    add  t2, t2, t4             # + k
    slli t2, t2, 2              # * 4 (float size)
    add  t3, a0, t2             # input address
    flw  ft0, 0(t3)             # ft0 = input val

    # Load weight[j * in_features + k]
    mul  t2, t1, a5             # j * in_features
    add  t2, t2, t4             # + k
    slli t2, t2, 2              # * 4
    add  t3, a2, t2             # weight address
    flw  ft1, 0(t3)             # ft1 = weight val

    # acc += input * weight
    fmul.s ft2, ft0, ft1
    fadd.s fa0, fa0, ft2

    addi t4, t4, 1              # k++
    j inner_loop

store_result:
    # output[i * out_features + j] = acc
    mul  t2, t0, a6             # i * out_features
    add  t2, t2, t1             # + j
    slli t2, t2, 2              # * 4
    add  t3, a1, t2             # output address
    fsw  fa0, 0(t3)

    addi t1, t1, 1              # j++
    j middle_loop

next_i:
    addi t0, t0, 1              # i++
    j outer_loop

end_linear:
    ret

.section .text
.global gelu

gelu:
    # --- Stack Setup ---
    addi sp, sp, -32        # Allocate stack space
    sw   ra, 28(sp)         # Save return address
    sw   s0, 24(sp)         # s0 = current buffer pointer
    sw   s1, 20(sp)         # s1 = end pointer
    fsw  fs0, 16(sp)        # fs0 = original x value

    mv   s0, a0             # Initialize current pointer
    slli t0, a1, 2          # total_elements * 4 bytes
    add  s1, a0, t0         # end address pointer

    # --- Pre-load Constants ---
    # We use hex bit-patterns for single-precision floats
    li   t1, 0x3F000000     # 0.5
    fmv.w.x ft8, t1
    li   t1, 0x3F800000     # 1.0
    fmv.w.x ft9, t1
    li   t1, 0x3D372713     # 0.044715
    fmv.w.x ft10, t1
    li   t1, 0x3F4C422A     # sqrt(2/pi) approx 0.79788456
    fmv.w.x ft11, t1

gelu_loop:
    bge  s0, s1, gelu_done  # If current_ptr >= end_ptr, exit

    flw  fs0, 0(s0)         # Load x

    # 1. Calculate x^3
    fmul.s ft0, fs0, fs0    # x^2
    fmul.s ft0, ft0, fs0    # x^3

    # 2. inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fmul.s ft0, ft0, ft10   # 0.044715 * x^3
    fadd.s ft0, ft0, fs0    # x + ...
    fmul.s fa0, ft0, ft11   # Multiply by sqrt(2/pi) -> Argument for tanh

    # 3. Call my_tanh (Argument in fa0, Result in fa0)
    # We must save ft registers if they were needed, but we re-load/compute
    call my_tanh

    # 4. final = 0.5 * x * (1.0 + tanh_result)
    fadd.s ft1, fa0, ft9    # (1.0 + tanh)
    fmul.s ft1, ft1, fs0    # x * (1.0 + tanh)
    fmul.s ft1, ft1, ft8    # 0.5 * ...

    fsw  ft1, 0(s0)         # Store result back to memory

    addi s0, s0, 4          # Move to next float
    j    gelu_loop

gelu_done:
    # --- Stack Teardown ---
    lw   ra, 28(sp)
    lw   s0, 24(sp)
    lw   s1, 20(sp)
    flw  fs0, 16(sp)
    addi sp, sp, 32
    ret

.section .text
.global softmax

softmax:
    # --- Stack Setup ---
    addi sp, sp, -48
    sw   ra, 44(sp)
    sw   s0, 40(sp)         # s0 = pointer to logits
    sw   s1, 36(sp)         # s1 = number of classes
    fsw  fs0, 32(sp)        # fs0 = max_val
    fsw  fs1, 28(sp)        # fs1 = sum of exps

    mv   s0, a0
    mv   s1, a1

    # --- Pass 1: Find Max ---
    flw  fs0, 0(s0)         # max_val = logits[0]
    li   t0, 1              # counter i = 1
find_max_loop:
    bge  t0, s1, end_max
    slli t1, t0, 2
    add  t2, s0, t1
    flw  ft0, 0(t2)
    fle.s t3, ft0, fs0      # if ft0 <= fs0, t3 = 1
    bne  t3, zero, not_new_max
    fmv.s fs0, ft0          # new max found
not_new_max:
    addi t0, t0, 1
    j    find_max_loop

end_max:
    # --- Pass 2: Exp(x - max) and Summing ---
    fmv.s.x fs1, zero       # sum = 0.0
    li   s2, 0              # i = 0 (using s2 because we call my_exp)
exp_sum_loop:
    bge  s2, s1, end_exp_sum
    slli t0, s2, 2
    add  t1, s0, t0
    flw  ft0, 0(t1)
    fsub.s fa0, ft0, fs0    # arg = x - max

    call my_exp             # fa0 = exp(x - max)

    slli t0, s2, 2
    add  t1, s0, t0
    fsw  fa0, 0(t1)          # Store exp(x-max) back in buffer
    fadd.s fs1, fs1, fa0     # sum += exp_val

    addi s2, s2, 1
    j    exp_sum_loop

end_exp_sum:
    # --- Pass 3: Normalize (Divide by Sum) ---
    li   t0, 0              # i = 0
normalize_loop:
    bge  t0, s1, softmax_done
    slli t1, t0, 2
    add  t2, s0, t1
    flw  ft0, 0(t2)
    fdiv.s ft0, ft0, fs1     # x / sum
    fsw  ft0, 0(t2)

    addi t0, t0, 1
    j    normalize_loop

softmax_done:
    # --- Stack Teardown ---
    lw   ra, 44(sp)
    lw   s0, 40(sp)
    lw   s1, 36(sp)
    flw  fs0, 32(sp)
    flw  fs1, 28(sp)
    addi sp, sp, 48
    ret
