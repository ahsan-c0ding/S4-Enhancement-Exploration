.global complex_mul
.global hilbert_scan
.global take_last_timestamp
.global linear
.global gelu
.global softmax

# Arguments: fa0=ar, fa1=ai, fa2=br, fa3=bi
#            a0=out_r (pointer), a1=out_i (pointer)
# Returns:   void (writes results to *out_r and *out_i)
complex_mul:
    fmul.s ft0, fa0, fa2    # ar * br
    fmul.s ft1, fa1, fa3    # ai * bi
    fsub.s ft4, ft0, ft1    # result_real = ar*br - ai*bi

    fmul.s ft2, fa0, fa3    # ar * bi
    fmul.s ft3, fa1, fa2    # ai * br
    fadd.s ft5, ft2, ft3    # result_imag = ar*bi + ai*br

    fsw    ft4, 0(a0)       # *out_r = result_real
    fsw    ft5, 0(a1)       # *out_i = result_imag
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

    # Safety check: if (flat_idx < 0 || flat_idx >= 4096) flat_idx = 0
    blt  t4, zero, h_reset
    li   t5, 4096
    blt  t4, t5, h_access
h_reset:
    li   t4, 0

h_access:
    # Channel loop over IN_CHANNELS
    # IN_CHANNELS = 1, so t6 = 1 and this loop body executes exactly once (c=0 only, never loops back)
    li   t5, 0              # c = 0
    li   t6, 1              # IN_CHANNELS = 1

channel_loop:
    bge  t5, t6, channel_done

    # Load input[c][flat_idx]
    # input is laid out as [IN_CHANNELS][SEQ_LEN] floats
    # addr = a0 + (c * 4096 + flat_idx) * 4
    li   t3, 4096
    mul  t3, t5, t3         # c * SEQ_LEN
    add  t3, t3, t4         # + flat_idx
    slli t3, t3, 2          # * 4
    add  t3, a0, t3
    flw  ft0, 0(t3)

    # Store output[d][c]
    # output is laid out as [SEQ_LEN][IN_CHANNELS] floats
    # addr = a1 + (d * IN_CHANNELS + c) * 4
    # Since IN_CHANNELS = 1, this simplifies to a1 + (d + c) * 4
    mul  t3, t0, t6         # d * IN_CHANNELS
    add  t3, t3, t5         # + c
    slli t3, t3, 2          # * 4
    add  t3, a1, t3
    fsw  ft0, 0(t3)

    addi t5, t5, 1          # c++
    j    channel_loop

channel_done:
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
    addi sp, sp, -32
    sw   ra, 28(sp)
    sw   s0, 24(sp)     # s0 = batch_size  (a4)
    sw   s1, 20(sp)     # s1 = in_features (a5)
    sw   s2, 16(sp)     # s2 = out_features (a6)
    sw   s3, 12(sp)     # s3 = input ptr (a0)
    sw   s4, 8(sp)      # s4 = output ptr (a1)
    sw   s5, 4(sp)      # s5 = weight ptr (a2)
    sw   s6, 0(sp)      # s6 = bias ptr (a3)

    mv   s0, a4         # batch_size
    mv   s1, a5         # in_features
    mv   s2, a6         # out_features
    mv   s3, a0         # input
    mv   s4, a1         # output
    mv   s5, a2         # weight
    mv   s6, a3         # bias

    li t0, 0                    # i = 0
outer_loop:
    bge t0, s0, end_linear

    li t1, 0                    # j = 0
middle_loop:
    bge t1, s2, next_i

    # acc = bias[j]
    slli t2, t1, 2
    add  t3, s6, t2
    flw  fa0, 0(t3)

    li t4, 0                    # k = 0
inner_loop:
    bge t4, s1, store_result

    # input[i * in_features + k]
    mul  t2, t0, s1
    add  t2, t2, t4
    slli t2, t2, 2
    add  t3, s3, t2
    flw  ft0, 0(t3)

    # weight[j * in_features + k]
    mul  t2, t1, s1
    add  t2, t2, t4
    slli t2, t2, 2
    add  t3, s5, t2
    flw  ft1, 0(t3)

    fmul.s ft2, ft0, ft1
    fadd.s fa0, fa0, ft2

    addi t4, t4, 1
    j inner_loop

store_result:
    # output[i * out_features + j]
    mul  t2, t0, s2
    add  t2, t2, t1
    slli t2, t2, 2
    add  t3, s4, t2
    fsw  fa0, 0(t3)

    addi t1, t1, 1
    j middle_loop

next_i:
    addi t0, t0, 1
    j outer_loop

end_linear:
    lw   ra, 28(sp)
    lw   s0, 24(sp)
    lw   s1, 20(sp)
    lw   s2, 16(sp)
    lw   s3, 12(sp)
    lw   s4, 8(sp)
    lw   s5, 4(sp)
    lw   s6, 0(sp)
    addi sp, sp, 32
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

    # 4. Reload constants destroyed by the call
    li   t1, 0x3F800000     # 1.0
    fmv.w.x ft9, t1
    li   t1, 0x3F000000     # 0.5
    fmv.w.x ft8, t1

    # 5. final = 0.5 * x * (1.0 + tanh_result)
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
    sw   s2, 24(sp)         # save s2 before use
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
    fmv.w.x fs1, zero       # sum = 0.0
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
    lw   s2, 24(sp)  # restore s2
    flw  fs0, 32(sp)
    flw  fs1, 28(sp)
    addi sp, sp, 48
    ret

.global complex_exp

# Arguments: fa0=a_real, fa1=a_imag
#            a0=out_real (pointer), a1=out_imag (pointer)
# Returns:   void (writes results to *out_real and *out_imag)
complex_exp:
    # --- Stack Setup ---
    addi sp, sp, -32
    sw   ra,  28(sp)
    sw   s0,  24(sp)        # s0 = out_real pointer
    sw   s1,  20(sp)        # s1 = out_imag pointer
    fsw  fs0, 16(sp)        # fs0 = a_imag (survives calls)
    fsw  fs1, 12(sp)        # fs1 = exp_a  (survives calls)

    # Save arguments that will be clobbered by calls
    fmv.s fs0, fa1          # fs0 = a_imag
    mv    s0,  a0           # s0  = out_real pointer
    mv    s1,  a1           # s1  = out_imag pointer

    # Step 1: exp_a = my_exp(a_real)
    # fa0 already = a_real, ready to call
    call  my_exp            # fa0 = exp(a_real)
    fmv.s fs1, fa0          # fs1 = exp_a (save before next call)

    # Step 2: out_real = exp_a * my_cos(a_imag)
    fmv.s fa0, fs0          # fa0 = a_imag (argument for cos)
    call  my_cos            # fa0 = cos(a_imag)
    fmul.s ft0, fs1, fa0    # ft0 = exp_a * cos(a_imag)
    fsw   ft0, 0(s0)        # *out_real = result

    # Step 3: out_imag = exp_a * my_sin(a_imag)
    fmv.s fa0, fs0          # fa0 = a_imag (argument for sin)
    call  my_sin            # fa0 = sin(a_imag)
    fmul.s ft0, fs1, fa0    # ft0 = exp_a * sin(a_imag)
    fsw   ft0, 0(s1)        # *out_imag = result

    # --- Stack Teardown ---
    lw   ra,  28(sp)
    lw   s0,  24(sp)
    lw   s1,  20(sp)
    flw  fs0, 16(sp)
    flw  fs1, 12(sp)
    addi sp, sp, 32
    ret

