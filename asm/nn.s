.global complex_mul
.global hilbert_scan
.global take_last_timestamp
.global linear
.global gelu
.global softmax

#Comments are here to help keep track of what C code we are replicating
#Abdul Rehman writes these comments and he does it very autistically  

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
# Arguments  a0=input_ptr, a1=output_ptr, a2=indices_ptr
# Constants  SEQ_LEN=4096, IN_CHANNELS=1
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
#  Copy input[4095][0...63] to output
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
    # using different temporary registers (t5, t6) to calculate weight offset
    mul  t5, t1, s1         # row offset = j * in_features
    add  t5, t5, t4         # + col offset (k)
    slli t6, t5, 2          # multiply by 4 bytes
    add  t6, s5, t6         # add to weight base ptr
    flw  ft1, 0(t6)         # load weight

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
    # --- Stack Setup --
    addi sp, sp, -32
    sw   ra, 28(sp)
    sw   s0, 24(sp)
    sw   s1, 20(sp)
    fsw  fs0, 16(sp)

    mv   s0, a0
    slli t0, a1, 2          # total_elements * 4 bytes
    add  s1, a0, t0         # end address pointer

gelu_loop:
    bge  s0, s1, gelu_done

    flw  fs0, 0(s0)         # Load x

    # 1. Calculate x^3
    fmul.s ft0, fs0, fs0    # x^2
    fmul.s ft0, ft0, fs0    # x^3

    # fix. the reload constants here because 'call my_tanh' destroyed them
    li   t1, 0x3D372713     # 0.044715
    fmv.w.x ft10, t1
    li   t1, 0x3F4C422A     # sqrt(2/pi) approx 0.79788456
    fmv.w.x ft11, t1

    # 2 inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fmul.s ft0, ft0, ft10
    fadd.s ft0, ft0, fs0
    fmul.s fa0, ft0, ft11

    # 3 Call my_tanh
    call my_tanh

    # 4 Reload the other constants destroyed by the call
    li   t1, 0x3F800000     # 1.0
    fmv.w.x ft9, t1
    li   t1, 0x3F000000     # 0.5
    fmv.w.x ft8, t1

    # 5 final = 0.5 * x * (1.0 + tanh_result)
    fadd.s ft1, fa0, ft9
    fmul.s ft1, ft1, fs0
    fmul.s ft1, ft1, ft8

    fsw  ft1, 0(s0)

    addi s0, s0, 4
    j    gelu_loop

gelu_done:
    # -- Stack Teardown ---
    lw   ra, 28(sp)
    lw   s0, 24(sp)
    lw   s1, 20(sp)
    flw  fs0, 16(sp)
    addi sp, sp, 32
    ret

.section .text
.global softmax

softmax:
    # --- stack Setup ---
    addi sp, sp, -48
    sw   ra, 44(sp)
    sw   s0, 40(sp)         # s0 = pointer to logits
    sw   s1, 36(sp)         # s1 = number of classes
    sw   s2, 24(sp)         # save s2 before use
    fsw  fs0, 32(sp)        # fs0 = max_val
    fsw  fs1, 28(sp)        # fs1 = sum of exps

    mv   s0, a0
    mv   s1, a1

    # --- Pass 1 Find Max ---
    flw  fs0, 0(s0)         # max_val = logits[0]
    li   t0, 1              # counter i = 1
find_max_loop:
    bge  t0, s1, end_max
    slli t1, t0, 2          # i * 4
    add  t2, s0, t1         # addr = base + i*4
    flw  ft0, 0(t2)         # val = logits[i]
    
    # is new_val > max_val
    flt.s t3, fs0, ft0      # if max_val < val, t3 = 1
    beq   t3, zero, not_new_max # if t3 is 0 (max is greater), skip update
    fmv.s fs0, ft0          # new max found
not_new_max:
    addi t0, t0, 1
    j    find_max_loop

end_max:
    #  Pass 2: Exp(x - max) and Summing
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
    #  Pass 3 Normalize (Divide by Sum) 
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
    # --- Stack Setup --
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

    # Step 1 exp_a = my_exp(a_real)
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

    # -- Stack Teardown --
    lw   ra,  28(sp)
    lw   s0,  24(sp)
    lw   s1,  20(sp)
    flw  fs0, 16(sp)
    flw  fs1, 12(sp)
    addi sp, sp, 32
    ret

.section .text
.global s4d_layer
# S4D LAYER
# Arguments list:
# a0=input, a1=output, a2=log_dt, a3=log_A_real, a4=A_imag, a5=C_real, a6=C_imag, a7=D

s4d_layer:
    # this part of the code is the stack preparation
    # save all callee registers we will use
    # we move the stack pointer down to create 80 bytes of temp storage
    addi sp, sp, -80

    # backup of the values of saved registers s0-s11
    sw   ra, 76(sp)
    sw   s0, 72(sp)
    sw   s1, 68(sp)
    sw   s2, 64(sp)
    sw   s3, 60(sp)
    sw   s4, 56(sp)
    sw   s5, 52(sp)
    sw   s6, 48(sp)
    sw   s7, 44(sp)
    sw   s8, 40(sp)
    sw   s9, 36(sp)
    sw   s10, 32(sp)
    sw   s11, 28(sp)

    #backup of the values of floating point registers fs0 and fs1
    fsw  fs0, 24(sp)
    fsw  fs1, 20(sp)

    #when the function finishes we can restore values in registers

    # recieves function arguments across registers a0-a7 which are then moved to s-registers for safer execution during loops
    mv   s7, a0         # s7 = input ptr
    mv   s8, a1         # s8 = output ptr
    mv   s6, a2         # s6 = log_dt ptr
    mv   s2, a3         # s2 = log_A_real ptr
    mv   s3, a4         # s3 = A_imag ptr
    mv   s4, a5         # s4 = C_real ptr
    mv   s5, a6         # s5 = C_imag ptr
    mv   s9, a7         # s9 = D ptr

    li   s1, 32         # loads 32 in s1 ("half-state" size)
    li   s0, 0          # s0 = 0 (tracks which of the 64 channels the function is currently processing)

.L_s4d_channel_loop:
    #this repeats 64 times
    li   t0, 64
    bge  s0, t0, .L_s4d_done #checks the stopping condition which is when counter reaches 64

    # Compute dt = exp(log_dt[h])
    slli t1, s0, 2    # multiplies the counter by 4 (2 bits left shift)
    add  t2, s6, t1   # adds offset to the base address of log_dt to exactly point to the current channel number
    flw  fa0, 0(t2)   # stores the above calculated value in a floating point register
    call my_exp      #calculates exponential of the value just loaded
    fmv.s fs0, fa0      # moves the result of the calculation in fs0 so that it can be used later


#discretization and kernel generation block
    li      t0, 16960
    sub     sp, sp, t0              # reserves 16960 bytes on the stack


# backup of saved registers
    sw      s6,   0(sp)
    sw      s7,   4(sp)
    sw      s8,   8(sp)
    sw      s9,  12(sp)
    sw      s10, 16(sp)
    sw      s11, 20(sp)

# backup of floating point registers
    fsw     fs1, 24(sp)
    fsw     fs2, 28(sp)
    fsw     fs3, 32(sp)


# fragmenting the large 16960-byte workspace into small sections. 
# loading offsets into a temporary register first to calculate pointers
    li      t4, 64
    add     s11, sp, t4   # table for real and img parts of A

# table for discretized values
    li      t4, 320
    add     s9, sp, t4
    
    li      t4, 448
    add     s10, sp, t4

    li      t4, 576
    add     s8, sp, t4

    li      s6, 0


.Ldisc_loop:

# this loop runs 32 times

    bge     s6, s1, .Ldisc_done
    mul     t0, s0, s1
    add     t0, t0, s6

    slli    t1, t0, 2   #calculation for the exact memory location for the current channel's A's real and img values
    add     t1, s2, t1
    flw     fa0, 0(t1)


    call    my_exp  #for calculating e^(A_real)
    fneg.s  fs1, fa0  #for getting -ve result used to step the model forward time


    # calculating offset for imaginary part now using different registers
    mul     t3, s0, s1       # channel * 32
    add     t3, t3, s6       # + n
    slli    t4, t3, 2        # multiply by 4 bytes for float size
    add     t4, s3, t4       # add to A_imag base pointer
    flw     fs2, 0(t4)       # load imaginary A
    slli    t1, s6, 2
    add     t2, s11, t1
    fsw     fs1, 0(t2)
    addi    t3, s11, 128
    add     t3, t3, t1
    fsw     fs2, 0(t3)

    fmul.s  fa0, fs1, fs0    # It scales the real and imaginary parts by dt
    fmul.s  fa1, fs2, fs0
    # Using sp+36 and sp+40 as a temporary 8-byte buffer to catch the 
    # out_real and out_imag return values from the complex function
    addi    a0, sp, 36
    addi    a1, sp, 40
    call    complex_exp   # It calculates the complex version of e^(A * dt)

    flw     ft0, 36(sp)
    flw     ft1, 40(sp)
    li      t0, 0x3F800000


# this section is implementing the specific formula ((e^(A * dt) - 1) * A^-1 * C)
    fmv.w.x ft2, t0
    fsub.s  ft3, ft0, ft2
    
    # Calculate denominator components separately
    fmul.s  ft4, fs1, fs1
    fmul.s  ft8, fs2, fs2      # ft8 = fs2 * fs2
    fadd.s  ft4, ft4, ft8      # ft4 = ft4 + ft8
    
    # Calculate numerator components separately
    fmul.s  ft5, ft3, fs1
    fmul.s  ft9, ft1, fs2      # ft9 = ft1 * fs2
    fadd.s  ft5, ft5, ft9      # ft5 = ft5 + ft9
    fdiv.s  ft5, ft5, ft4
    fmul.s  ft6, ft1, fs1
    fmul.s  ft7, ft3, fs2
    fsub.s  ft6, ft6, ft7
    fdiv.s  ft6, ft6, ft4


    mul     t0, s0, s1       # channel * 32
    add     t0, t0, s6       # + n
    
    #  2 floats per complex number and 4 bytes per float
    slli    t1, t0, 1        # multiply index by 2
    slli    t1, t1, 2        # multiply by 4 bytes to get final memory offset
    
    add     t2, s4, t1       # add offset to C_real base
    flw     fa0, 0(t2)
    add     t3, s5, t1       # add offset to C_imag base
    flw     fa1, 0(t3)
    fmv.s   fa2, ft5
    fmv.s   fa3, ft6
    slli    t1, s6, 2
    add     a0, s9,  t1
    add     a1, s10, t1

# it multiplies the result by the C matrix values
    call    complex_mul
    addi    s6, s6, 1
    j       .Ldisc_loop
.Ldisc_done:

    li      s7, 0
.Lkernel_t_loop:
    li      t0, 4096    # The loop is set to run 4,096 times

    bge     s7, t0, .Lkernel_done
    slli    t1, s7, 2
    add     t2, s8, t1

    fmv.w.x fa0, zero  #  it clears the previous value in the output memory (s8) by writing a zero to it
    fsw     fa0, 0(t2)
    fcvt.s.w fs3, s7   # It converts the current loop counter into a floating-point number.
    fmul.s   fs3, fs3, fs0   # It multiplies that number by the time-step (dt) calculated in the very first block. This gives the exact timestamp
    li      s6, 0


.Lkernel_n_loop:

#  For every single one of those 4,096 time steps, the computer must sum up the contributions of 32 different states
#   pulls the A matrix values (real and imaginary) that were stored in the s11 table during the discretization phase

    bge     s6, s1, .Lkernel_n_done
    slli    t1, s6, 2
    add     t2, s11, t1
    flw     ft0, 0(t2)
    addi    t3, s11, 128
    add     t3, t3, t1
    flw     ft1, 0(t3)
    fmul.s  fa0, fs3, ft0
    fmul.s  fa1, fs3, ft1
    addi    a0, sp, 36
    addi    a1, sp, 40


    call    complex_exp   #   calculates the complex exponential e^(A*t)
    slli    t1, s6, 2

#   It loads the discretized C values from the s9 and s10 tables

    add     t2, s9,  t1
    flw     fa0, 0(t2)
    add     t3, s10, t1
    flw     fa1, 0(t3)
    flw     fa2, 36(sp)
    flw     fa3, 40(sp)
    addi    a0, sp, 36
    addi    a1, sp, 40


    call    complex_mul   # It multiplies the evolved state by the weight

#   result is added to the total for the particular timestamp

    flw     ft0, 36(sp)      # load result
    li      t0, 0x40000000   # 2.0
    fmv.w.x ft1, t0          # ft1 = 2.0 float
    
    # calculate address for kernel[t] using different registers
    slli    t4, s7, 2        # t4 = time step * 4 bytes
    add     t5, s8, t4       # t5 = base + offset
    flw     ft2, 0(t5)       # load current kernel[t]
    
    # Humanized math: multiply then add instead of Fused Multiply-Add
    fmul.s  ft3, ft1, ft0    # ft3 = 2.0 * result
    fadd.s  ft2, ft2, ft3    # kernel[t] = kernel[t] + (2.0 * result)
    
    fsw     ft2, 0(t5)       # store back to memory
    addi    s6, s6, 1
    j       .Lkernel_n_loop
.Lkernel_n_done:
    addi    s7, s7, 1
    j       .Lkernel_t_loop
.Lkernel_done:
#  the cleanup phase
# integers and decimal floating point numbers are restored from the registers
    lw      s6,   0(sp)
    lw      s7,   4(sp)
    lw      s8,   8(sp)
    lw      s9,  12(sp)
    lw      s10, 16(sp)
    lw      s11, 20(sp)
    flw     fs1, 24(sp)
    flw     fs2, 28(sp)
    flw     fs3, 32(sp)

    addi    sp, sp, 576   #  moves the Stack Pointer (sp) back up by 576 bytes




#casual convolution
#  Now that the computer has the Kernel (the memory weights) and the D value (the direct skip connection),
# it combines them with the input data to produce the final output.
    li   s10, 0         # s10 = k = 0
    li   t6, 4096       # SEQ_LEN
.L_conv_k_loop:

#   This loop iterates through the entire sequence of 4,096 steps.
    bge  s10, t6, .L_conv_k_done

    # Load D[h]
    slli t0, s0, 2
    add  t0, s9, t0     # s9 is D ptr
    flw  fs1, 0(t0)     # fs1 = D[h], It loads the value D for the current channel.

    # acc = D[h] * input[k][h]
    li   t1, 64
    mul  t0, s10, t1    # k * 64
    add  t0, t0, s0     # k * 64 + h
    slli t0, t0, 2
    add  t0, s7, t0     # s7 is input ptr
    flw  ft0, 0(t0)     #  it loads the Input (x) at the current time step k.
    fmul.s fs1, fs1, ft0 # acc = D[h] * input[k][h]

    # Inner loop j
    li   s11, 0         # s11 = j = 0
.L_conv_j_loop:
#  loop for j only goes from 0 up to k
    bgt  s11, s10, .L_conv_j_done

    # Load kernel[j]
    slli t0, s11, 2
    add  t0, sp, t0     # sp points to kernel array!
    flw  ft0, 0(t0)     # ft0 = kernel[j], grabs the weight from the Kernel array

    # Load input[k-j][h]
    sub  t0, s10, s11   # k - j
    li   t1, 64
    mul  t0, t0, t1     # (k-j) * 64
    add  t0, t0, s0     # (k-j) * 64 + h
    slli t0, t0, 2
    add  t0, s7, t0     # s7 is input ptr
    flw  ft1, 0(t0)     # picks up that past input value.

    # acc += kernel[j] * input[k-j][h]
    # doing standard multiply then add instead of fused instruction
    fmul.s ft2, ft0, ft1
    fadd.s fs1, fs1, ft2

    addi s11, s11, 1
    j    .L_conv_j_loop

.L_conv_j_done:
    # Store output[k][h]
    li   t1, 64
    mul  t0, s10, t1
    add  t0, t0, s0
    slli t0, t0, 2
    add  t0, s8, t0     # s8 is output ptr
    fsw  fs1, 0(t0)

    addi s10, s10, 1
    j    .L_conv_k_loop

.L_conv_k_done:
    # Reclaim the 16384 bytes of stack used for kernel array
    li   t0, 16384
    add  sp, sp, t0

    # Advance to next channel
    addi s0, s0, 1
    j    .L_s4d_channel_loop

.L_s4d_done:
    # Restore saved registers
    lw   ra, 76(sp)
    lw   s0, 72(sp)
    lw   s1, 68(sp)
    lw   s2, 64(sp)
    lw   s3, 60(sp)
    lw   s4, 56(sp)
    lw   s5, 52(sp)
    lw   s6, 48(sp)
    lw   s7, 44(sp)
    lw   s8, 40(sp)
    lw   s9, 36(sp)
    lw   s10, 32(sp)
    lw   s11, 28(sp)

    flw  fs0, 24(sp)
    flw  fs1, 20(sp)
    addi sp, sp, 80
    ret

# -----------------------------------------------------------------
# STATIC MEMORY ALLOCATION (.bss)
# As required by M3 rubric: "Use static arrays for intermediate buffers"
# -----------------------------------------------------------------
.section .bss
.align 4

buf_hilbert: .space 16384       # 4096 * 1 * 4 bytes
buf_proj:    .space 1048576     # 4096 * 64 * 4 bytes
buf_s4d1:    .space 1048576     # 4096 * 64 * 4 bytes
buf_s4d2:    .space 1048576     # 4096 * 64 * 4 bytes
buf_pooled:  .space 256         # 64 * 4 bytes
buf_logits:  .space 16          # 4 * 4 bytes

# -----------------------------------------------------------------
# MODEL FORWARD
# Arguments: a0 = image_ptr, a1 = probabilities_ptr, 
#            a2 = weights_ptr, a3 = hilbert_indices_ptr
# -----------------------------------------------------------------
.section .text
.global model_forward

model_forward:
    addi sp, sp, -64
    sw   ra, 60(sp)
    sw   s0, 56(sp)     # weights_ptr base
    sw   s1, 52(sp)     # image_ptr
    sw   s2, 48(sp)     # probabilities_ptr
    sw   s3, 44(sp)     # hilbert_ptr

    mv   s0, a2
    mv   s1, a0
    mv   s2, a1
    mv   s3, a3

    # --- 1 Hilbert Scan ---
    mv   a0, s1
    la   a1, buf_hilbert
    mv   a2, s3
    call hilbert_scan

    # --- 2. Input Projection (Linear) ---
    la   a0, buf_hilbert
    la   a1, buf_proj
    
    li   t0, 16384
    add  a2, s0, t0     # uproject_weight = base + 16384
    li   t0, 16640
    add  a3, s0, t0     # uproject_bias = base + 16640
    
    li   a4, 4096       # batch_size
    li   a5, 1          # in_features
    li   a6, 64         # out_features
    call linear

    # --- 3. S4D Layer 1 --
    la   a0, buf_proj
    la   a1, buf_s4d1
    
    li   t0, 16896
    add  a2, s0, t0     # log_dt = base + 16896
    li   t0, 17152
    add  a3, s0, t0     # log_A_real = base + 17152
    li   t0, 25344
    add  a4, s0, t0     # A_imag = base + 25344
    li   t0, 33536
    add  a5, s0, t0     # C_real = base + 33536
    li   t0, 33540
    add  a6, s0, t0     # C_imag = base + 33540 (Offset by 4 bytes for stride-2)
    li   t0, 49920
    add  a7, s0, t0     # D = base + 49920
    
    call s4d_layer

    # --- 4. GELU 1 ---
    la   a0, buf_s4d1
    li   a1, 262144     # 4096 * 64 elements
    call gelu

    # --- 5. S4D Layer 2 ---
    la   a0, buf_s4d1
    la   a1, buf_s4d2
    
    li   t0, 50176
    add  a2, s0, t0     # log_dt = base + 50176
    li   t0, 50432
    add  a3, s0, t0     # log_A_real = base + 50432
    li   t0, 58624
    add  a4, s0, t0     # A_imag = base + 58624
    li   t0, 66816
    add  a5, s0, t0     # C_real = base + 66816
    li   t0, 66820
    add  a6, s0, t0     # C_imag = base + 66820 (Offset by 4 bytes)
    li   t0, 83200
    add  a7, s0, t0     # D = base + 83200
    
    call s4d_layer

    # --- 6. GELU 2 ---
    la   a0, buf_s4d2
    li   a1, 262144
    call gelu

    # --- 7. Take Last Timestamp ---
    la   a0, buf_s4d2
    la   a1, buf_pooled
    call take_last_timestamp

    # --- 8. Final FC (Linear) ---
    la   a0, buf_pooled
    la   a1, buf_logits
    
    li   t0, 83456
    add  a2, s0, t0     # fc_weight = base + 83456
    li   t0, 84480
    add  a3, s0, t0     # fc_bias = base + 84480
    
    li   a4, 1          # batch_size
    li   a5, 64         # in_features
    li   a6, 4          # out_features
    call linear

    # --- 9. Softmax ---
    la   a0, buf_logits
    li   a1, 4          # N_CLASSES
    call softmax

    # --- 10. Copy logits to output probabilities array ---
    la   t0, buf_logits
    flw  ft0, 0(t0)
    flw  ft1, 4(t0)
    flw  ft2, 8(t0)
    flw  ft3, 12(t0)
    
    fsw  ft0, 0(s2)
    fsw  ft1, 4(s2)
    fsw  ft2, 8(s2)
    fsw  ft3, 12(s2)

    # --- finality ---
    lw   ra, 60(sp)
    lw   s0, 56(sp)
    lw   s1, 52(sp)
    lw   s2, 48(sp)
    lw   s3, 44(sp)
    addi sp, sp, 64
    ret
