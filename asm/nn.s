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
