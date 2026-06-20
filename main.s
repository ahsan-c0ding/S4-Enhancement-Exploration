.section .data
output_probs: .space 16

.section .text
.global _start
_start:
    la a0, image_data         
    la a1, output_probs       
    la a2, weights_data       
    la a3, weights_data       
    call model_forward

find_argmax:
    la t0, output_probs
    flw ft0, 0(t0)
    flw ft1, 4(t0)
    flw ft2, 8(t0)
    flw ft3, 12(t0)

    fmv.s fs0, ft0
    li s1, 0
    fle.s t1, ft1, fs0
    bnez t1, check_2
    fmv.s fs0, ft1
    li s1, 1
check_2:
    fle.s t1, ft2, fs0
    bnez t1, check_3
    fmv.s fs0, ft2
    li s1, 2
check_3:
    fle.s t1, ft3, fs0
    bnez t1, store_answer
    fmv.s fs0, ft3
    li s1, 3
store_answer:
    # Clean Linux Exit (Syscall 93)
    li a7, 93               
    li a0, 0                
    ecall