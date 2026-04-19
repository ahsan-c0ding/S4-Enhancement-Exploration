.section .data
output_probs: .space 16
print_buf:    .space 9      # Buffer for 8 hex chars + 1 newline

.section .text
.global _start
_start:
    la a0, image_data         
    la a1, output_probs       
    la a2, weights_data       
    la a3, weights_data       
    call model_forward

    # --------------------------------------------------------
    # DATA EXTRACTION DUMPING ALL LAYERS
    # --------------------------------------------------------
    
    # 1. Hilbert Scan (4,096 floats)
    la a0, buf_hilbert
    li a1, 4096
    call print_array

    # 2. Input Projection (Linear 1) (262,144 floats)
    la a0, buf_proj
    li a1, 262144
    call print_array

    # 3. S4D 1 + GELU 1 (262,144 floats)
    la a0, buf_s4d1
    li a1, 262144
    call print_array

    # 4. S4D 2 + GELU 2 (262,144 floats)
    la a0, buf_s4d2
    li a1, 262144
    call print_array

    # 5. TakeLastTimestep (64 floats)
    la a0, buf_pooled
    li a1, 64
    call print_array

    # 6. Softmax Logits (4 floats)
    la a0, buf_logits
    li a1, 4
    call print_array

    # Clean Linux Exit (Syscall 93)
    li a7, 93               
    li a0, 0                
    ecall                   

# --------------------------------------------------------
#  PRINT FUNCTIONS
# --------------------------------------------------------

print_array:
    addi sp, sp, -16
    sw ra, 12(sp)
    sw s0, 8(sp)
    sw s1, 4(sp)
    
    mv s0, a0               
    mv s1, a1               
    li t5, 0                
    
.L_print_loop:
    bge t5, s1, .L_print_done
    lw a0, 0(s0)            
    call print_hex_syscall          
    
    addi s0, s0, 4          
    addi t5, t5, 1          
    j .L_print_loop
    
.L_print_done:
    lw ra, 12(sp)
    lw s0, 8(sp)
    lw s1, 4(sp)
    addi sp, sp, 16
    ret

print_hex_syscall:
    addi sp, sp, -16
    sw ra, 12(sp)
    
    la t3, print_buf
    li t1, 28               
    li t2, 0                
    
.L_hex_loop:
    blt t1, zero, .L_hex_done
    srl t4, a0, t1          
    andi t4, t4, 0xF        
    
    li t5, 9
    bgt t4, t5, .L_letter
    addi t4, t4, 48         
    j .L_store
.L_letter:
    addi t4, t4, 55         
.L_store:
    add t6, t3, t2
    sb t4, 0(t6)            
    
    addi t1, t1, -4         
    addi t2, t2, 1
    j .L_hex_loop
    
.L_hex_done:
    li t4, 10
    add t6, t3, t2
    sb t4, 0(t6)
    
    li a7, 64               
    li a0, 1                
    la a1, print_buf        
    li a2, 9                
    ecall                   
    
    lw ra, 12(sp)
    addi sp, sp, 16
    ret