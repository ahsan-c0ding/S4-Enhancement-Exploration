	.file	"nn.c"
	.text
	.p2align 4
	.globl	hilbert_scan
	.type	hilbert_scan, @function
hilbert_scan:
.LFB25:
	.cfi_startproc
	endbr64
	xorl	%r8d, %r8d
	jmp	.L3
	.p2align 4,,10
	.p2align 3
.L8:
	movl	%ecx, %eax
	andl	$63, %ecx
	sarl	$6, %eax
.L2:
	cltq
	salq	$6, %rax
	addq	%rcx, %rax
	movss	(%rdi,%rax,4), %xmm0
	movss	%xmm0, (%rsi,%r8)
	addq	$4, %r8
	cmpq	$16384, %r8
	je	.L7
.L3:
	movl	(%rdx,%r8), %ecx
	cmpl	$4095, %ecx
	jbe	.L8
	xorl	%ecx, %ecx
	xorl	%eax, %eax
	jmp	.L2
.L7:
	ret
	.cfi_endproc
.LFE25:
	.size	hilbert_scan, .-hilbert_scan
	.p2align 4
	.globl	linear_uproject
	.type	linear_uproject, @function
linear_uproject:
.LFB26:
	.cfi_startproc
	endbr64
	movq	%rdx, %r9
	movq	%rcx, %r8
	movq	%rsi, %rdx
	movq	%rdi, %rcx
	leaq	16384(%rdi), %rsi
.L10:
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L11:
	movss	(%r8,%rax), %xmm1
	movss	%xmm1, (%rdx,%rax)
	movss	(%rcx), %xmm0
	mulss	(%r9,%rax), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rdx,%rax)
	addq	$4, %rax
	cmpq	$256, %rax
	jne	.L11
	addq	$4, %rcx
	addq	$256, %rdx
	cmpq	%rsi, %rcx
	jne	.L10
	ret
	.cfi_endproc
.LFE26:
	.size	linear_uproject, .-linear_uproject
	.p2align 4
	.globl	linear_fc
	.type	linear_fc, @function
linear_fc:
.LFB27:
	.cfi_startproc
	endbr64
	leaq	16(%rsi), %r8
.L16:
	movss	(%rcx), %xmm1
	xorl	%eax, %eax
	movss	%xmm1, (%rsi)
	.p2align 4,,10
	.p2align 3
.L15:
	movss	(%rdi,%rax), %xmm0
	mulss	(%rdx,%rax), %xmm0
	addq	$4, %rax
	addss	%xmm0, %xmm1
	movss	%xmm1, (%rsi)
	cmpq	$256, %rax
	jne	.L15
	addq	$4, %rsi
	addq	$4, %rcx
	addq	$256, %rdx
	cmpq	%r8, %rsi
	jne	.L16
	ret
	.cfi_endproc
.LFE27:
	.size	linear_fc, .-linear_fc
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC0:
	.string	"\rProcessing S4D channel %d out of 64..."
	.text
	.p2align 4
	.globl	s4d_layer
	.type	s4d_layer, @function
s4d_layer:
.LFB28:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	leaq	-16384(%rsp), %r11
	.cfi_def_cfa 11, 16440
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	.cfi_def_cfa_register 7
	subq	$696, %rsp
	.cfi_def_cfa_offset 17136
	movq	17136(%rsp), %rax
	movq	%rcx, 104(%rsp)
	xorl	%ecx, %ecx
	leaq	416(%rsp), %r13
	leaq	544(%rsp), %r12
	movq	%r8, 112(%rsp)
	leaq	672(%rsp), %r8
	leaq	160(%rsp), %rbp
	movq	%rcx, %r11
	movq	%r9, 120(%rsp)
	movq	17144(%rsp), %r9
	leaq	288(%rsp), %rbx
	movq	%rdi, 80(%rsp)
	movq	%rsi, 88(%rsp)
	movq	%rdx, 96(%rsp)
	movq	%rax, 128(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 17064(%rsp)
	xorl	%eax, %eax
	leaq	676(%rsp), %rax
	movq	%r9, 144(%rsp)
	movq	%rax, 136(%rsp)
	movq	%r8, 72(%rsp)
.L25:
	leal	1(%r11), %edx
	leaq	.LC0(%rip), %rsi
	xorl	%eax, %eax
	xorl	%r14d, %r14d
	movl	$2, %edi
	movq	%r11, 8(%rsp)
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	call	fflush@PLT
	movq	96(%rsp), %rax
	movq	8(%rsp), %r11
	movss	(%rax,%r11,4), %xmm0
	call	my_exp@PLT
	movq	8(%rsp), %r11
	movq	104(%rsp), %rax
	movq	%rbp, 48(%rsp)
	movq	%r12, 56(%rsp)
	movq	%r11, %r15
	movq	%r11, %rdx
	movq	%r11, 152(%rsp)
	salq	$7, %r15
	salq	$8, %rdx
	movss	%xmm0, 36(%rsp)
	leaq	(%rax,%r15), %rsi
	movq	112(%rsp), %rax
	addq	%r15, %rax
	movq	%rax, 64(%rsp)
	movq	120(%rsp), %rax
	leaq	(%rax,%rdx), %rcx
	movq	128(%rsp), %rax
	movq	%rcx, %rbp
	leaq	(%rax,%rdx), %r15
	movq	%r15, %r12
	movq	%r14, %r15
	movq	%rbx, %r14
	movq	%rsi, %rbx
.L20:
	movss	(%rbx,%r15), %xmm0
	call	my_exp@PLT
	movq	64(%rsp), %rax
	movss	36(%rsp), %xmm6
	movaps	%xmm0, %xmm1
	xorps	.LC1(%rip), %xmm1
	movss	%xmm0, 40(%rsp)
	movss	(%rax,%r15), %xmm3
	movaps	%xmm6, %xmm7
	movq	56(%rsp), %rax
	mulss	%xmm1, %xmm6
	movss	%xmm1, 0(%r13,%r15)
	mulss	%xmm3, %xmm7
	movss	%xmm3, (%rax,%r15)
	movss	%xmm3, 32(%rsp)
	movss	%xmm1, 28(%rsp)
	movaps	%xmm6, %xmm0
	movss	%xmm7, 8(%rsp)
	call	my_exp@PLT
	movss	%xmm0, 20(%rsp)
	movss	8(%rsp), %xmm0
	call	my_cos@PLT
	movss	20(%rsp), %xmm4
	mulss	%xmm0, %xmm4
	movss	8(%rsp), %xmm0
	movss	%xmm4, 24(%rsp)
	call	my_sin@PLT
	movss	32(%rsp), %xmm3
	movq	48(%rsp), %rax
	movss	20(%rsp), %xmm5
	movss	40(%rsp), %xmm2
	movss	28(%rsp), %xmm1
	movss	24(%rsp), %xmm4
	movaps	%xmm3, %xmm6
	mulss	%xmm0, %xmm5
	movaps	%xmm3, %xmm0
	subss	.LC2(%rip), %xmm4
	mulss	%xmm3, %xmm0
	mulss	%xmm2, %xmm2
	mulss	%xmm4, %xmm3
	mulss	%xmm5, %xmm6
	addss	%xmm0, %xmm2
	movaps	%xmm1, %xmm0
	mulss	%xmm4, %xmm0
	movss	0(%rbp,%r15,2), %xmm4
	mulss	%xmm5, %xmm1
	addss	%xmm6, %xmm0
	subss	%xmm3, %xmm1
	movss	(%r12,%r15,2), %xmm3
	divss	%xmm2, %xmm0
	divss	%xmm2, %xmm1
	movaps	%xmm0, %xmm2
	mulss	%xmm4, %xmm2
	mulss	%xmm3, %xmm0
	movaps	%xmm1, %xmm5
	mulss	%xmm3, %xmm5
	mulss	%xmm4, %xmm1
	subss	%xmm5, %xmm2
	addss	%xmm1, %xmm0
	movss	%xmm2, (%rax,%r15)
	movss	%xmm0, (%r14,%r15)
	addq	$4, %r15
	cmpq	$128, %r15
	jne	.L20
	movq	152(%rsp), %r11
	movq	%r14, %rbx
	movq	72(%rsp), %r14
	movq	%rax, %rbp
	movq	56(%rsp), %r12
	xorl	%eax, %eax
	movq	%r11, 40(%rsp)
	movq	%r14, %r15
	.p2align 4,,10
	.p2align 3
.L22:
	pxor	%xmm2, %xmm2
	xorl	%r14d, %r14d
	movl	%eax, 32(%rsp)
	cvtsi2ssl	%eax, %xmm2
	mulss	36(%rsp), %xmm2
	movq	%r15, %rax
	movl	$0x00000000, (%r15)
	movq	%r14, %r15
	movq	%rax, %r14
	.p2align 4,,10
	.p2align 3
.L21:
	movss	(%r12,%r15), %xmm7
	movaps	%xmm2, %xmm5
	movss	%xmm2, 28(%rsp)
	mulss	0(%r13,%r15), %xmm5
	mulss	%xmm2, %xmm7
	movaps	%xmm5, %xmm0
	movss	%xmm7, 8(%rsp)
	call	my_exp@PLT
	movss	%xmm0, 20(%rsp)
	movss	8(%rsp), %xmm0
	call	my_cos@PLT
	movss	20(%rsp), %xmm4
	mulss	%xmm0, %xmm4
	movss	8(%rsp), %xmm0
	movss	%xmm4, 24(%rsp)
	call	my_sin@PLT
	movss	24(%rsp), %xmm1
	movss	28(%rsp), %xmm2
	movaps	%xmm0, %xmm3
	movss	0(%rbp,%r15), %xmm0
	mulss	%xmm1, %xmm0
	movss	20(%rsp), %xmm1
	mulss	%xmm3, %xmm1
	mulss	(%rbx,%r15), %xmm1
	addq	$4, %r15
	cmpq	$128, %r15
	subss	%xmm1, %xmm0
	addss	%xmm0, %xmm0
	addss	(%r14), %xmm0
	movss	%xmm0, (%r14)
	jne	.L21
	movl	32(%rsp), %eax
	movq	%r14, %r15
	addq	$4, %r15
	addl	$1, %eax
	cmpl	$4096, %eax
	jne	.L22
	movq	40(%rsp), %r11
	movq	80(%rsp), %rax
	xorl	%r10d, %r10d
	movq	136(%rsp), %rsi
	movq	144(%rsp), %r9
	leaq	0(,%r11,4), %rcx
	movq	72(%rsp), %r8
	leaq	(%rax,%rcx), %rdi
	movq	88(%rsp), %rax
	addq	%rax, %rcx
	.p2align 4,,10
	.p2align 3
.L24:
	movss	(%r9,%r11,4), %xmm1
	mulss	(%rdi), %xmm1
	movq	%rdi, %rdx
	movq	%r8, %rax
	movss	%xmm1, (%rcx)
	.p2align 4,,10
	.p2align 3
.L23:
	movss	(%rax), %xmm0
	mulss	(%rdx), %xmm0
	addq	$4, %rax
	subq	$256, %rdx
	addss	%xmm0, %xmm1
	movss	%xmm1, (%rcx)
	cmpq	%rsi, %rax
	jne	.L23
	addl	$1, %r10d
	addq	$256, %rdi
	addq	$256, %rcx
	leaq	4(%rax), %rsi
	cmpl	$4096, %r10d
	jne	.L24
	addq	$1, %r11
	cmpq	$64, %r11
	jne	.L25
	movq	17064(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L34
	addq	$17080, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L34:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE28:
	.size	s4d_layer, .-s4d_layer
	.p2align 4
	.globl	gelu
	.type	gelu, @function
gelu:
.LFB29:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movslq	%esi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movq	%rdi, %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	movss	.LC4(%rip), %xmm0
	call	my_sqrt@PLT
	movss	%xmm0, 12(%rsp)
	testl	%ebp, %ebp
	jle	.L35
	leaq	(%rbx,%rbp,4), %rbp
	.p2align 4,,10
	.p2align 3
.L37:
	movss	(%rbx), %xmm1
	addq	$4, %rbx
	movaps	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	.LC5(%rip), %xmm0
	addss	%xmm1, %xmm0
	mulss	12(%rsp), %xmm0
	call	my_tanh@PLT
	movss	.LC6(%rip), %xmm1
	mulss	-4(%rbx), %xmm1
	addss	.LC2(%rip), %xmm0
	mulss	%xmm0, %xmm1
	movss	%xmm1, -4(%rbx)
	cmpq	%rbx, %rbp
	jne	.L37
.L35:
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE29:
	.size	gelu, .-gelu
	.p2align 4
	.globl	softmax
	.type	softmax, @function
softmax:
.LFB30:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	movss	(%rdi), %xmm1
	cmpl	$1, %esi
	jle	.L41
	leal	-2(%rsi), %edx
	leaq	4(%rdi), %rax
	leaq	8(%rdi,%rdx,4), %rdx
	.p2align 4,,10
	.p2align 3
.L43:
	movss	(%rax), %xmm0
	addq	$4, %rax
	maxss	%xmm1, %xmm0
	movaps	%xmm0, %xmm1
	cmpq	%rdx, %rax
	jne	.L43
.L44:
	movslq	%esi, %rsi
	movq	%rbx, %rbp
	pxor	%xmm2, %xmm2
	leaq	(%rbx,%rsi,4), %r12
	.p2align 4,,10
	.p2align 3
.L46:
	movss	0(%rbp), %xmm0
	movss	%xmm2, 12(%rsp)
	addq	$4, %rbp
	movss	%xmm1, 8(%rsp)
	subss	%xmm1, %xmm0
	call	my_exp@PLT
	movss	12(%rsp), %xmm2
	movss	8(%rsp), %xmm1
	movss	%xmm0, -4(%rbp)
	cmpq	%r12, %rbp
	addss	%xmm0, %xmm2
	jne	.L46
	.p2align 4,,10
	.p2align 3
.L47:
	movss	(%rbx), %xmm0
	addq	$4, %rbx
	divss	%xmm2, %xmm0
	movss	%xmm0, -4(%rbx)
	cmpq	%r12, %rbx
	jne	.L47
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L41:
	.cfi_restore_state
	je	.L44
	addq	$16, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE30:
	.size	softmax, .-softmax
	.p2align 4
	.globl	take_last_timestamp
	.type	take_last_timestamp, @function
take_last_timestamp:
.LFB31:
	.cfi_startproc
	endbr64
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L55:
	movss	1048320(%rdi,%rax), %xmm0
	movss	%xmm0, (%rsi,%rax)
	addq	$4, %rax
	cmpq	$256, %rax
	jne	.L55
	ret
	.cfi_endproc
.LFE31:
	.size	take_last_timestamp, .-take_last_timestamp
	.p2align 4
	.globl	model_forward
	.type	model_forward, @function
model_forward:
.LFB32:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdi, %r8
	leaq	17152(%rdx), %rdi
	movq	%rdx, %rax
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	addq	$84480, %rax
	leaq	16384(%rdx), %r9
	leaq	16640(%rdx), %r10
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	leaq	16896(%rdx), %r14
	leaq	33540(%rdx), %r11
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	leaq	66820(%rdx), %r13
	leaq	33536(%rdx), %r12
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	leaq	83200(%rdx), %r15
	leaq	49920(%rdx), %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, %rbx
	movq	%rcx, %rsi
	xorl	%ecx, %ecx
	subq	$72, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, (%rsp)
	leaq	25344(%rdx), %rdi
	movq	%rdi, 8(%rsp)
	leaq	50176(%rdx), %rdi
	movq	%rdi, 16(%rsp)
	leaq	50432(%rdx), %rdi
	movq	%rdi, 24(%rsp)
	leaq	58624(%rdx), %rdi
	movq	%rdi, 32(%rsp)
	leaq	66816(%rdx), %rdi
	movq	%rdi, 40(%rsp)
	leaq	83456(%rdx), %rdi
	movq	%rdi, 48(%rsp)
	leaq	hilbert_out.5(%rip), %rdi
	movq	%rax, 56(%rsp)
	jmp	.L59
	.p2align 4,,10
	.p2align 3
.L73:
	movl	%edx, %eax
	andl	$63, %edx
	sarl	$6, %eax
.L58:
	cltq
	salq	$6, %rax
	addq	%rdx, %rax
	movss	(%r8,%rax,4), %xmm0
	movss	%xmm0, (%rdi,%rcx)
	addq	$4, %rcx
	cmpq	$16384, %rcx
	je	.L72
.L59:
	movl	(%rsi,%rcx), %edx
	cmpl	$4095, %edx
	jbe	.L73
	xorl	%edx, %edx
	xorl	%eax, %eax
	jmp	.L58
.L72:
	movq	%r10, %rcx
	leaq	proj_out.4(%rip), %r10
	movq	%r9, %rdx
	movq	%r10, %rsi
	call	linear_uproject
	pushq	%rbp
	.cfi_def_cfa_offset 136
	movq	%r12, %r9
	leaq	s4d1_out.3(%rip), %r12
	pushq	%r11
	.cfi_def_cfa_offset 144
	movq	24(%rsp), %r8
	movq	%r14, %rdx
	movq	%r12, %rsi
	movq	16(%rsp), %rcx
	movq	%r10, %rdi
	movq	%r12, %r14
	leaq	1048576(%r12), %rbp
	call	s4d_layer
	movss	.LC4(%rip), %xmm0
	call	my_sqrt@PLT
	movss	%xmm0, 16(%rsp)
	popq	%rcx
	.cfi_def_cfa_offset 136
	popq	%rsi
	.cfi_def_cfa_offset 128
	.p2align 4,,10
	.p2align 3
.L60:
	movss	(%r14), %xmm1
	addq	$4, %r14
	movaps	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	.LC5(%rip), %xmm0
	addss	%xmm1, %xmm0
	mulss	(%rsp), %xmm0
	call	my_tanh@PLT
	movaps	%xmm0, %xmm1
	movss	.LC6(%rip), %xmm0
	addss	.LC2(%rip), %xmm1
	mulss	-4(%r14), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -4(%r14)
	cmpq	%rbp, %r14
	jne	.L60
	pushq	%r15
	.cfi_def_cfa_offset 136
	leaq	s4d2_out.2(%rip), %rbp
	movq	%r12, %rdi
	pushq	%r13
	.cfi_def_cfa_offset 144
	movq	56(%rsp), %r9
	movq	%rbp, %rsi
	movq	%rbp, %r12
	movq	48(%rsp), %r8
	movq	40(%rsp), %rcx
	addq	$1048576, %rbp
	movq	32(%rsp), %rdx
	call	s4d_layer
	movss	.LC4(%rip), %xmm0
	call	my_sqrt@PLT
	movss	%xmm0, 16(%rsp)
	popq	%rax
	.cfi_def_cfa_offset 136
	popq	%rdx
	.cfi_def_cfa_offset 128
	.p2align 4,,10
	.p2align 3
.L61:
	movss	(%r12), %xmm1
	addq	$4, %r12
	movaps	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	mulss	.LC5(%rip), %xmm0
	addss	%xmm1, %xmm0
	mulss	(%rsp), %xmm0
	call	my_tanh@PLT
	movaps	%xmm0, %xmm1
	movss	.LC6(%rip), %xmm0
	addss	.LC2(%rip), %xmm1
	mulss	-4(%r12), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -4(%r12)
	cmpq	%rbp, %r12
	jne	.L61
	movdqa	1048320+s4d2_out.2(%rip), %xmm7
	movdqa	1048352+s4d2_out.2(%rip), %xmm3
	leaq	logits.0(%rip), %rsi
	leaq	pooled.1(%rip), %rdi
	movdqa	1048368+s4d2_out.2(%rip), %xmm2
	movq	%rbx, %rbp
	leaq	16(%rbx), %r12
	movaps	%xmm7, pooled.1(%rip)
	movdqa	1048336+s4d2_out.2(%rip), %xmm7
	movaps	%xmm3, 32+pooled.1(%rip)
	movdqa	1048400+s4d2_out.2(%rip), %xmm3
	movaps	%xmm7, 16+pooled.1(%rip)
	movdqa	1048384+s4d2_out.2(%rip), %xmm7
	movaps	%xmm2, 48+pooled.1(%rip)
	movdqa	1048416+s4d2_out.2(%rip), %xmm2
	movaps	%xmm7, 64+pooled.1(%rip)
	movdqa	1048432+s4d2_out.2(%rip), %xmm7
	movaps	%xmm3, 80+pooled.1(%rip)
	movdqa	1048448+s4d2_out.2(%rip), %xmm3
	movaps	%xmm2, 96+pooled.1(%rip)
	movdqa	1048464+s4d2_out.2(%rip), %xmm2
	movaps	%xmm7, 112+pooled.1(%rip)
	movdqa	1048480+s4d2_out.2(%rip), %xmm7
	movaps	%xmm3, 128+pooled.1(%rip)
	movdqa	1048496+s4d2_out.2(%rip), %xmm3
	movaps	%xmm2, 144+pooled.1(%rip)
	movdqa	1048512+s4d2_out.2(%rip), %xmm2
	movaps	%xmm7, 160+pooled.1(%rip)
	movdqa	1048528+s4d2_out.2(%rip), %xmm7
	movaps	%xmm3, 176+pooled.1(%rip)
	movdqa	1048544+s4d2_out.2(%rip), %xmm3
	movaps	%xmm2, 192+pooled.1(%rip)
	movdqa	1048560+s4d2_out.2(%rip), %xmm2
	movaps	%xmm3, 224+pooled.1(%rip)
	movaps	%xmm2, 240+pooled.1(%rip)
	movaps	%xmm7, 208+pooled.1(%rip)
	movq	56(%rsp), %rcx
	movq	48(%rsp), %rdx
	call	linear_fc
	movss	logits.0(%rip), %xmm3
	movss	%xmm3, (%rbx)
	movss	4+logits.0(%rip), %xmm1
	movss	%xmm1, 4(%rbx)
	maxss	%xmm3, %xmm1
	movss	8+logits.0(%rip), %xmm0
	movss	%xmm0, 8(%rbx)
	movss	12+logits.0(%rip), %xmm2
	maxss	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	movss	%xmm2, 12(%rbx)
	maxss	%xmm0, %xmm2
	movss	%xmm2, 8(%rsp)
	.p2align 4,,10
	.p2align 3
.L65:
	movss	0(%rbp), %xmm0
	subss	8(%rsp), %xmm0
	movss	%xmm1, (%rsp)
	addq	$4, %rbp
	call	my_exp@PLT
	movss	(%rsp), %xmm1
	movss	%xmm0, -4(%rbp)
	addss	%xmm0, %xmm1
	cmpq	%rbp, %r12
	jne	.L65
	movups	(%rbx), %xmm0
	shufps	$0, %xmm1, %xmm1
	divps	%xmm1, %xmm0
	movups	%xmm0, (%rbx)
	addq	$72, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE32:
	.size	model_forward, .-model_forward
	.local	logits.0
	.comm	logits.0,16,16
	.local	pooled.1
	.comm	pooled.1,256,32
	.local	s4d2_out.2
	.comm	s4d2_out.2,1048576,32
	.local	s4d1_out.3
	.comm	s4d1_out.3,1048576,32
	.local	proj_out.4
	.comm	proj_out.4,1048576,32
	.local	hilbert_out.5
	.comm	hilbert_out.5,16384,32
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC1:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC2:
	.long	1065353216
	.align 4
.LC4:
	.long	1059256707
	.align 4
.LC5:
	.long	1027024659
	.align 4
.LC6:
	.long	1056964608
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
