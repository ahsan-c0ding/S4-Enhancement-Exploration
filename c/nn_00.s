	.file	"nn.c"
	.text
	.type	complex_mul, @function
complex_mul:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -4(%rbp)
	movss	%xmm1, -8(%rbp)
	movss	%xmm2, -12(%rbp)
	movss	%xmm3, -16(%rbp)
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movss	-4(%rbp), %xmm0
	mulss	-12(%rbp), %xmm0
	movss	-8(%rbp), %xmm1
	mulss	-16(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movq	-24(%rbp), %rax
	movss	%xmm0, (%rax)
	movss	-4(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-16(%rbp), %xmm1
	movss	-8(%rbp), %xmm0
	mulss	-12(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	complex_mul, .-complex_mul
	.type	complex_exp, @function
complex_exp:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	%rdi, -32(%rbp)
	movq	%rsi, -40(%rbp)
	movl	-20(%rbp), %eax
	movd	%eax, %xmm0
	call	my_exp@PLT
	movd	%xmm0, %eax
	movl	%eax, -4(%rbp)
	movl	-24(%rbp), %eax
	movd	%eax, %xmm0
	call	my_cos@PLT
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	mulss	-4(%rbp), %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	movl	-24(%rbp), %eax
	movd	%eax, %xmm0
	call	my_sin@PLT
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	mulss	-4(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movss	%xmm0, (%rax)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	complex_exp, .-complex_exp
	.globl	hilbert_scan
	.type	hilbert_scan, @function
hilbert_scan:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	$0, -20(%rbp)
	jmp	.L4
.L9:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	cmpl	$0, -16(%rbp)
	js	.L5
	cmpl	$4095, -16(%rbp)
	jle	.L6
.L5:
	movl	$0, -16(%rbp)
.L6:
	movl	-16(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, -8(%rbp)
	movl	-16(%rbp), %edx
	movl	%edx, %eax
	sarl	$31, %eax
	shrl	$26, %eax
	addl	%eax, %edx
	andl	$63, %edx
	subl	%eax, %edx
	movl	%edx, -4(%rbp)
	movl	$0, -12(%rbp)
	jmp	.L7
.L8:
	movl	-12(%rbp), %eax
	cltq
	salq	$14, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	movl	-4(%rbp), %ecx
	movslq	%ecx, %rcx
	movl	-8(%rbp), %esi
	movslq	%esi, %rsi
	salq	$6, %rsi
	addq	%rsi, %rcx
	movss	(%rax,%rcx,4), %xmm0
	movl	-12(%rbp), %eax
	cltq
	movss	%xmm0, (%rdx,%rax,4)
	addl	$1, -12(%rbp)
.L7:
	cmpl	$0, -12(%rbp)
	jle	.L8
	addl	$1, -20(%rbp)
.L4:
	cmpl	$4095, -20(%rbp)
	jle	.L9
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	hilbert_scan, .-hilbert_scan
	.globl	linear_uproject
	.type	linear_uproject, @function
linear_uproject:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	%rcx, -48(%rbp)
	movl	$0, -12(%rbp)
	jmp	.L11
.L16:
	movl	$0, -8(%rbp)
	jmp	.L12
.L15:
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$8, %rcx
	movq	-32(%rbp), %rdx
	addq	%rcx, %rdx
	movss	(%rax), %xmm0
	movl	-8(%rbp), %eax
	cltq
	movss	%xmm0, (%rdx,%rax,4)
	movl	$0, -4(%rbp)
	jmp	.L13
.L14:
	movl	-12(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movl	-8(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm1
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-4(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm2
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-4(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm0
	mulss	%xmm2, %xmm0
	movl	-12(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	addss	%xmm1, %xmm0
	movl	-8(%rbp), %eax
	cltq
	movss	%xmm0, (%rdx,%rax,4)
	addl	$1, -4(%rbp)
.L13:
	cmpl	$0, -4(%rbp)
	jle	.L14
	addl	$1, -8(%rbp)
.L12:
	cmpl	$63, -8(%rbp)
	jle	.L15
	addl	$1, -12(%rbp)
.L11:
	cmpl	$4095, -12(%rbp)
	jle	.L16
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	linear_uproject, .-linear_uproject
	.globl	linear_fc
	.type	linear_fc, @function
linear_fc:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	%rcx, -48(%rbp)
	movl	$0, -8(%rbp)
	jmp	.L18
.L21:
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-32(%rbp), %rax
	addq	%rcx, %rax
	movss	(%rdx), %xmm0
	movss	%xmm0, (%rax)
	movl	$0, -4(%rbp)
	jmp	.L19
.L20:
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm2
	movl	-8(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-4(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm0
	mulss	%xmm2, %xmm0
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	addss	%xmm1, %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -4(%rbp)
.L19:
	cmpl	$63, -4(%rbp)
	jle	.L20
	addl	$1, -8(%rbp)
.L18:
	cmpl	$3, -8(%rbp)
	jle	.L21
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	linear_fc, .-linear_fc
	.section	.rodata
	.align 8
.LC0:
	.string	"\rProcessing S4D channel %d out of 64..."
	.text
	.globl	s4d_layer
	.type	s4d_layer, @function
s4d_layer:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-16384(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$688, %rsp
	movq	%rdi, -17016(%rbp)
	movq	%rsi, -17024(%rbp)
	movq	%rdx, -17032(%rbp)
	movq	%rcx, -17040(%rbp)
	movq	%r8, -17048(%rbp)
	movq	%r9, -17056(%rbp)
	movq	16(%rbp), %rax
	movq	%rax, -17064(%rbp)
	movq	24(%rbp), %rax
	movq	%rax, -17072(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$32, -16960(%rbp)
	movl	$0, -16984(%rbp)
	jmp	.L23
.L34:
	movl	-16984(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movl	-16984(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-17032(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movd	%eax, %xmm0
	call	my_exp@PLT
	movd	%xmm0, %eax
	movl	%eax, -16956(%rbp)
	movl	$0, -16980(%rbp)
	jmp	.L24
.L25:
	movl	-16984(%rbp), %eax
	imull	-16960(%rbp), %eax
	movl	%eax, %edx
	movl	-16980(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-17040(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movd	%eax, %xmm0
	call	my_exp@PLT
	movd	%xmm0, %eax
	movss	.LC1(%rip), %xmm0
	movd	%eax, %xmm4
	xorps	%xmm0, %xmm4
	movaps	%xmm4, %xmm0
	movss	%xmm0, -16948(%rbp)
	movl	-16984(%rbp), %eax
	imull	-16960(%rbp), %eax
	movl	%eax, %edx
	movl	-16980(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-17048(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -16944(%rbp)
	movl	-16980(%rbp), %eax
	cltq
	movss	-16948(%rbp), %xmm0
	movss	%xmm0, -16656(%rbp,%rax,4)
	movl	-16980(%rbp), %eax
	cltq
	movss	-16944(%rbp), %xmm0
	movss	%xmm0, -16528(%rbp,%rax,4)
	movss	-16944(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-16956(%rbp), %xmm1
	movss	-16948(%rbp), %xmm0
	mulss	-16956(%rbp), %xmm0
	movd	%xmm0, %eax
	leaq	-16988(%rbp), %rcx
	leaq	-16992(%rbp), %rdx
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movd	%eax, %xmm0
	call	complex_exp
	movss	-16992(%rbp), %xmm0
	movss	.LC2(%rip), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -16940(%rbp)
	movss	-16988(%rbp), %xmm0
	movss	%xmm0, -16936(%rbp)
	movss	-16948(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	%xmm0, %xmm1
	movss	-16944(%rbp), %xmm0
	mulss	%xmm0, %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -16932(%rbp)
	movss	-16940(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-16948(%rbp), %xmm1
	movss	-16936(%rbp), %xmm0
	mulss	-16944(%rbp), %xmm0
	addss	%xmm1, %xmm0
	divss	-16932(%rbp), %xmm0
	movss	%xmm0, -16928(%rbp)
	movss	-16936(%rbp), %xmm0
	mulss	-16948(%rbp), %xmm0
	movss	-16940(%rbp), %xmm1
	mulss	-16944(%rbp), %xmm1
	subss	%xmm1, %xmm0
	divss	-16932(%rbp), %xmm0
	movss	%xmm0, -16924(%rbp)
	movl	-16984(%rbp), %eax
	imull	-16960(%rbp), %eax
	movl	%eax, %edx
	movl	-16980(%rbp), %eax
	addl	%edx, %eax
	addl	%eax, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-17056(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -16920(%rbp)
	movl	-16984(%rbp), %eax
	imull	-16960(%rbp), %eax
	movl	%eax, %edx
	movl	-16980(%rbp), %eax
	addl	%edx, %eax
	addl	%eax, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-17064(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -16916(%rbp)
	leaq	-16784(%rbp), %rax
	movl	-16980(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	leaq	(%rax,%rdx), %rcx
	leaq	-16912(%rbp), %rax
	movl	-16980(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rax, %rdx
	movss	-16924(%rbp), %xmm2
	movss	-16928(%rbp), %xmm1
	movss	-16916(%rbp), %xmm0
	movl	-16920(%rbp), %eax
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movaps	%xmm2, %xmm3
	movaps	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	complex_mul
	addl	$1, -16980(%rbp)
.L24:
	movl	-16980(%rbp), %eax
	cmpl	-16960(%rbp), %eax
	jl	.L25
	movl	$0, -16976(%rbp)
	jmp	.L26
.L29:
	movl	-16976(%rbp), %eax
	cltq
	pxor	%xmm0, %xmm0
	movss	%xmm0, -16400(%rbp,%rax,4)
	movl	$0, -16972(%rbp)
	jmp	.L27
.L28:
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-16976(%rbp), %xmm0
	movss	-16956(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -16952(%rbp)
	movl	-16972(%rbp), %eax
	cltq
	movss	-16528(%rbp,%rax,4), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-16952(%rbp), %xmm1
	movl	-16972(%rbp), %eax
	cltq
	movss	-16656(%rbp,%rax,4), %xmm0
	mulss	-16952(%rbp), %xmm0
	movd	%xmm0, %eax
	leaq	-16996(%rbp), %rcx
	leaq	-17000(%rbp), %rdx
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movd	%eax, %xmm0
	call	complex_exp
	movss	-16996(%rbp), %xmm2
	movss	-17000(%rbp), %xmm1
	movl	-16972(%rbp), %eax
	cltq
	movss	-16784(%rbp,%rax,4), %xmm0
	movl	-16972(%rbp), %eax
	cltq
	movl	-16912(%rbp,%rax,4), %eax
	leaq	-16988(%rbp), %rcx
	leaq	-16992(%rbp), %rdx
	movq	%rcx, %rsi
	movq	%rdx, %rdi
	movaps	%xmm2, %xmm3
	movaps	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	complex_mul
	movl	-16976(%rbp), %eax
	cltq
	movss	-16400(%rbp,%rax,4), %xmm1
	movss	-16992(%rbp), %xmm0
	addss	%xmm0, %xmm0
	addss	%xmm1, %xmm0
	movl	-16976(%rbp), %eax
	cltq
	movss	%xmm0, -16400(%rbp,%rax,4)
	addl	$1, -16972(%rbp)
.L27:
	movl	-16972(%rbp), %eax
	cmpl	-16960(%rbp), %eax
	jl	.L28
	addl	$1, -16976(%rbp)
.L26:
	cmpl	$4095, -16976(%rbp)
	jle	.L29
	movl	$0, -16968(%rbp)
	jmp	.L30
.L33:
	movl	-16984(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-17072(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	-16968(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-17016(%rbp), %rax
	addq	%rax, %rdx
	movl	-16984(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm0
	movl	-16968(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-17024(%rbp), %rax
	addq	%rax, %rdx
	mulss	%xmm1, %xmm0
	movl	-16984(%rbp), %eax
	cltq
	movss	%xmm0, (%rdx,%rax,4)
	movl	$0, -16964(%rbp)
	jmp	.L31
.L32:
	movl	-16968(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-17024(%rbp), %rax
	addq	%rax, %rdx
	movl	-16984(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm1
	movl	-16964(%rbp), %eax
	cltq
	movss	-16400(%rbp,%rax,4), %xmm2
	movl	-16968(%rbp), %eax
	subl	-16964(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-17016(%rbp), %rax
	addq	%rax, %rdx
	movl	-16984(%rbp), %eax
	cltq
	movss	(%rdx,%rax,4), %xmm0
	mulss	%xmm2, %xmm0
	movl	-16968(%rbp), %eax
	cltq
	salq	$8, %rax
	movq	%rax, %rdx
	movq	-17024(%rbp), %rax
	addq	%rax, %rdx
	addss	%xmm1, %xmm0
	movl	-16984(%rbp), %eax
	cltq
	movss	%xmm0, (%rdx,%rax,4)
	addl	$1, -16964(%rbp)
.L31:
	movl	-16964(%rbp), %eax
	cmpl	-16968(%rbp), %eax
	jle	.L32
	addl	$1, -16968(%rbp)
.L30:
	cmpl	$4095, -16968(%rbp)
	jle	.L33
	addl	$1, -16984(%rbp)
.L23:
	cmpl	$63, -16984(%rbp)
	jle	.L34
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L35
	call	__stack_chk_fail@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	s4d_layer, .-s4d_layer
	.globl	gelu
	.type	gelu, @function
gelu:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movss	.LC4(%rip), %xmm0
	movss	%xmm0, -24(%rbp)
	movss	.LC5(%rip), %xmm0
	divss	-24(%rbp), %xmm0
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	call	my_sqrt@PLT
	movd	%xmm0, %eax
	movl	%eax, -20(%rbp)
	movss	.LC6(%rip), %xmm0
	movss	%xmm0, -16(%rbp)
	movl	$0, -28(%rbp)
	jmp	.L37
.L38:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm0, %xmm1
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -12(%rbp)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movss	-16(%rbp), %xmm0
	mulss	-12(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	-20(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -8(%rbp)
	movl	-8(%rbp), %eax
	movd	%eax, %xmm0
	call	my_tanh@PLT
	movd	%xmm0, %eax
	movl	%eax, -4(%rbp)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movss	.LC7(%rip), %xmm0
	mulss	%xmm0, %xmm1
	movss	-4(%rbp), %xmm2
	movss	.LC2(%rip), %xmm0
	addss	%xmm2, %xmm0
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	mulss	%xmm1, %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -28(%rbp)
.L37:
	movl	-28(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L38
	nop
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	gelu, .-gelu
	.globl	softmax
	.type	softmax, @function
softmax:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movq	-56(%rbp), %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -36(%rbp)
	movl	$1, -32(%rbp)
	jmp	.L40
.L43:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	comiss	-36(%rbp), %xmm0
	jbe	.L41
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -36(%rbp)
.L41:
	addl	$1, -32(%rbp)
.L40:
	movl	-32(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jl	.L43
	pxor	%xmm0, %xmm0
	movss	%xmm0, -28(%rbp)
	movl	$0, -24(%rbp)
	jmp	.L44
.L45:
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	subss	-36(%rbp), %xmm0
	movd	%xmm0, %eax
	movl	-24(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	leaq	(%rcx,%rdx), %rbx
	movd	%eax, %xmm0
	call	my_exp@PLT
	movd	%xmm0, %eax
	movl	%eax, (%rbx)
	movl	-24(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-28(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -28(%rbp)
	addl	$1, -24(%rbp)
.L44:
	movl	-24(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jl	.L45
	movl	$0, -20(%rbp)
	jmp	.L46
.L47:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	divss	-28(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -20(%rbp)
.L46:
	movl	-20(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jl	.L47
	nop
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	softmax, .-softmax
	.globl	take_last_timestamp
	.type	take_last_timestamp, @function
take_last_timestamp:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L50
.L51:
	movq	-24(%rbp), %rax
	leaq	1048320(%rax), %rcx
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movl	-4(%rbp), %eax
	cltq
	movss	(%rcx,%rax,4), %xmm0
	movss	%xmm0, (%rdx)
	addl	$1, -4(%rbp)
.L50:
	cmpl	$63, -4(%rbp)
	jle	.L51
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	take_last_timestamp, .-take_last_timestamp
	.globl	model_forward
	.type	model_forward, @function
model_forward:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movq	%rdi, -152(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%rcx, -176(%rbp)
	movl	$0, -132(%rbp)
	movl	-132(%rbp), %eax
	addl	$16384, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -128(%rbp)
	movl	-132(%rbp), %eax
	addl	$256, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -120(%rbp)
	movl	-132(%rbp), %eax
	addl	$256, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -112(%rbp)
	movl	-132(%rbp), %eax
	addl	$256, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -104(%rbp)
	movl	-132(%rbp), %eax
	addl	$8192, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -96(%rbp)
	movl	-132(%rbp), %eax
	addl	$8192, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	addq	$4, %rax
	movq	%rax, -80(%rbp)
	movl	-132(%rbp), %eax
	addl	$16384, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -72(%rbp)
	movl	-132(%rbp), %eax
	addl	$256, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -64(%rbp)
	movl	-132(%rbp), %eax
	addl	$256, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -56(%rbp)
	movl	-132(%rbp), %eax
	addl	$8192, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -48(%rbp)
	movl	-132(%rbp), %eax
	addl	$8192, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	addq	$4, %rax
	movq	%rax, -32(%rbp)
	movl	-132(%rbp), %eax
	addl	$16384, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movl	-132(%rbp), %eax
	addl	$256, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movl	-132(%rbp), %eax
	addl	$1024, %eax
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -8(%rbp)
	movq	-176(%rbp), %rdx
	movq	-152(%rbp), %rax
	leaq	hilbert_out.5(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	hilbert_scan
	movq	-120(%rbp), %rdx
	movq	-128(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	proj_out.4(%rip), %rax
	movq	%rax, %rsi
	leaq	hilbert_out.5(%rip), %rax
	movq	%rax, %rdi
	call	linear_uproject
	movq	-88(%rbp), %rsi
	movq	-96(%rbp), %rcx
	movq	-104(%rbp), %rdx
	movq	-112(%rbp), %rax
	pushq	-72(%rbp)
	pushq	-80(%rbp)
	movq	%rsi, %r9
	movq	%rcx, %r8
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	s4d1_out.3(%rip), %rax
	movq	%rax, %rsi
	leaq	proj_out.4(%rip), %rax
	movq	%rax, %rdi
	call	s4d_layer
	addq	$16, %rsp
	movl	$262144, %esi
	leaq	s4d1_out.3(%rip), %rax
	movq	%rax, %rdi
	call	gelu
	movq	-40(%rbp), %rsi
	movq	-48(%rbp), %rcx
	movq	-56(%rbp), %rdx
	movq	-64(%rbp), %rax
	pushq	-24(%rbp)
	pushq	-32(%rbp)
	movq	%rsi, %r9
	movq	%rcx, %r8
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	s4d2_out.2(%rip), %rax
	movq	%rax, %rsi
	leaq	s4d1_out.3(%rip), %rax
	movq	%rax, %rdi
	call	s4d_layer
	addq	$16, %rsp
	movl	$262144, %esi
	leaq	s4d2_out.2(%rip), %rax
	movq	%rax, %rdi
	call	gelu
	leaq	pooled.1(%rip), %rax
	movq	%rax, %rsi
	leaq	s4d2_out.2(%rip), %rax
	movq	%rax, %rdi
	call	take_last_timestamp
	movq	-8(%rbp), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	logits.0(%rip), %rax
	movq	%rax, %rsi
	leaq	pooled.1(%rip), %rax
	movq	%rax, %rdi
	call	linear_fc
	movl	$0, -136(%rbp)
	jmp	.L53
.L54:
	movl	-136(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	addq	%rdx, %rax
	movl	-136(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	logits.0(%rip), %rdx
	movss	(%rcx,%rdx), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -136(%rbp)
.L53:
	cmpl	$3, -136(%rbp)
	jle	.L54
	movq	-160(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	softmax
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	model_forward, .-model_forward
	.local	hilbert_out.5
	.comm	hilbert_out.5,16384,32
	.local	proj_out.4
	.comm	proj_out.4,1048576,32
	.local	s4d1_out.3
	.comm	s4d1_out.3,1048576,32
	.local	s4d2_out.2
	.comm	s4d2_out.2,1048576,32
	.local	pooled.1
	.comm	pooled.1,256,32
	.local	logits.0
	.comm	logits.0,16,16
	.section	.rodata
	.align 16
.LC1:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 4
.LC2:
	.long	1065353216
	.align 4
.LC4:
	.long	1078530011
	.align 4
.LC5:
	.long	1073741824
	.align 4
.LC6:
	.long	1027024659
	.align 4
.LC7:
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
