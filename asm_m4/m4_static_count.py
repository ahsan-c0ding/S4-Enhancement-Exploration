import re
import collections

files = ['main.s', 'nn.s', 'math.s']
counts = collections.defaultdict(int)

def classify(instr):
    instr = instr.split('.')[0] # Strip .s, .vv, .vs, etc.
    if instr.startswith('v'): return 'V-type'
    if instr in ['fadd', 'fsub', 'fmul', 'fdiv', 'fmadd', 'fsqrt', 'fcvt', 'fmv', 'flw', 'fsw', 'flt', 'fle', 'feq', 'fabs', 'fneg', 'fmax', 'fmin', 'fsgnjx', 'fsgnj', 'fsgnjn']: return 'F-type'
    if instr in ['sw', 'sh', 'sb']: return 'S-type'
    if instr in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'bgt', 'ble']: return 'B-type'
    if instr in ['lui', 'auipc', 'la']: return 'U-type'
    if instr in ['jal', 'j', 'call']: return 'J-type'
    if instr in ['addi', 'lw', 'lh', 'lb', 'jalr', 'srai', 'srli', 'slli', 'andi', 'ori', 'xori', 'li', 'mv', 'ret']: return 'I-type'
    return 'R-type' # add, sub, and, or, xor, mul, etc.

total = 0
for f in files:
    with open(f, 'r') as fp:
        for line in fp:
            line = line.split('#')[0].strip()
            if not line or ':' in line or line.startswith('.'): continue
            parts = line.split()
            if parts:
                counts[classify(parts[0])] += 1
                total += 1

print("\\begin{table}[h!]")
print("\\centering")
print("\\begin{tabular}{|l|c|c|}")
print("\\hline")
print("Instruction Family & Static Count & Percentage \\\\")
print("\\hline")
for fam in ['R-type', 'I-type', 'S-type', 'B-type', 'U-type', 'J-type', 'F-type', 'V-type']:
    pct = (counts[fam]/total)*100 if total > 0 else 0
    print(f"{fam} & {counts[fam]:,} & {pct:.2f}\\% \\\\")
print("\\hline")
print(f"\\textbf{{Total}} & \\textbf{{{total:,}}} & \\textbf{{100.00\\%}} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{Task 3.3.1: Static Instruction Breakdown (Source Code)}")
print("\\label{tab:static_count}")
print("\\end{table}")
