import os

files = ['main.s', 'math.s', 'nn.s']

# Strict Rubric Families
families = {
    'R': ['add', 'sub', 'and', 'or', 'xor', 'sll', 'srl', 'sra', 'mul'],
    'I': ['addi', 'andi', 'ori', 'xori', 'slli', 'srli', 'srai', 'lw', 'lh', 'lb', 'jalr', 'li', 'mv', 'nop', 'ret', 'ecall'],
    'S': ['sw', 'sh', 'sb'],
    'B': ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'bgt', 'ble'],
    'U': ['lui', 'auipc', 'la'], 
    'J': ['jal', 'j', 'call', 'tail']
}

def get_family(opcode):
    if opcode.startswith('f'): return 'F' # Catches fadd.s, fsw, flw, fcvt, fmv, etc.
    for fam, ops in families.items():
        if opcode in ops: return fam
    return 'Unknown'

results = {f: {'R':0, 'I':0, 'S':0, 'B':0, 'U':0, 'J':0, 'F':0, 'Total':0} for f in files}
aggregate = {'R':0, 'I':0, 'S':0, 'B':0, 'U':0, 'J':0, 'F':0, 'Total':0}

for file in files:
    if not os.path.exists(file):
        print(f" ERROR: Cannot find {file} in the current directory.")
        continue
        
    with open(file, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip() # Remove comments
            if not line or line.endswith(':') or line.startswith('.'): continue
            
            opcode = line.split()[0]
            fam = get_family(opcode)
            
            if fam != 'Unknown':
                results[file][fam] += 1
                results[file]['Total'] += 1
                aggregate[fam] += 1
                aggregate['Total'] += 1

print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
print(r"\hline")
print(r"Module & R-type & I-type & S-type & B-type & U-type & J-type & F-type & Total \\")
print(r"\hline")
for file in files:
    c = results[file]
    print(rf"{file} & {c['R']} & {c['I']} & {c['S']} & {c['B']} & {c['U']} & {c['J']} & {c['F']} & {c['Total']} \\")
print(r"\hline")
print(rf"\textbf{{Aggregate}} & \textbf{{{aggregate['R']}}} & \textbf{{{aggregate['I']}}} & \textbf{{{aggregate['S']}}} & \textbf{{{aggregate['B']}}} & \textbf{{{aggregate['U']}}} & \textbf{{{aggregate['J']}}} & \textbf{{{aggregate['F']}}} & \textbf{{{aggregate['Total']}}} \\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\caption{Task 3.4.1: Static Instruction Counts per Source Module}")
print(r"\label{tab:static_count}")
print(r"\end{table}")