import re

log_file = 'build/logs/galaxy_0.txt'

# Strict M3 Rubric Families
families = {
    'R': ['add', 'sub', 'and', 'or', 'xor', 'sll', 'srl', 'sra', 'mul', 'c.add', 'c.sub', 'c.and', 'c.or', 'c.xor', 'c.mv'],
    'I': ['addi', 'andi', 'ori', 'xori', 'slli', 'srli', 'srai', 'lw', 'lh', 'lb', 'jalr', 'li', 'mv', 'nop', 'ret', 'ecall', 'c.addi', 'c.li', 'c.slli', 'c.srli', 'c.srai', 'c.lw', 'c.lwsp', 'c.jr', 'c.addi4spn', 'c.addi16sp', 'c.nop'],
    'S': ['sw', 'sh', 'sb', 'c.sw', 'c.swsp'],
    'B': ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'c.beqz', 'c.bnez'],
    'U': ['lui', 'auipc', 'c.lui'],
    'J': ['jal', 'j', 'call', 'tail', 'c.j', 'c.jal']
}

def get_family(opcode):
    if opcode.startswith('f') or opcode.startswith('c.f'): return 'F'
    for fam, ops in families.items():
        if opcode in ops: return fam
    return 'Unknown'

counts = {'R':0, 'I':0, 'S':0, 'B':0, 'U':0, 'J':0, 'F':0, 'Unknown': 0}
total_dynamic = 0

try:
    with open(log_file, 'r') as f:
        for line in f:
            # We ONLY want lines that start with a letter (the opcode) and end with a number (the count)
            # We must ignore lines starting with space, '+', or standard text like 'Retired'
            if line.startswith(' ') or line.startswith('+') or line.startswith('Retired') or line.startswith('PMP') or line.startswith('Interrupts'):
                continue
                
            line = line.strip()
            if not line: continue
            
            # Match "opcode number" exactly at the start of the line
            m = re.match(r'^([a-z0-9\.]+)\s+(\d+)$', line)
            if m:
                opcode = m.group(1)
                count = int(m.group(2))
                
                fam = get_family(opcode)
                if fam != 'Unknown':
                    counts[fam] += count
                    total_dynamic += count

    # Generate the LaTeX Table
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{|l|c|c|}")
    print(r"\hline")
    print(r"Instruction Family & Dynamic Count & Percentage \\")
    print(r"\hline")
    
    for fam in ['R', 'I', 'S', 'B', 'U', 'J', 'F']:
        pct = (counts[fam] / total_dynamic) * 100 if total_dynamic > 0 else 0
        print(rf"{fam}-type & {counts[fam]:,} & {pct:.2f}\% \\")
        
    print(r"\hline")
    print(rf"\textbf{{Total}} & \textbf{{{total_dynamic:,}}} & \textbf{{100.00\%}} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Task 3.4.2: Dynamic Instruction Execution Breakdown (VeeR-iSS)}")
    print(r"\label{tab:dynamic_count}")
    print(r"\end{table}")

except FileNotFoundError:
    print(f" Error: Could not find {log_file}.")