import re
from collections import defaultdict

families = {
    'R-type': ['add', 'sub', 'sll', 'slt', 'sltu', 'xor', 'srl', 'sra', 'or', 'and', 'mul', 'mulh', 'mulhsu', 'mulhu', 'div', 'divu', 'rem', 'remu'],
    'I-type': ['addi', 'slti', 'sltiu', 'xori', 'ori', 'andi', 'slli', 'srli', 'srai', 'lb', 'lh', 'lw', 'lbu', 'lhu', 'jalr', 'ecall', 'ebreak', 'csrr', 'csrw', 'csrwi', 'li', 'mv', 'ret'],
    'S-type': ['sb', 'sh', 'sw'],
    'B-type': ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'blez', 'bgez', 'bltz', 'bgtz'],
    'U-type': ['lui', 'auipc', 'la'],
    'J-type': ['jal', 'j', 'call', 'tail', 'jr'],
    'F-type': ['fadd', 'fsub', 'fmul', 'fdiv', 'fsqrt', 'fsgnj', 'fsgnjn', 'fsgnjx', 'fmin', 'fmax', 'fcvt', 'fmv', 'feq', 'flt', 'fle', 'fclass', 'flw', 'fsw', 'fmadd', 'fmsub', 'fnmsub', 'fnmadd', 'fabs', 'fneg']
}

counts = defaultdict(int)
total = 0

try:
    with open('build/logs/galaxy_0.txt', 'r') as f:
        for line in f:
            # Whisper profile sub-lines start with space/+, we only want the main instruction lines
            if line.startswith(' ') or line.startswith('+'): 
                continue
                
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            instr = parts[0]
            # Strip RISC-V Compressed prefix if present
            if instr.startswith('c.'):
                instr = instr[2:]
                
            try:
                count = int(parts[-1])
            except ValueError:
                continue

            # Classify
            base_instr = instr.split('.')[0]
            if instr.startswith('v'):
                family = 'V-type'
            else:
                family = 'R-type' # Default fallback
                for fam, instrs in families.items():
                    if base_instr in instrs or instr in instrs:
                        family = fam
                        break
            
            counts[family] += count
            total += count

    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|}")
    print("\\hline")
    print("\\textbf{Instruction Family} & \\textbf{Dynamic Count} & \\textbf{Percentage} \\\\")
    print("\\hline")
    
    # Print in a fixed, consistent order
    for fam in ['R-type', 'I-type', 'S-type', 'B-type', 'U-type', 'J-type', 'F-type', 'V-type']:
        count = counts[fam]
        pct = (count / total * 100) if total > 0 else 0
        print(f"{fam} & {count:,} & {pct:.2f}\\% \\\\")
        
    print("\\hline")
    print(f"\\textbf{{Total Processed}} & \\textbf{{{total:,}}} & \\textbf{{100.00\\%}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Task 3.3.2: Dynamic Instruction Execution Breakdown (VeeR-iSS)}")
    print("\\label{tab:dynamic_count_veer}")
    print("\\end{table}")

except FileNotFoundError:
    print("Log file not found. Please run ./run_final_benchmark.sh first.")
