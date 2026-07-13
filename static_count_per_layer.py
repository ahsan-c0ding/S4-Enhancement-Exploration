#!/usr/bin/env python3
"""Per-layer STATIC instruction count (scalar vs RVV). Run from repo root:
      python3 static_count_per_layer.py
Companion to dynamic_count_per_layer.py (executed-instruction counts)."""
import re, collections
FILES=['main.s','nn.s','math.s']
FAMS=['R-type','I-type','S-type','B-type','U-type','J-type','F-type','V-type']
def classify(i):
    i=i.split('.')[0]
    if i.startswith('v'): return 'V-type'
    if i in ['fadd','fsub','fmul','fdiv','fmadd','fsqrt','fcvt','fmv','flw','fsw','flt','fle','feq','fabs','fneg','fmax','fmin','fsgnjx','fsgnj','fsgnjn','fnmsub','fmsub','fnmadd']: return 'F-type'
    if i in ['sw','sh','sb']: return 'S-type'
    if i in ['beq','bne','blt','bge','bltu','bgeu','beqz','bnez','bgt','ble','bgtz','bltz','blez','bgez']: return 'B-type'
    if i in ['lui','auipc','la']: return 'U-type'
    if i in ['jal','j','call','tail']: return 'J-type'
    if i in ['addi','lw','lh','lb','jalr','srai','srli','slli','andi','ori','xori','li','mv','ret']: return 'I-type'
    return 'R-type'
ENTRY={'_start','complex_mul','hilbert_scan','take_last_timestamp','linear','gelu','softmax','complex_exp','s4d_layer','model_forward','my_exp','my_log','my_sin','my_cos','my_tanh','my_pow_int','my_pow','my_sqrt','v_my_tanh'}
LAYER_GROUPS=[('Hilbert Scan',['hilbert_scan']),('Linear (proj + fc)',['linear']),('S4D Layer (recurrent)',['s4d_layer']),('  complex helpers',['complex_mul','complex_exp']),('GELU',['gelu','v_my_tanh']),('Softmax',['softmax']),('TakeLastTimestep',['take_last_timestamp']),('Math library',['my_exp','my_log','my_sin','my_cos','my_tanh','my_sqrt','my_pow','my_pow_int']),('Driver (model_forward+_start)',['model_forward','_start'])]
def main():
    fc=collections.defaultdict(collections.Counter); cur=None
    for f in FILES:
        for line in open(f):
            s=line.split('#')[0].strip()
            if not s: continue
            m=re.match(r'^([A-Za-z_][A-Za-z0-9_]*):',s)
            if m:
                if m.group(1) in ENTRY: cur=m.group(1)
                continue
            if ':' in s or s.startswith('.'): continue
            p=s.split()
            if p and cur: fc[cur][classify(p[0])]+=1
    rows=[]; grand=collections.Counter()
    for name,fns in LAYER_GROUPS:
        c=collections.Counter()
        for fn in fns: c+=fc.get(fn,collections.Counter())
        if not name.startswith('  '): grand+=c
        rows.append((name.strip(),c))
    w=34
    print(f"{'Layer':<{w}}{'Scalar':>8}{'Vector':>8}{'Total':>8}"); print('-'*(w+24))
    for name,c in rows:
        t=sum(c.values()); v=c['V-type']; print(f"{name:<{w}}{t-v:>8}{v:>8}{t:>8}")
    print('-'*(w+24)); gt=sum(grand.values()); gv=grand['V-type']
    print(f"{'TOTAL':<{w}}{gt-gv:>8}{gv:>8}{gt:>8}")
    print(f"\nStatic vector share: {gv/gt*100:.1f}%   |   "+", ".join(f"{k}={grand[k]}" for k in FAMS))
    print("\n% ---- LaTeX (Task 3.3.1 per-layer) ----")
    print("\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|l|c|c|c|}\n\\hline")
    print("\\textbf{Layer} & \\textbf{Scalar} & \\textbf{Vector} & \\textbf{Total} \\\\\n\\hline")
    for name,c in rows:
        t=sum(c.values()); v=c['V-type']; print(f"{name} & {t-v} & {v} & {t} \\\\")
    print("\\hline"); print(f"\\textbf{{Total}} & \\textbf{{{gt-gv}}} & \\textbf{{{gv}}} & \\textbf{{{gt}}} \\\\")
    print("\\hline\n\\end{tabular}\n\\caption{Per-Layer Static Instruction Count (scalar vs RVV)}\n\\label{tab:static_per_layer}\n\\end{table}")
if __name__=='__main__': main()
