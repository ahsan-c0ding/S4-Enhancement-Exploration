#!/usr/bin/env python3
"""
Per-layer DYNAMIC instruction count for the S4D RISC-V pipeline.

The whisper --profile output is only an instruction-mix histogram (no PCs), so it
cannot be split per layer. This script instead attributes each *executed* instruction
to a function by its program counter (PC), using the ELF symbol table for the
address ranges, then groups functions into layers.

USAGE
  # 1) make sure the benchmark ELF exists (run_final_benchmark.sh builds it):
  #      build/exe/galaxy_0_veer.exe
  # 2) stream a whisper instruction trace straight into this script (no giant file):
  whisper --configfile veer/whisper.json build/exe/galaxy_0_veer.exe 2>/dev/null \
      | python3 dynamic_count_per_layer.py build/exe/galaxy_0_veer.exe -

  # or from a saved trace log:
  python3 dynamic_count_per_layer.py build/exe/galaxy_0_veer.exe build/logs/galaxy_0_stdout.txt

  # sanity-check the bucketing logic without whisper:
  python3 dynamic_count_per_layer.py --selftest

NOTES
* Attribution is "leaf" / by-PC: when a layer calls a shared routine (e.g. GELU -> the
  vector tanh, or S4D -> scalar my_exp for dt), those instructions are counted under
  the routine's own bucket. That's why "Math library" is listed separately.
* PC detection is format-robust: it takes the first hex token on each trace line whose
  value falls inside the program's .text address range, so it survives small
  differences in whisper's trace column layout.
"""
import sys, re, subprocess, bisect
from collections import defaultdict

# ---- true function entry points (everything else is an internal loop label) ----
ENTRY = {
    '_start','complex_mul','hilbert_scan','take_last_timestamp','linear','gelu',
    'softmax','complex_exp','s4d_layer','model_forward',
    'my_exp','my_log','my_sin','my_cos','my_tanh','my_pow_int','my_pow','my_sqrt','v_my_tanh',
}
# ---- group entry points into reporting "layers" ----
LAYER_GROUPS = [
    ('Hilbert Scan',              ['hilbert_scan']),
    ('Linear (proj + fc)',        ['linear']),
    ('S4D Layer (recurrent)',     ['s4d_layer']),
    ('  complex helpers',         ['complex_mul','complex_exp']),
    ('GELU',                      ['gelu','v_my_tanh']),
    ('Softmax',                   ['softmax']),
    ('TakeLastTimestep',          ['take_last_timestamp']),
    ('Math library',              ['my_exp','my_log','my_sin','my_cos','my_tanh','my_sqrt','my_pow','my_pow_int']),
    ('Driver (model_forward+_start)', ['model_forward','_start']),
]
ENTRY_TO_LAYER = {fn: name.strip() for name, fns in LAYER_GROUPS for fn in fns}

def load_ranges(elf):
    """Return (starts[], entries[], text_lo, text_hi) attributing every symbol to its
    enclosing entry-point function."""
    out = None
    for nm in ('riscv32-unknown-elf-nm','riscv64-unknown-elf-nm','nm'):
        try:
            out = subprocess.check_output([nm, '-n', elf], text=True); break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    if out is None:
        sys.exit("ERROR: need 'nm' (ideally riscv32-unknown-elf-nm) on PATH")
    syms = []
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] in ('t','T'):     # text symbols only
            syms.append((int(p[0],16), p[2]))
    syms.sort()
    if not syms:
        sys.exit("ERROR: no text symbols found via nm")
    starts, entries = [], []
    cur = None
    for a, n in syms:
        if n in ENTRY:
            cur = n
        starts.append(a); entries.append(cur)
    text_lo = syms[0][0]
    text_hi = syms[-1][0] + 0x10000
    return starts, entries, text_lo, text_hi

HEX = re.compile(r'\b[0-9a-fA-F]{4,8}\b')
def extract_pc(line, lo, hi):
    for tok in HEX.findall(line):
        v = int(tok, 16)
        if lo <= v < hi:
            return v
    return None

def run(elf, trace_fh):
    starts, entries, lo, hi = load_ranges(elf)
    per_entry = defaultdict(int)
    matched = unmatched = 0
    for line in trace_fh:
        pc = extract_pc(line, lo, hi)
        if pc is None:
            unmatched += 1; continue
        i = bisect.bisect_right(starts, pc) - 1
        if i < 0:
            unmatched += 1; continue
        per_entry[entries[i]] += 1
        matched += 1
    report(per_entry, matched, unmatched)

def report(per_entry, matched, unmatched):
    rows = []
    grand = 0
    for name, fns in LAYER_GROUPS:
        c = sum(per_entry.get(fn, 0) for fn in fns)
        if not name.startswith('  '):
            grand += c
        rows.append((name.strip(), c))
    total = grand if grand else 1
    print(f"\n{'Layer':<34}{'Dynamic count':>16}{'%':>9}")
    print('-'*59)
    for name, c in rows:
        print(f"{name:<34}{c:>16,}{c/total*100:>8.2f}%")
    print('-'*59)
    print(f"{'TOTAL':<34}{grand:>16,}{100.0:>8.2f}%")
    print(f"\n[bucketed {matched:,} instrs | {unmatched:,} lines had no in-range PC "
          f"(headers/blank/summary)]")
    # LaTeX
    print("\n% ---- LaTeX (Task 3.3.2 per-layer) ----")
    print("\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|l|c|c|}\n\\hline")
    print("\\textbf{Layer} & \\textbf{Dynamic Count} & \\textbf{Percentage} \\\\\n\\hline")
    for name, c in rows:
        print(f"{name} & {c:,} & {c/total*100:.2f}\\% \\\\")
    print("\\hline")
    print(f"\\textbf{{Total}} & \\textbf{{{grand:,}}} & \\textbf{{100.00\\%}} \\\\")
    print("\\hline\n\\end{tabular}")
    print("\\caption{Per-Layer Dynamic Instruction Count (VeeR-iSS, PC-attributed)}")
    print("\\label{tab:dynamic_per_layer}\n\\end{table}")

def selftest():
    # fake symbol layout: entry funcs + internal loop labels between them
    global load_ranges
    starts = [0x100,0x140,0x200,0x260,0x400]
    entries= ['s4d_layer','s4d_layer','gelu','gelu','my_exp']   # 0x140 is an internal loop of s4d
    lo, hi = 0x100, 0x10100
    import bisect as _b
    per = defaultdict(int)
    trace = [
        "0 0 00000120 abc reg",   # s4d
        "0 0 00000150 def reg",   # s4d internal loop -> still s4d
        "0 0 00000210 111 reg",   # gelu
        "0 0 00000410 222 reg",   # my_exp
        "garbage line no pc",
        "0 0 00000410 333 reg",   # my_exp
    ]
    m=u=0
    for line in trace:
        pc=extract_pc(line,lo,hi)
        if pc is None: u+=1; continue
        i=_b.bisect_right(starts,pc)-1
        per[entries[i]]+=1; m+=1
    assert per['s4d_layer']==2, per
    assert per['gelu']==1, per
    assert per['my_exp']==2, per
    assert m==5 and u==1
    print("SELFTEST OK: s4d=2 (incl internal loop), gelu=1, my_exp=2, 1 unmatched")

if __name__ == '__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--selftest':
        selftest(); sys.exit(0)
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    elf, trace = sys.argv[1], sys.argv[2]
    fh = sys.stdin if trace=='-' else open(trace)
    run(elf, fh)
