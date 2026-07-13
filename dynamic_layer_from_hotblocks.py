#!/usr/bin/env python3
"""
Per-layer DYNAMIC instruction count from a QEMU libhotblocks dump.

Usage:
  ELF=.../galaxy_0_veer.exe
  HOT=.../libhotblocks.so
  qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32 -plugin $HOT -d plugin $ELF > /tmp/hot.txt 2>&1
  python3 dynamic_layer_from_hotblocks.py $ELF /tmp/hot.txt [--nm riscv32-unknown-elf-nm] [--tsv]

hotblocks line:  0x<addr>, <trans_count>, <insns_per_block>, <exec_count>
instructions executed by a block = insns_per_block * exec_count  (last two ints).
Blocks are attributed to the enclosing function by address; functions -> sheet layers.
For the recurrent build the repeated layers execute identical counts (fixed-iteration
math + fixed loops), so s4d and gelu totals are split 50/50 into _1/_2, and linear is
split input_proj:output_proj by its known iteration ratio (SEQ_LEN*D_MODEL : N_CLASSES*D_MODEL).
"""
import sys, re, subprocess, bisect, collections

ENTRY={'_start','complex_mul','hilbert_scan','take_last_timestamp','linear','gelu','softmax',
       'complex_exp','s4d_layer','model_forward','my_exp','my_log','my_sin','my_cos','my_tanh',
       'my_pow_int','my_pow','my_sqrt','v_my_tanh'}

def load_ranges(elf, nm):
    out=None
    for cand in ([nm] if nm else ['riscv32-unknown-elf-nm','nm']):
        try: out=subprocess.check_output([cand,'-n',elf],text=True); break
        except (FileNotFoundError,subprocess.CalledProcessError): pass
    if out is None: sys.exit("need nm (riscv32-unknown-elf-nm)")
    syms=[]
    for line in out.splitlines():
        p=line.split()
        if len(p)>=3 and p[1] in ('t','T'): syms.append((int(p[0],16),p[2]))
    syms.sort()
    starts=[a for a,_ in syms]; entries=[]; cur=None
    for a,n in syms:
        if n in ENTRY: cur=n
        entries.append(cur)
    return starts, entries

HEXLINE=re.compile(r'0x([0-9a-fA-F]+)\D+(.*)')
def parse_hot(path, starts, entries):
    per=collections.Counter(); matched=0; skipped=0
    for line in open(path):
        m=HEXLINE.search(line)
        if not m: continue
        addr=int(m.group(1),16)
        ints=[int(x) for x in re.findall(r'\d+', m.group(2))]
        if len(ints)<2: skipped+=1; continue
        instrs=ints[-1]*ints[-2]          # insns_per_block * exec_count
        i=bisect.bisect_right(starts,addr)-1
        if i<0 or entries[i] is None: per['<unmapped>']+=instrs; continue
        per[entries[i]]+=instrs; matched+=1
    return per, matched, skipped

def build_layers(per):
    g=lambda k: per.get(k,0)
    s4d = g('s4d_layer')+g('complex_mul')+g('complex_exp')
    gel = g('gelu')+g('v_my_tanh')
    math= sum(g(k) for k in ['my_exp','my_log','my_sin','my_cos','my_tanh','my_pow','my_pow_int','my_sqrt'])
    lin = g('linear')
    R=1024/1025.0                          # input_proj : output_proj  = SEQ_LEN*D_MODEL : N_CLASSES*D_MODEL
    # math lib is called almost entirely from s4d (dt) + softmax; fold into s4d (dominant)
    s4d_total = s4d + math
    rows=[
      ('hilbert',      g('hilbert_scan')),
      ('input_proj',   lin*R),
      ('s4_1',         s4d_total/2),
      ('gelu_1',       gel/2),
      ('s4_2',         s4d_total/2),
      ('gelu_2',       gel/2),
      ('ttls',         g('take_last_timestamp')),
      ('output_proj',  lin*(1-R)),
      ('softmax',      g('softmax')),
    ]
    overhead = g('model_forward')+g('_start')+g('<unmapped>')
    return rows, overhead

def main():
    if len(sys.argv)==2 and sys.argv[1]=='--selftest': selftest(); return
    if len(sys.argv)<3: print(__doc__); sys.exit(1)
    elf,hot=sys.argv[1],sys.argv[2]; nm=None; tsv='--tsv' in sys.argv
    if '--nm' in sys.argv: nm=sys.argv[sys.argv.index('--nm')+1]
    starts,entries=load_ranges(elf,nm)
    per,matched,skipped=parse_hot(hot,starts,entries)
    rows,overhead=build_layers(per)
    total=sum(v for _,v in rows)
    if tsv:
        print("layer\tinstrs"); [print(f"{n}\t{int(round(v))}") for n,v in rows]
        print(f"# compute_total\t{int(round(total))}\n# overhead(driver/glue)\t{int(round(overhead))}"); return
    print(f"{'Layer':<14}{'Dynamic instrs':>16}{'%':>8}")
    print('-'*40)
    for n,v in rows: print(f"{n:<14}{int(round(v)):>16,}{(v/total*100 if total else 0):>7.2f}%")
    print('-'*40)
    print(f"{'COMPUTE TOTAL':<14}{int(round(total)):>16,}")
    print(f"{'overhead*':<14}{int(round(overhead)):>16,}   (_start/model_forward glue, not a layer)")
    print("\n* per-function raw:", {k:int(v) for k,v in sorted(per.items(),key=lambda x:-x[1])})

def selftest():
    # synthetic: two funcs; hotblocks lines "addr, tcount, insns, exec"
    starts=[0x80000100,0x80000200]; entries=['s4d_layer','gelu']
    lines=["0x0000000080000104, 2, 11, 262208","0x0000000080000150, 1, 58, 131072",
           "0x0000000080000210, 3, 7, 4097","garbage"]
    import tempfile,os
    p=tempfile.mktemp(); open(p,'w').write("\n".join(lines))
    per,m,s=parse_hot(p,starts,entries); os.unlink(p)
    exp_s4d=11*262208+58*131072; exp_gelu=7*4097
    assert per['s4d_layer']==exp_s4d, per
    assert per['gelu']==exp_gelu, per
    print(f"SELFTEST OK: s4d={exp_s4d:,} (2 blocks), gelu={exp_gelu:,}, matched={m}, skipped={s}")

if __name__=='__main__': main()
