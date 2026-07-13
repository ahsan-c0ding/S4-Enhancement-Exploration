#!/usr/bin/env python3
"""
Re-derive STATIC per-layer instruction counts (total, vector, %vector) for the
C builds at O2 / O3 / RAW(-O0), straight from the compiler's own assembly.
Kills legacy risk: every number comes from the real .s the compiler emits.

USAGE (convolution = Analysis (Initial); run in the conv C dir):
    python3 rederive_static_vec.py --src-dir ~/s4-enhancement/main_conv/c \
        --cc riscv32-unknown-elf-gcc --files nn.c math.c main.c

For the recurrent build (Analysis (Post Optimization) RAW column) point --src-dir
at that C dir instead.
"""
import argparse, subprocess, re, collections, os, tempfile

# C function  ->  sheet layer(s)
GROUP = {
    'hilbert_scan':'hilbert', 'linear':'linear(proj/fc)',
    's4d_layer':'s4_x', 'complex_mul':'s4_x', 'complex_exp':'s4_x',
    'gelu':'gelu_x', 'softmax':'softmax', 'take_last_timestamp':'ttls',
    'model_forward':'driver',
    'my_exp':'math','my_sin':'math','my_cos':'math','my_tanh':'math',
    'my_sqrt':'math','my_log':'math','my_pow':'math','my_pow_int':'math',
}
LAYERS = ['hilbert','linear(proj/fc)','s4_x','gelu_x','ttls','softmax','math','driver']

def count_from_s(s_text):
    cur=None; vec=collections.Counter(); tot=collections.Counter()
    for ln in s_text.splitlines():
        s=ln.split('#')[0].strip()
        if not s: continue
        m=re.match(r'^([A-Za-z_][A-Za-z0-9_.$]*):', s)
        if m:
            cur=GROUP.get(m.group(1), cur if m.group(1).startswith('.L') else None)
            continue
        if s.startswith('.'): continue
        op=s.split()[0]
        if cur:
            tot[cur]+=1
            if op.startswith('v'): vec[cur]+=1
    return vec, tot

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--cc', default='riscv32-unknown-elf-gcc')
    ap.add_argument('--files', nargs='+', default=['nn.c','math.c','main.c'])
    a=ap.parse_args()
    flags={'RAW':'-O0','O2':'-O2','O3':'-O3'}
    print(f"{'layer':16s} " + " ".join(f"{k+'(v/tot/%)':>16s}" for k in flags))
    results={}
    for lvl,opt in flags.items():
        merged=""
        for f in a.files:
            src=os.path.join(a.src_dir,f)
            if not os.path.exists(src): continue
            with tempfile.NamedTemporaryFile('w+',suffix='.s',delete=False) as tf:
                out=tf.name
            cmd=[a.cc,opt,'-S','-march=rv32gcv','-mabi=ilp32f','-include','string.h',src,'-o',out]
            r=subprocess.run(cmd,capture_output=True,text=True)
            if r.returncode!=0:
                print(f"  [warn] {f}@{lvl} compile failed: {r.stderr.strip().splitlines()[-1:]}")
                continue
            merged+=open(out).read()+"\n"; os.unlink(out)
        results[lvl]=count_from_s(merged)
    for lyr in LAYERS:
        row=f"{lyr:16s} "
        for lvl in flags:
            vec,tot=results[lvl]
            v,t=vec[lyr],tot[lyr]
            row+=f"{f'{v}/{t}/{(v/t*100 if t else 0):.0f}%':>16s} "
        print(row)
    print("\nNote: at -O2/-O3 small fns may inline into callers (esp. into s4_x/driver);")
    print("cross-check the 'total' against the sheet's existing static totals before recording.")

if __name__=='__main__':
    main()
