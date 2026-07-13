#!/usr/bin/env python3
# Run inside ~/s4-enhancement on the `riscv` branch: python3 patch_math_remez.py
# Swaps my_exp from 4th-order Taylor -> validated 6th-order Remez (Cody-Waite ln2 split).
import re, sys
new='''# r = x - n*ln2_hi - n*ln2_lo   (Cody-Waite split for accuracy)
    li t0, 0x3F317200            # ln2_hi
    fmv.w.x ft3, t0
    fmul.s ft3, ft2, ft3
    fsub.s ft4, fa0, ft3
    li t0, 0x35BFBE8E            # ln2_lo
    fmv.w.x ft7, t0
    fmul.s ft7, ft2, ft7
    fsub.s ft4, ft4, ft7

    # P(r) = Horner(c6..c0), 6th-order Remez minimax
    li t0, 0x3AB6ECC1            # c6
    fmv.w.x ft5, t0
    li t0, 0x3C0937D6            # c5
    fmv.w.x ft6, t0
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6
    li t0, 0x3D2AAA0E            # c4
    fmv.w.x ft6, t0
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6
    li t0, 0x3E2AAA02            # c3
    fmv.w.x ft6, t0
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6
    li t0, 0x3F000000            # c2
    fmv.w.x ft6, t0
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6
    li t0, 0x3F800000            # c1 (=c0=1.0)
    fmv.w.x ft6, t0
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6
    fmul.s ft5, ft5, ft4
    fadd.s ft5, ft5, ft6         # + c0 (1.0, reuse ft6)
'''
pat=re.compile(r'# r = x - n \* ln2.*?(?=\n\n\s*# 2\^n)', re.DOTALL)
for fn in ['math.s','math_qemu.s']:
    s=open(fn).read(); m=len(pat.findall(s))
    if m!=1: sys.exit(f"{fn}: expected 1 match, got {m} (already patched?)")
    open(fn,'w').write(pat.sub(new,s,1)); print(f"{fn}: patched")
print("done -- now: python3 run_task2_qemu_final.py   (expect 5/5)  then  bash run_final_benchmark.sh")
