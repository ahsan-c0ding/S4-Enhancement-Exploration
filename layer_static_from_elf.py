#!/usr/bin/env python3
"""
Unified STATIC per-layer instruction count from a compiled ELF (works for the
hand-written asm builds AND the C -O2/-O3 builds -- disassembles the binary, so
it's the same methodology for every implementation).

USAGE
  python3 layer_static_from_elf.py <elf> [--objdump riscv32-unknown-elf-objdump] [--tsv]

Counts every instruction in the disassembly, classifies scalar vs vector, and
groups by function into the 9 sheet layers. Repeated-call layers that share one
function (input_proj/output_proj -> linear, s4_1/s4_2 -> s4d_layer,
gelu_1/gelu_2 -> gelu) share the same static code, so they're reported once under
the function; dynamic counts (separate script) split them per invocation.
"""
import sys, re, subprocess, collections

FAMS=['R-type','I-type','S-type','B-type','U-type','J-type','F-type','V-type']
def classify(m):
    m=m.lower()
    if m.startswith('c.'): m=m[2:]
    base=m.split('.')[0]
    if m.startswith('v'): return 'V-type'
    if base in ['fadd','fsub','fmul','fdiv','fmadd','fmsub','fnmadd','fnmsub','fsqrt','fcvt','fmv','flw','fsw','flt','fle','feq','fabs','fneg','fmax','fmin','fsgnj','fsgnjn','fsgnjx','fclass','fld','fsd']: return 'F-type'
    if base in ['sw','sh','sb','sd','fsw','fsd']: return 'S-type'
    if base in ['beq','bne','blt','bge','bltu','bgeu','beqz','bnez','bgt','ble','bgtz','bltz','blez','bgez']: return 'B-type'
    if base in ['lui','auipc']: return 'U-type'
    if base in ['jal','j','jr','call','tail','ret','jalr']: return 'J-type'   # control transfer
    if base in ['addi','slti','sltiu','xori','ori','andi','slli','srli','srai','lw','lh','lb','lbu','lhu','li','mv','nop','lla','la','ecall','ebreak','csrr','csrw','csrrw','csrrs','fence','ld']: return 'I-type'
    return 'R-type'

# function -> sheet layer
FUNC_LAYER={
 'hilbert_scan':'hilbert',
 'linear':'linear(proj/fc)','linear_uproject':'linear(proj/fc)','linear_fc':'linear(proj/fc)',
 's4d_layer':'s4_1/s4_2 (s4d)',
 'gelu':'gelu_1/gelu_2','v_my_tanh':'gelu_1/gelu_2','softmax':'softmax',
 'take_last_timestamp':'ttls','complex_mul':'s4d helpers','complex_exp':'s4d helpers',
}
MATH={'my_exp','my_log','my_sin','my_cos','my_tanh','my_pow','my_pow_int','my_sqrt'}
DRIVER={'model_forward','_start','main','find_argmax'}
def layer_of(fn):
    if fn in FUNC_LAYER: return FUNC_LAYER[fn]
    if fn in MATH: return 'math lib'
    if fn in DRIVER: return 'driver'
    return 'other'

LAYER_ORDER=['hilbert','linear(proj/fc)','s4_1/s4_2 (s4d)','s4d helpers','gelu_1/gelu_2','ttls','softmax','math lib','driver','other']

FUNC_HDR=re.compile(r'^[0-9a-fA-F]+\s+<([^>]+)>:')
# an instruction line: "   10074:\t1101      \taddi\tsp,sp,-32"
INSN=re.compile(r'^\s*[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2,8}\s+)+([a-zA-Z][\w.]*)')

def parse(disasm_lines):
    per=collections.defaultdict(collections.Counter); cur=None
    for line in disasm_lines:
        h=FUNC_HDR.match(line)
        if h: cur=h.group(1); per.setdefault(cur,collections.Counter()); continue
        m=INSN.match(line)
        if m and cur is not None:
            per[cur][classify(m.group(1))]+=1
    return per

def report(per, tsv=False):
    layers=collections.OrderedDict((l,collections.Counter()) for l in LAYER_ORDER)
    for fn,c in per.items():
        layers[layer_of(fn)]+=c
    rows=[(l,c) for l,c in layers.items() if sum(c.values())>0]
    if tsv:
        print("layer\tscalar\tvector\ttotal\t"+"\t".join(FAMS))
        for l,c in rows:
            t=sum(c.values()); v=c['V-type']
            print(f"{l}\t{t-v}\t{v}\t{t}\t"+"\t".join(str(c[f]) for f in FAMS))
        return
    w=22
    print(f"{'Layer':<{w}}{'Scalar':>8}{'Vector':>8}{'Total':>8}")
    print('-'*(w+24)); g=collections.Counter()
    for l,c in rows:
        t=sum(c.values()); v=c['V-type']; g+=c
        print(f"{l:<{w}}{t-v:>8}{v:>8}{t:>8}")
    print('-'*(w+24)); gt=sum(g.values()); gv=g['V-type']
    print(f"{'TOTAL':<{w}}{gt-gv:>8}{gv:>8}{gt:>8}")
    print("families: "+", ".join(f"{f}={g[f]}" for f in FAMS))

def selftest():
    sample=""".text disassembly
00010074 <hilbert_scan>:
   10074:\t1101      \taddi\tsp,sp,-32
   10076:\tce06      \tsw\tra,28(sp)
   10078:\t8082      \tret
00010100 <s4d_layer>:
   10100:\t0d757057 \tvsetvli\tt0,a0,e32,m2,ta,ma
   10104:\t9a1420d7 \tvfmacc.vv\tv8,v2,v4
   10108:\t062420d7 \tvfredosum.vs\tv1,v8,v2
   1010c:\t102170d3 \tfmul.s\tft0,ft1,ft2
   10110:\tfa06d6e3 \tbeqz\ta6,10100
00010200 <my_exp>:
   10200:\t60b7f543 \tfmadd.s\tfa0,fa1,fa2,fa3
   10204:\t0505      \taddi\ta0,a0,1
"""
    per=parse(sample.splitlines())
    assert per['hilbert_scan']['I-type']==1 and per['hilbert_scan']['S-type']==1 and per['hilbert_scan']['J-type']==1, dict(per['hilbert_scan'])
    assert per['s4d_layer']['V-type']==3, dict(per['s4d_layer'])
    assert per['s4d_layer']['F-type']==1 and per['s4d_layer']['B-type']==1, dict(per['s4d_layer'])
    assert per['my_exp']['F-type']==1 and per['my_exp']['I-type']==1, dict(per['my_exp'])
    print("SELFTEST OK: hilbert(I/S/J), s4d(3 vector,1 F,1 B), my_exp(F,I) parsed & classified correctly")

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--selftest': selftest(); sys.exit(0)
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    elf=sys.argv[1]; objdump='riscv32-unknown-elf-objdump'; tsv=False
    a=sys.argv[2:]
    if '--tsv' in a: tsv=True
    if '--objdump' in a: objdump=a[a.index('--objdump')+1]
    try:
        dis=subprocess.check_output([objdump,'-d',elf],text=True).splitlines()
    except FileNotFoundError:
        sys.exit(f"ERROR: {objdump} not found; pass --objdump <your riscv objdump>")
    report(parse(dis), tsv=tsv)
