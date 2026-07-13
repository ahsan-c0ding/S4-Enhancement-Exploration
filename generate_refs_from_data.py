#!/usr/bin/env python3
"""
Build PyTorch reference tensors for ALL 10 samples by reading the EXACT image
already baked into each riscv/data_N.s (the same bytes the RISC-V runs), so the
references always match the hardware input (Hilbert MSE -> 0 by construction).

This fixes the earlier mistake of regenerating images with torch.manual_seed():
only sample 0 was seeded that way; data_1..9 came from a different generator.

RUN FROM the dir that has the `model/` package + model_params/ (e.g. main_conv):
    python3 generate_refs_from_data.py --riscv-dir ~/s4-enhancement/riscv

It reads <riscv-dir>/data_0.s .. data_9.s and writes <riscv-dir>/test_data/sample_*.bin
Tip: `cd <riscv-dir> && git checkout -- test_data/` first if you want a clean base.
"""
import os, re, sys, glob, struct, argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.gclassifier import GalaxyClassifierS4D

def image_from_data_s(path):
    """Extract the 4096 float32 image words following the `image_data:` label."""
    lines = open(path).read().splitlines()
    idx = next(k for k, l in enumerate(lines) if l.strip().startswith("image_data:"))
    words = []
    for l in lines[idx+1:]:
        m = re.match(r"\s*\.word\s+0x([0-9a-fA-F]+)", l)
        if m:
            words.append(int(m.group(1), 16))
        elif l.strip() and not l.strip().startswith("."):
            break
    if len(words) != 4096:
        sys.exit(f"ERROR: {path} image_data has {len(words)} words, expected 4096")
    flt = np.array([struct.unpack("<f", struct.pack("<I", w))[0] for w in words], dtype=np.float32)
    return torch.from_numpy(flt.reshape(1, 1, 64, 64))   # row-major, matches data_0==seed0

def dump(t, path):
    np.asarray(t.detach().cpu().numpy(), dtype=np.float32).flatten().tofile(path)

def find_pth():
    c = glob.glob("model_params/galaxys4*.pth") + glob.glob("../model_params/galaxys4*.pth")
    if not c: sys.exit("ERROR: no model_params/galaxys4*.pth found.")
    return sorted(c)[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--riscv-dir", required=True, help="dir holding data_N.s and test_data/")
    args = ap.parse_args()
    out = os.path.join(args.riscv_dir, "test_data")
    os.makedirs(out, exist_ok=True)

    pth = find_pth()
    print(f"[*] checkpoint: {pth}")
    model = GalaxyClassifierS4D(colored=False)
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    model.eval()

    acts = {}
    def hook(n):
        def h(m, i, o): acts[n] = (o[0] if isinstance(o, tuple) else o).detach()
        return h
    model.hilbert_scan.register_forward_hook(hook("hilbert"))
    model.uproject.register_forward_hook(hook("uproject"))
    model.act1.register_forward_hook(hook("gelu_1"))
    model.act2.register_forward_hook(hook("gelu_2"))
    model.take_last.register_forward_hook(hook("takelast"))

    CLS = ["Round", "In-between", "Cigar", "Edge-on"]
    for i in range(10):
        img = image_from_data_s(os.path.join(args.riscv_dir, f"data_{i}.s"))
        with torch.no_grad():
            probs = model(img, return_logits=False)
        acts["softmax"] = probs.detach()
        pfx = os.path.join(out, f"sample_{i}")
        dump(img,             f"{pfx}_img.bin")
        dump(acts["hilbert"], f"{pfx}_hilbert.bin")
        dump(acts["uproject"],f"{pfx}_uproject.bin")
        dump(acts["gelu_1"],  f"{pfx}_gelu_1.bin")
        dump(acts["gelu_2"],  f"{pfx}_gelu_2.bin")
        dump(acts["takelast"],f"{pfx}_takelast.bin")
        dump(acts["softmax"], f"{pfx}_softmax.bin")
        p = acts["softmax"].numpy().flatten()
        print(f"  sample {i}: class {int(np.argmax(p))} ({CLS[int(np.argmax(p))]:10s}) probs {np.round(p,4)}")

    print(f"\n[*] wrote refs for 10 samples to {out}")
    print("[*] re-run:  python3 run_task2_qemu_final.py   -> expect 10/10 and Hilbert MSE 0.00e+00 everywhere")

if __name__ == "__main__":
    main()
