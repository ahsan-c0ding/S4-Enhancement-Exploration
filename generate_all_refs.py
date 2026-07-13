#!/usr/bin/env python3
"""
Regenerate PyTorch reference tensors for ALL 10 samples (0-9), including the
per-layer taps the validator compares against (hilbert, uproject, gelu_1,
gelu_2, takelast, softmax). The old generate_10_samples.py only wrote
img+softmax and pointed at a .pth filename that no longer exists.

RUN FROM the repo dir that contains the `model/` package (the main branch):
    python3 generate_all_refs.py --out /ABSOLUTE/path/to/riscv/test_data

If --out is omitted it writes to ./test_data.
It regenerates every sample and prints each prediction so you can eyeball that
samples 0-4 still match what the RISC-V produces (Round, Round, Round, In-between,
In-between). If those line up, the seed+weight pipeline is consistent and the
freshly generated 5-9 references are trustworthy.
"""
import os, sys, glob, argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.gclassifier import GalaxyClassifierS4D

def dump(t, path):
    np.asarray(t.detach().cpu().numpy(), dtype=np.float32).flatten().tofile(path)

def find_pth():
    cands = glob.glob("model_params/galaxys4*.pth") + glob.glob("../model_params/galaxys4*.pth")
    if not cands:
        sys.exit("ERROR: no model_params/galaxys4*.pth found. Run from the repo root that has model_params/.")
    return sorted(cands)[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_data", help="output dir for sample_*.bin (point at riscv/test_data)")
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pth = find_pth()
    print(f"[*] loading checkpoint: {pth}")
    model = GalaxyClassifierS4D(colored=False)
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    model.eval()

    acts = {}
    def hook(name):
        def h(m, i, o):
            acts[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return h
    model.hilbert_scan.register_forward_hook(hook("hilbert"))
    model.uproject.register_forward_hook(hook("uproject"))
    model.act1.register_forward_hook(hook("gelu_1"))
    model.act2.register_forward_hook(hook("gelu_2"))
    model.take_last.register_forward_hook(hook("takelast"))

    CLASSES = ["Round", "In-between", "Cigar", "Edge-on"]
    for i in range(args.samples):
        torch.manual_seed(i)                 # same seed convention as the original generator
        image = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            probs = model(image, return_logits=False)
        acts["softmax"] = probs.detach()

        prefix = os.path.join(args.out, f"sample_{i}")
        dump(image,            f"{prefix}_img.bin")
        dump(acts["hilbert"],  f"{prefix}_hilbert.bin")
        dump(acts["uproject"], f"{prefix}_uproject.bin")
        dump(acts["gelu_1"],   f"{prefix}_gelu_1.bin")
        dump(acts["gelu_2"],   f"{prefix}_gelu_2.bin")
        dump(acts["takelast"], f"{prefix}_takelast.bin")
        dump(acts["softmax"],  f"{prefix}_softmax.bin")

        p = acts["softmax"].numpy().flatten()
        print(f"  sample {i}: class {np.argmax(p)} ({CLASSES[int(np.argmax(p))]:10s})  probs {np.round(p,4)}")

    print(f"\n[*] wrote references for {args.samples} samples to {args.out}")
    print("[*] sanity: samples 0-4 should read Round, Round, Round, In-between, In-between")
    print("[*] now re-run:  python3 run_task2_qemu_final.py   -> should show 10/10 labeled")

if __name__ == "__main__":
    main()
