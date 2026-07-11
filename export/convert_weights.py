#!/usr/bin/env python
"""Convert exported model_weights.bin -> a loadable .pth checkpoint."""
import os
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_BIN = os.path.join(_SCRIPT_DIR, "..", "model_params", "model_weights.bin")
OUT_PTH = os.path.join(_SCRIPT_DIR, "..", "model_params", "galaxys4-30EPOCH-STANDARD.pth")

import sys
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from model import GalaxyClassifierS4D   # safe now: this runs *after* `model` is fully importable

layout = [
    ('hilbert_scan.indices', (4096,), torch.int32),
    ('uproject.weight',      (64, 1), torch.float32),
    ('uproject.bias',        (64,),   torch.float32),
    ('s4_1.log_dt',          (64,),   torch.float32),
    ('s4_1.log_A_real',      (64, 32), torch.float32),
    ('s4_1.A_imag',          (64, 32), torch.float32),
    ('s4_1.C',               (64, 32, 2), torch.float32),
    ('s4_1.D',               (64,),   torch.float32),
    ('s4_2.log_dt',          (64,),   torch.float32),
    ('s4_2.log_A_real',      (64, 32), torch.float32),
    ('s4_2.A_imag',          (64, 32), torch.float32),
    ('s4_2.C',               (64, 32, 2), torch.float32),
    ('s4_2.D',               (64,),   torch.float32),
    ('fc.weight',            (4, 64), torch.float32),
    ('fc.bias',              (4,),    torch.float32),
]

raw = np.fromfile(SRC_BIN, dtype='<f4')
raw_i32 = np.fromfile(SRC_BIN, dtype='<i4')

state, offset = {}, 0
for name, shape, dtype in layout:
    n = int(np.prod(shape))
    if dtype == torch.int32:
        chunk = raw_i32[offset:offset+n].astype(np.int64)
        t = torch.from_numpy(chunk).reshape(shape)
    else:
        chunk = raw[offset:offset+n].astype(np.float32)
        t = torch.from_numpy(chunk.copy()).reshape(shape)
    state[name] = t
    offset += n

assert offset == 21124, f"consumed {offset} floats, expected 21124 -- layout mismatch"

model = GalaxyClassifierS4D(colored=False)
model.load_state_dict(state, strict=True)
print("Loaded cleanly. Saving checkpoint...")
torch.save(model.state_dict(), OUT_PTH)
print(f"Wrote {OUT_PTH}")