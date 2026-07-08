import torch
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.gclassifier import GalaxyClassifierS4D

SRC_BIN = r"C:\Users\abdul\Desktop\S4-Enhancement-Exploration\model_params\model_weights.bin"
OUT_PTH = r"C:\Users\abdul\Desktop\S4-Enhancement-Exploration\model_params\galaxys4-30EPOCH-STANDARD.pth"
# Exact key order + shapes generate_test_data.py wrote them in
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

raw = np.fromfile(SRC_BIN, dtype='<f4')  # read everything as float32 first
raw_i32 = np.fromfile(SRC_BIN, dtype='<i4')  # and as int32, for the indices block

state = {}
offset = 0
for name, shape, dtype in layout:
    n = int(np.prod(shape))
    if dtype == torch.int32:
        chunk = raw_i32[offset:offset+n].astype(np.int64)  # buffer is int64 in the model
        t = torch.from_numpy(chunk).reshape(shape)
    else:
        chunk = raw[offset:offset+n].astype(np.float32)
        t = torch.from_numpy(chunk.copy()).reshape(shape)
    state[name] = t
    offset += n

assert offset == 21124, f"consumed {offset} floats, expected 21124 -- layout mismatch"

# Sanity check: load into a real model to make sure shapes/keys match exactly
model = GalaxyClassifierS4D(colored=False)
missing, unexpected = model.load_state_dict(state, strict=True)
print("Loaded cleanly. Saving checkpoint...")

torch.save(model.state_dict(), OUT_PTH)
print(f"Wrote {OUT_PTH}")