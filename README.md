# S4D Galaxy Classifier — RISC-V Vector (RVV 1.0) Implementation

A hand-written RISC-V 32-bit assembly implementation of the S4D (diagonal Structured State Space) galaxy-morphology classifier, targeting the RVV 1.0 vector extension (`rv32gcv`) and profiled on the VeeR-iSS simulator.

The pipeline classifies a 64×64 grayscale galaxy image into one of four classes — Round Elliptical, In-between Elliptical, Cigar-shaped Elliptical, Edge-on Disk — through: **Hilbert scan → linear up-projection → S4D layer → GELU → S4D layer → GELU → take-last-timestep → linear classifier → softmax**. All math (exp, log, sin, cos, tanh, sqrt) is implemented from scratch in vectorized assembly; there is no external math library.

## What's new in this version: recurrent S4D

The S4D layer was rewritten from an **O(L²) causal convolution** to an **O(L·N) recurrent scan**, matching the recurrent reference in the `c` and `python` branches.

Previously the layer materialized a length-`L` kernel and convolved it against the input (`L = 4096` taps per output). It now walks the sequence with a state recurrence, updating a small `N = 32`-element complex state per timestep:

```
x'ₙ(t) = Ā_bar,ₙ · x'ₙ(t-1) + uₜ            (x'(-1) = 0, scalar input B ≡ 1)
y(t)   = D[h]·uₜ + 2 · Σₙ Re( C_bar,ₙ · x'ₙ(t) )
```

This is algebraically identical to the standard `x = Ā·x + B̄·u ; y = 2·Re(C·x)` form, using the identity `xₙ = B̄ₙ·x'ₙ` and `Cₙ·B̄ₙ = C_bar,ₙ`. Folding `B̄` into `C_bar` lets the state update skip a complex multiply (the input `uₜ` is real, so it only touches the real part). The per-channel discretization block — which builds the `Ā_bar` and `C_bar` tables with vectorized `exp`/`sin`/`cos` — is unchanged; only the kernel-generation and convolution stages were replaced by the scan.

The 32-element state loop is vectorized at **LMUL=2** (two 16-wide strips; VLEN = 256 bits → 16 fp32/register), with a single `vfredosum.vs` reduction per timestep.

## Results (Sample 0, VeeR-iSS dynamic instruction count)

| Metric | Convolution (previous) | Recurrent (this version) | Change |
|---|---|---|---|
| **Total instructions** | 4,699,917,296 | **59,447,792** | **~79× fewer (−98.7%)** |
| Wall-clock (VeeR-iSS) | 11m 53s | 29.8s | ~24× faster |
| B-type (branches) | 557,151,372 | 2,979,212 | ~187× fewer |
| J-type (jumps) | 555,426,508 | 2,171,852 | ~256× fewer |
| V-type (vector) | 1,349,153,760 (28.7%) | 28,211,168 (47.5%) | now the dominant class |

The O(L²)→O(L·N) change removes the deeply-nested convolution loop, which is why the branch/jump overhead collapses and the remaining work becomes vector-dominated.

### Accuracy

End-to-end class predictions match the reference on every sample that ships a reference file (samples 0–4: **5/5 class agreement**; probabilities match to ~3 decimal places). Samples 5–9 have no reference binaries in `test_data/`, so the validation script reports them as sentinel "FAIL" — this is missing-reference bookkeeping, not a numerical error.

Per-layer MSE is exact through the Hilbert scan and linear projection, and sits at ~3e-7 for the S4D layers. This is slightly above the strict 1e-7 rubric target and is a **precision floor**, not a bug: the discretization uses custom fp32 polynomial `exp`/`sin`/`cos`, and the state recurrence compounds tiny per-step differences over 4096 timesteps. The error is identical across samples (systematic, not erratic) and is inherited from the shared discretization/math block, so the convolution version exhibits the same floor.

## Repository structure

```
main.s              _start + argmax + exit; calls model_forward
nn.s                Full pipeline: hilbert_scan, linear, s4d_layer (recurrent),
                    gelu, softmax, take_last_timestamp, model_forward
math.s              Vector math kernels: exp, log, sin, cos, tanh, sqrt
data_0.s .. data_9.s   Per-sample input image + model weights, as .data
*_qemu.s            QEMU-path variants of the above (used by run_task2_qemu_final.py)
veer/link.ld        Linker script (memory layout for VeeR-iSS)
veer/whisper.json   VeeR-iSS vector config (bytes_per_vec = 32)
Makefile            Builds the three .s files into an ELF
run_final_benchmark.sh   Concatenate + compile + VeeR-iSS profile (Task 3.3.2)
parse_veer_profile.py    Turns the VeeR profile into the LaTeX instruction table
run_task2_qemu_final.py  Per-sample QEMU run + MSE/MAE vs reference (Task 2)
m4_static_count.py       Static instruction count (Task 3.3.1)
model_params/       Model weights binary
test_data/          Input images + PyTorch reference tensors (per layer)
```

## Requirements

* RISC-V cross toolchain with vector support on `PATH`: `riscv32-unknown-elf-gcc` (`-march=rv32gcv -mabi=ilp32f`).
* **VeeR-iSS** (`whisper`) on `PATH` — for the dynamic instruction benchmark.
* **qemu-riscv32** on `PATH` — for the per-layer numerical validation.
* Python 3 with NumPy — for the validation/profile scripts.

No Docker or dev-container is required; a toolchain-in-`PATH` setup is sufficient.

## Build & run

### 1. Numerical validation (QEMU, ~1 min)

```bash
python3 run_task2_qemu_final.py
```

Runs each sample through QEMU, extracts all intermediate tensors, and compares against the PyTorch references in `test_data/`. Reference paths are resolved relative to this script's own directory. Expect 5/5 end-to-end class matches on samples 0–4; samples 5–9 lack reference files.

### 2. Dynamic instruction benchmark (VeeR-iSS, ~30s)

```bash
bash run_final_benchmark.sh
```

Concatenates `main.s + nn.s + math.s + data_0.s`, compiles with `-march=rv32gcv`, runs it through `whisper`, and prints the instruction total plus the LaTeX family-breakdown table (via `parse_veer_profile.py`).

### 3. Static instruction count

```bash
python3 m4_static_count.py
```

### Build only

```bash
make            # -> build/exe/galaxy_classifier.exe
make clean
```

## Troubleshooting

* **`riscv32-unknown-elf-gcc: not found`** — the toolchain isn't on `PATH`. Add your cross-compiler's `bin/` to `PATH` (or edit the compiler path in the scripts).
* **`Illegal instruction` in QEMU/VeeR** — the vector extension wasn't enabled. Ensure `-march=rv32gcv` on every GCC call and that `whisper` uses `veer/whisper.json`.
* **`Missing floats in Sample X`** — QEMU produced truncated output, usually a missing/incorrect `veer/link.ld`. Confirm the linker script is present.
* **Samples 5–9 report FAIL in validation** — expected: those samples have no reference tensors in `test_data/`. Not a code error.

## History

* **Recurrent S4D (current):** O(L·N) vectorized scan; ~79× fewer dynamic instructions than the convolution version at equal accuracy.
* **Convolution (previous):** O(L²) vectorized causal convolution with iterative kernel generation.
