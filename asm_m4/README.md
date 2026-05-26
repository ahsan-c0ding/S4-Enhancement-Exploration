# Milestone 4: RISC-V Vector Implementation (S4 Galaxy Classifier)

This repository contains the complete, validated RISC-V 32-bit Vector (RVV) assembly implementation of the S4D inference pipeline for Galaxy Classification. It builds upon the scalar baseline by utilizing vector strip-mining, branchless vector math approximations, and optimized memory strides to significantly reduce the dynamic instruction count.

## Repository Structure

* `/asm_m4/` - Contains all core assembly source files (`main.s`, `math.s`, `nn.s`), the data samples (`data_0.s` to `data_9.s`), and the automated Python/Bash evaluation scripts.
* `Makefile` - The build system required to compile the assembly files.
* `.gitignore` - Ensures repository hygiene by excluding compiled binaries and logs.

## Deployment and Evaluation Guide

The files in this repository are packaged for deployment into a standard `riscv-env-setup` workspace. Please follow these instructions carefully to replicate the validation results and benchmark the speedup.

### Step 1: Environment Preparation

This code requires the RISC-V GNU Toolchain, QEMU, and VeeR-iSS configured with Vector Extension (RVV 1.0) support. It is assumed you are running this within the provided Docker container (via VS Code Dev Containers) or an environment where `riscv32-unknown-elf-gcc` is in your system PATH.

Copy the contents of the deployment package directly into the root of your `riscv-env-setup` directory:

1. Copy all files from the `asm_m4` directory into the root of your environment:
`cp -r CAAL-S4-Galaxy/asm_m4/* /path/to/riscv-env-setup/`

### Step 2: Task 2 Validation (Automated QEMU Extraction)

We have provided an automated script (`run_task2_qemu_final.py`) that handles file stitching, GCC compilation (with `-march=rv32gcv`), QEMU execution, output extraction, and MSE/MAE calculation against the PyTorch reference binaries.

Navigate to the root of your `riscv-env-setup` directory and run:
`python3 run_task2_qemu_final.py`

**Expected Behavior:** The script will iterate through samples 0 to 9. For each sample, it will execute the fully vectorized forward pass via QEMU, extract the floating-point probabilities, and evaluate them. Because the code utilizes an artificial vector length throttle (capped at VL=4) to produce realistic edge-device cycle counts, QEMU will take slightly longer per sample than a pure scalar run. It will output a 10/10 PASS End-to-End match.

### Step 3: Task 3.3.1 (Static Instruction Profile)

To generate the static instruction count and family classification directly from the source code, run the following script:
`python3 m4_static_count.py`

### Step 4: Task 3.3.2 (Dynamic Instruction Profile - VeeR-iSS)

To evaluate the architectural performance and extract the dynamic instruction breakdown, we have fully automated the VeeR-iSS simulation.

To generate the profile for Sample 0, simply run the provided bash script:
`./run_final_benchmark.sh`

**Expected Behavior:** This script automatically concatenates `main.s`, `nn.s`, `math.s`, and `data_0.s`, compiles them, and executes the binary through VeeR-iSS using the `whisper.json` configuration. The simulation takes approximately 90 to 120 seconds. Once complete, it automatically triggers `parse_veer_profile.py` to print the final LaTeX table, proving the heavy utilization of V-type instructions and the massive reduction in overall instruction count.

## Troubleshooting Guide

### Issue 1: "FileNotFoundError: [Errno 2] No such file or directory: 'riscv32-unknown-elf-gcc'"

**Cause:** The Python scripts rely on the `subprocess` module to call GCC and QEMU. If you execute the scripts in a standard terminal (outside the VS Code Dev Container), Linux will not be able to locate the compiler.
**Solution A:** Ensure you have opened the `riscv-env-setup` folder in VS Code and selected "Reopen in Container". Run the script from the VS Code terminal.
**Solution B:** If you are bypassing Docker, manually edit the subprocess calls in the Python scripts to match the exact absolute path of your local toolchain (e.g., `/opt/riscv/bin/riscv32-unknown-elf-gcc`).

### Issue 2: "QEMU Timed Out for Sample X"

**Cause:** The automated testing script sets a timeout for QEMU execution. Because the M4 implementation heavily utilizes the RVV extension and explicitly throttles the Vector Length (VL) to simulate realistic hardware constraints, QEMU must translate billions of micro-operations into x86 instructions.
**Solution:** Open `run_task2_qemu_final.py`, locate the `subprocess.run` call for QEMU, and increase the timeout threshold to allow the emulation to finish.

### Issue 3: "Illegal Instruction" during QEMU or VeeR-iSS Execution

**Cause:** The simulator or compiler was not instructed to enable the Vector Extension.
**Solution:** Ensure that the `-march=rv32gcv` flag is present in all GCC compilation commands, and that QEMU is invoked with `-cpu rv32,v=true`. The provided `run_final_benchmark.sh` and Python scripts already include these flags by default.

### Issue 4: "Missing floats in Sample X" or "IndexError" during Array Slicing

**Cause:** This occurs if QEMU fails to execute completely, resulting in a truncated stdout log. This is often caused by a missing linker script.
**Solution:** Ensure that the `veer/link.ld` file is present in your environment, as the GCC compilation commands explicitly depend on it for correct memory layout.
