# Milestone 3: RISC-V Scalar Implementation (S4 Galaxy Classifier)

This repository contains the complete, validated RISC-V 32-bit assembly implementation of the S4D inference pipeline for Galaxy Classification. It includes all required mathematical functions, memory layout management (excluding heap usage), and instruction-level profiling scripts.

## Repository Structure
* `/asm/` - Contains all core assembly source files (`main.s`, `math.s`, `nn.s`), QEMU evaluation variants, and automated evaluation scripts.
* `/test_data/` - Contains the binary PyTorch reference data used for end-to-end validation testing across 10 samples.
* `Makefile` - The build system required to compile the assembly files.

## Deployment and Evaluation Guide (For Grading)

The files in this repository are packaged for deployment into a standard `riscv-env-setup` workspace. Please follow these instructions carefully to replicate the validation results.

### Step 1: Environment Preparation
This code requires the RISC-V GNU Toolchain, QEMU, and VeeR-iSS. It is assumed you are running this within the provided Docker container (via VS Code Dev Containers) or an environment where `riscv32-unknown-elf-gcc` is in your system PATH.

Copy the contents of the deployment package directly into the root of your `riscv-env-setup` directory:

1. Copy all files from the `asm` directory into the root of your environment:
   `cp -r CAAL-S4-Galaxy/asm/* /path/to/riscv-env-setup/`
2. Copy the `test_data` directory into the root of your environment:
   `cp -r CAAL-S4-Galaxy/test_data /path/to/riscv-env-setup/`

### Step 2: Task 2 Validation (Automated QEMU Extraction)
We have provided an automated script (`run_task2_qemu_final.py`) that handles file stitching, GCC compilation, QEMU execution, output extraction, and MSE/MAE calculation against the PyTorch reference binaries.

Navigate to the root of your `riscv-env-setup` directory and run:
`python3 run_task2_qemu_final.py`

**Expected Behavior:** The script will iterate through samples 0 to 9. For each sample, it will output a 790,000+ line execution trace via QEMU stdout, extract the floating-point values, and evaluate them against the references in the `test_data` folder. At the end, it will generate the LaTeX tables required for the report.

### Step 3: Task 3.4.1 (Static Instruction Profile)
To generate the static instruction count and family classification, run the following script from the root of your environment:
`python3 run_task3_static.py`

### Step 4: Task 3.4.2 (Dynamic Instruction Profile)
The dynamic instruction count requires the VeeR-iSS simulator. Because this simulation takes approximately 40 minutes per sample, it is not automated in a loop.

To generate the profile for Sample 0:
1. Stitch the files manually:
   `cat main.s nn.s math.s data_0.s > galaxy_0_veer.s`
2. Compile the binary using the provided build script or Makefile:
   `./build.sh -a galaxy_0_veer.s`
3. Execute VeeR-iSS with the profile flag:
   `./veer/build-Linux/whisper --profileinst build/logs/galaxy_0.txt build/exe/galaxy_0.exe`
4. Once the simulation completes, run the parser to generate the LaTeX table:
   `python3 parse_veer_profile.py`

## Troubleshooting Guide

### Issue 1: "FileNotFoundError: [Errno 2] No such file or directory: 'riscv32-unknown-elf-gcc'"
**Cause:** The Python scripts rely on the `subprocess` module to call GCC and QEMU. If you execute the scripts in a standard terminal (outside the VS Code Dev Container), Linux will not be able to locate the compiler.
**Solution A:** Ensure you have opened the `riscv-env-setup` folder in VS Code and selected "Reopen in Container". Run the script from the VS Code terminal.
**Solution B:** If you are bypassing Docker, manually edit lines 52 and 58 in `run_task2_qemu_final.py` to match the exact absolute path of your local toolchain (e.g., `/opt/riscv/bin/riscv32-unknown-elf-gcc`).

### Issue 2: "QEMU Timed Out for Sample X"
**Cause:** The automated testing script sets a strict 180-second timeout for QEMU execution to prevent infinite loops. If your host machine is under heavy load, printing 790,000 lines to stdout may exceed this threshold.
**Solution:** Open `run_task2_qemu_final.py`, locate the `subprocess.run` call for QEMU (around line 96), and increase `timeout=180` to `timeout=300`.

### Issue 3: "Missing floats in Sample X" or "IndexError" during Array Slicing
**Cause:** This occurs if QEMU fails to execute completely, resulting in a truncated stdout log. This is often caused by a missing linker script.
**Solution:** Ensure that the `veer/link.ld` file is present in your environment, as the Makefile and GCC compilation commands explicitly depend on it for memory layout.