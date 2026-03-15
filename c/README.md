# S4 Galaxy Classifier (C Implementation)

A **C implementation** of our neural network for **galaxy morphology classification** using a **Structured State Space (S4D) model**.

The project performs **end-to-end inference on 64×64 grayscale galaxy images** without relying on external machine learning or math libraries. All numerical operations are implemented manually for portability and control over numerical precision.

The classifier predicts one of four galaxy morphology classes:

* Round Elliptical
* In-between Elliptical
* Cigar-shaped Elliptical
* Edge-on Disk

The pipeline includes **Hilbert curve scanning, sequence modeling using S4D layers, nonlinear activations, and final classification via softmax**.

---
## Quick Start (Clean Clone Setup)

To verify the implementation from a fresh clone, run the following:

```bash
# 1. Enter the source directory
cd c

# 2. Build the inference and test applications
make

# 3. Run a sample inference demo
./galaxy_app ../test_data/sample_0_img.bin
```

# Features

* Pure **C implementation** of a neural network inference pipeline
* Custom **math library** - no use of external libraries
* Implementation of **Structured State Space (S4D) sequence modeling**
* **Hilbert curve scan** for converting 2D images into 1D sequences
* End-to-end **numerical validation against Python reference**
* **Benchmarking script** for compiler optimization analysis

---




The Hilbert scan preserves spatial locality when transforming the image into a sequence of length **4096**.

---

## Repository Structure

```
.
├── c/                         # Core C implementation
│   ├── main.c                 # Inference entry point
│   ├── test.c                 # Validation entry point
│   ├── nn.c                   # Layer implementations
│   ├── nn.h                   # Layer headers
│   ├── math.c                 # Math primitives
│   ├── math.h                 # Math headers
│   ├── Makefile               # Build system
│   ├──benchmark.sh           # Optimization script
|   └── README.md                  # Usage instructions
├── model_params/              # Parameter storage
│   └── model_weights.bin      # Binary weights
├── test_data/                 # Validation split samples
│   ├── sample_*_img.bin       # Test input images
│   └── sample_*_ref.bin       # Python reference tensors
└──export/                    # Automation and charts
    ├── generate_test_data.py  # Data generation script
    ├── plot_benchmarks.py     # Timing analysis plotter 
    ├── plot_errors.py         # Error distribution plotter
    ├── plot_instructions.py   # Instruction count plotter 
    ├── run_test.py            # Batch testing script 
    └── charts/                # Figure storage 
```

External assets required at runtime:

```
../model_params/model_weights.bin
../test_data/input_image.bin
```

These files contain the **trained model parameters** and **test input image** respectively.

---

# Building the Project

Compile both applications using the provided Makefile.

```bash
make
```

This produces two executables:

```
galaxy_app
test_app
```

Clean build artifacts:

```bash
make clean
```

---

# Running Inference

Run the standalone inference application by passing a test image as an argument:

```bash
./galaxy_app ../test_data/sample_0_img.bin
```

The program performs the following:

1. Load model weights
2. Load a test input image
3. Run the full neural network forward pass
4. Output the predicted probability distribution

Example output:

```
====================================
 Galaxy Class Predictions
====================================
Class 0 [Round Elliptical]        : 18.45%
Class 1 [In-between Elliptical]   :  9.02%
Class 2 [Cigar-shaped Elliptical] : 51.68%
Class 3 [Edge-on Disk]            : 20.86%
====================================
 FINAL PREDICTION: Cigar-shaped Elliptical
====================================
```

---

# Validation

The project includes an **end-to-end validation tool** that compares C inference results with a Python reference implementation.

Run:

```bash
./test_app ../test_data/sample_0
```

The program computes:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

Example output:

```
====================================
End-to-End Pipeline Validation
====================================
Mean Squared Error: 3.14e-07
Mean Absolute Error: 2.11e-04
PASSED (Predictions match Python closely!)
====================================
```

Passing thresholds ensure numerical equivalence between implementations.

---
##  Automated Batch Validation (Task 2)
The rubric requires rigorous validation across multiple samples[cite: 751]. We provide a Python wrapper to automate layer-by-layer validation across the entire test suite. 

```bash
cd ../export
python3 run_test.py
 ```
---
##  Benchmarking (Task 3)

Compiler optimization levels can significantly affect runtime performance.
To reproduce the performance analysis and instruction reduction results found in the report
Run the benchmarking script:

```bash
cat << 'EOF' > benchmark.sh
#!/bin/bash
levels=("-O0" "-O1" "-O2" "-O3" "-Ofast")
echo "Optimization Level | Inference Time (seconds)"
echo "-------------------|------------------------"

# We define the image path here
IMAGE="../test_data/sample_0_img.bin"

for opt in "${levels[@]}"; do
    # Added -fno-strict-aliasing so GCC -O3 doesn't break your Hilbert index pointer!
    gcc $opt -fno-strict-aliasing -Wall -Wextra -o galaxy_bench main.c nn.c math.c
    
    start=$(date +%s.%N)
    # Actually pass the image argument to the program!
    ./galaxy_bench "$IMAGE" > /dev/null 2>&1
    end=$(date +%s.%N)
    
    runtime=$(echo "$end - $start" | bc)
    
    # %.3f rounds it to 3 decimal places so it looks clean
    printf "%-18s | %.3f\n" "$opt" "$runtime"
done
rm -f galaxy_bench
EOF

# Run it immediately
bash benchmark.sh
```

Example output:

```
Optimization Level | Inference Time (seconds)
-------------------|------------------------
-O0                | 2.31
-O1                | 1.42
-O2                | 0.88
-O3                | 0.81
-Ofast             | 0.78
```

This script automatically recompiles the model at different optimization levels and measures execution time.

---

# Custom Math Library

The project includes a lightweight math implementation to avoid external dependencies.

Implemented functions include:

* `my_exp` – exponential function using Taylor series
* `my_log` – logarithm via Newton's method
* `my_sin` / `my_cos` – trigonometric functions via Taylor expansion
* `my_tanh` – hyperbolic tangent
* `my_pow` – general exponentiation
* `my_sqrt` – square root via Babylonian method


---

# Numerical Precision

The implementation is validated against our Python codebase to ensure numerical correctness.

Target tolerances:

| Layer         | Tolerance   |
| ------------- | ----------- |
| Hilbert Scan  | MSE < 1e-12 |
| Linear Layers | MSE < 1e-8  |
| S4D Layers    | MSE < 1e-7  |
| GELU          | MSE < 1e-7  |
| Softmax       | MSE < 1e-8  |

---

# Requirements

* Binary model weights
* Binary test input image
* **Compiler**: GCC >= 7.0 (supporting C11 standard) 
* **Python**: Version 3.11+ (required for automated validation scripts) 
* **Tools**: Valgrind (required for instruction count analysis in Task 3) 
