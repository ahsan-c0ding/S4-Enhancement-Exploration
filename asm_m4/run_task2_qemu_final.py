import os
import re
import struct
import subprocess
import time
import numpy as np

NUM_SAMPLES = 10
WORKSPACE_DIR = os.path.expanduser("~/CAAL_M3_WORKSPACE/CAAL-S4-Galaxy")
EXPECTED_FLOATS = 4096 + 262144 + 262144 + 262144 + 64 + 4

# Rubric Tolerances
TOLERANCES = {
    "Hilbert Scan":       {"mse": 1e-12, "mae": None},
    "Linear Projection":  {"mse": 1e-8,  "mae": 1e-6},
    "S4D Layer 1 + GELU": {"mse": 1e-7,  "mae": 1e-4},
    "S4D Layer 2 + GELU": {"mse": 1e-7,  "mae": 1e-4},
    "TakeLastTimestep":   {"mse": 1e-12, "mae": None},
    "Softmax Logits":     {"mse": 1e-8,  "mae": 1e-4}
}

def hex_to_float32_array(hex_list):
    floats = [struct.unpack('>f', bytes.fromhex(h))[0] for h in hex_list]
    return np.array(floats, dtype=np.float32)

def load_ref_bin(filepath, num_floats):
    if not os.path.exists(filepath): return None
    with open(filepath, "rb") as f:
        return np.array(struct.unpack(f'<{num_floats}f', f.read(num_floats * 4)), dtype=np.float32)

def calc_mse(arr_riscv, arr_ref):
    if arr_ref is None: return 999.99
    return float(np.mean((arr_riscv - arr_ref)**2))

def calc_mae(arr_riscv, arr_ref):
    if arr_ref is None: return 999.99
    return float(np.mean(np.abs(arr_riscv - arr_ref)))

def evaluate_layer(name, mse, mae):
    target_mse = TOLERANCES[name]["mse"]
    target_mae = TOLERANCES[name]["mae"]
    
    pass_mse = mse < target_mse
    pass_mae = True if target_mae is None else (mae < target_mae)
    status = "PASS" if (pass_mse and pass_mae) else "FAIL"
    
    # Custom note for TakeLastTimestep cumulative error
    if name == "TakeLastTimestep" and status == "FAIL":
        status = "FAIL*"
        
    return {
        "mse": mse, "mae": mae, 
        "t_mse": target_mse, "t_mae": target_mae, 
        "status": status
    }

print("\n INITIATING AUTOMATED TASK 2 VALIDATION SUITE (ALL REQUIREMENTS) ")
start_time = time.time()

master_results = []
e2e_agreements = 0

for i in range(NUM_SAMPLES):
    print("="*70)
    print(f"[*] PROCESSING SAMPLE {i}...")
    sample_start = time.time()
    
    # 1. Stitch Files
    data_s = f"data_{i}.s"
    with open("qemu_compile.s", "w") as outfile:
        for fname in ["main_qemu.s", "nn_qemu.s", "math_qemu.s", data_s]:
            with open(fname, "r") as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    # 2. Compile
    os.makedirs("build/exe", exist_ok=True)
    compile_cmd = [
        "riscv32-unknown-elf-gcc", "-march=rv32gcv", "-mabi=ilp32f", 
        "-T", "veer/link.ld", "-o", "build/exe/qemu_compile.exe", 
        "qemu_compile.s", "-nostartfiles", "-lm"
    ]
    subprocess.run(compile_cmd, capture_output=True, text=True)

    # 3. Run QEMU
    qemu_path = "./qemu/bin/qemu-riscv32" if os.path.exists("./qemu/bin/qemu-riscv32") else "qemu-riscv32"
    print(f"[*] Executing via QEMU (Extracting 790k+ floats)...", flush=True)
    try:
        qemu_proc = subprocess.run([qemu_path, "-cpu", "rv32,v=true", "build/exe/qemu_compile.exe"], capture_output=True, text=True, timeout=180)
        raw_output = qemu_proc.stdout + qemu_proc.stderr
    except subprocess.TimeoutExpired:
        print(f" QEMU Timed Out for Sample {i}.")
        continue

    # 4. Parse QEMU Output
    hex_lines = re.findall(r'^[0-9A-F]{8}$', raw_output, re.MULTILINE)
    if len(hex_lines) < EXPECTED_FLOATS:
        print(f" ERROR: Missing floats in Sample {i}.")
        continue
    
    hex_lines = hex_lines[:EXPECTED_FLOATS] 
    rv_hilbert = hex_to_float32_array(hex_lines[:4096])
    rv_proj    = hex_to_float32_array(hex_lines[4096:266240])
    rv_s4d1    = hex_to_float32_array(hex_lines[266240:528384])
    rv_s4d2    = hex_to_float32_array(hex_lines[528384:790528])
    rv_pooled  = hex_to_float32_array(hex_lines[790528:790592])
    rv_softmax = hex_to_float32_array(hex_lines[790592:])
    rv_class = int(np.argmax(rv_softmax))

    # 5. Load PyTorch References
    test_dir = f"{WORKSPACE_DIR}/test_data"
    pt_hilbert = load_ref_bin(f"{test_dir}/sample_{i}_hilbert.bin", 4096)
    pt_proj    = load_ref_bin(f"{test_dir}/sample_{i}_uproject.bin", 262144)
    pt_s4d1    = load_ref_bin(f"{test_dir}/sample_{i}_gelu_1.bin", 262144)
    pt_s4d2    = load_ref_bin(f"{test_dir}/sample_{i}_gelu_2.bin", 262144)
    pt_pooled  = load_ref_bin(f"{test_dir}/sample_{i}_takelast.bin", 64)
    pt_softmax = load_ref_bin(f"{test_dir}/sample_{i}_softmax.bin", 4)
    
    pt_class = int(np.argmax(pt_softmax)) if pt_softmax is not None else -1

    # 6. Evaluate all layers
    layers = {
        "Hilbert Scan":       evaluate_layer("Hilbert Scan",       calc_mse(rv_hilbert, pt_hilbert), calc_mae(rv_hilbert, pt_hilbert)),
        "Linear Projection":  evaluate_layer("Linear Projection",  calc_mse(rv_proj, pt_proj),       calc_mae(rv_proj, pt_proj)),
        "S4D Layer 1 + GELU": evaluate_layer("S4D Layer 1 + GELU", calc_mse(rv_s4d1, pt_s4d1),       calc_mae(rv_s4d1, pt_s4d1)),
        "S4D Layer 2 + GELU": evaluate_layer("S4D Layer 2 + GELU", calc_mse(rv_s4d2, pt_s4d2),       calc_mae(rv_s4d2, pt_s4d2)),
        "TakeLastTimestep":   evaluate_layer("TakeLastTimestep",   calc_mse(rv_pooled, pt_pooled),   calc_mae(rv_pooled, pt_pooled)),
        "Softmax Logits":     evaluate_layer("Softmax Logits",     calc_mse(rv_softmax, pt_softmax), calc_mae(rv_softmax, pt_softmax))
    }

    e2e_match = "PASS" if rv_class == pt_class else "FAIL"
    if e2e_match == "PASS": e2e_agreements += 1

    master_results.append({
        "sample": i,
        "rv_probs": rv_softmax.tolist(),
        "rv_class": rv_class,
        "pt_class": pt_class,
        "e2e": e2e_match,
        "layers": layers
    })
    
    print(f"[+] Sample {i} Evaluated in {time.time()-sample_start:.1f}s | E2E Match: {e2e_match}")

print("\n" + "="*70)
print(f" ALL SIMULATIONS COMPLETED IN {(time.time()-start_time)/60:.1f} MINS.")
print(f" End-to-End Agreement: {e2e_agreements}/{NUM_SAMPLES} ({(e2e_agreements/NUM_SAMPLES)*100}%)")
print("="*70 + "\n")

# =========================================================================
# GENERATE LATEX TABLE 1: PROBABILITIES & END-TO-END
# =========================================================================
latex_table_1 = r"""
\begin{table}[h!]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Sample} & \textbf{RISC-V Predicted Probabilities (C0, C1, C2, C3)} & \textbf{RV Class} & \textbf{PT Class} & \textbf{E2E Match} \\
\hline
"""
for res in master_results:
    probs_str = f"[{res['rv_probs'][0]:.4f}, {res['rv_probs'][1]:.4f}, {res['rv_probs'][2]:.4f}, {res['rv_probs'][3]:.4f}]"
    latex_table_1 += f"{res['sample']} & {probs_str} & {res['rv_class']} & {res['pt_class']} & {res['e2e']} \\\\\n"

latex_table_1 += r"""\hline
\end{tabular}
}
\caption{End-to-End Inference Validation: Probabilities and Class Agreement}
\label{tab:e2e_validation}
\end{table}
"""
print(latex_table_1)

# =========================================================================
# GENERATE LATEX TABLE 2: MEGA PER-LAYER ERROR METRICS
# =========================================================================
latex_table_2 = r"""
\begin{table}[h!]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{|c|l|c|c|c|c|c|}
\hline
\textbf{Sample} & \textbf{Layer} & \textbf{MSE} & \textbf{Target MSE} & \textbf{MAE} & \textbf{Target MAE} & \textbf{Status} \\
\hline
"""
for res in master_results:
    sample_id = res['sample']
    for idx, (layer_name, metrics) in enumerate(res['layers'].items()):
        # Only print the Sample ID on the first row of its block
        disp_sample = f"\multirow{{6}}{{*}}{{{sample_id}}}" if idx == 0 else ""
        
        t_mae_str = f"$< 10^{{{int(np.log10(metrics['t_mae']))}}}$" if metrics['t_mae'] is not None else "N/A"
        t_mse_str = f"$< 10^{{{int(np.log10(metrics['t_mse']))}}}$"
        
        latex_table_2 += f"{disp_sample} & {layer_name} & {metrics['mse']:.2e} & {t_mse_str} & {metrics['mae']:.2e} & {t_mae_str} & {metrics['status']} \\\\\n"
    latex_table_2 += "\\hline\n"

latex_table_2 += r"""\end{tabular}
}
\caption{Comprehensive Per-Layer Validation Metrics Across All 10 Test Samples.}
\label{tab:per_layer_metrics}
\vspace{1ex}
{\footnotesize \textit{*Note: TakeLastTimestep inherits upstream S4D error since it is tested end-to-end, exceeding the isolated $10^{-12}$ target.}}
\end{table}
"""
print(latex_table_2)