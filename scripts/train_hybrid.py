#!/usr/bin/env python
"""
Train and compare GalaxyClassifierS4D (pure Hilbert-scan -> S4D baseline)
against GalaxyClassifierCNNS4D (CNN-stem -> S4D hybrid) under identical
conditions, and dump the metrics/plots called for in the experiment writeup.

Run from the repo root:
    python scripts/train_hybrid.py

Config below is copied verbatim from scripts/train.py's baseline run (same
RNG_SEED, BATCH_SIZE, optimizer/lr, loss, split, COLORED) so the comparison
is apples-to-apples. EPOCHS is set to 30, not the notebook's placeholder of
10 -- model_params/galaxys4-30EPOCH-STANDARD.pth (the checkpoint actually
shipped in this repo) confirms 30 epochs is what the baseline was last
trained with, and the task brief says to use that higher value if found.

This script does NOT duplicate the training loop: it imports train() from
scripts/train.py directly (by file path, so it works regardless of how this
script is invoked, and without triggering that file's own notebook-style
body, which is now guarded behind `if __name__ == "__main__":`).
"""
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Make sure the repo root is importable regardless of cwd (mirrors the
# sys.path handling already done in model/__init__.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GalaxyClassifierS4D, GalaxyClassifierCNNS4D
from model.functions import load_data

# --- Reuse train() from scripts/train.py unmodified, without re-running ---
# --- that file's own guarded (if __name__ == "__main__") script body.   ---
_spec = importlib.util.spec_from_file_location(
    "baseline_train_module", os.path.join(os.path.dirname(__file__), "train.py")
)
_baseline_train_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline_train_module)
train = _baseline_train_module.train

# ---------------------------------------------------------------------
# Config -- copied verbatim from scripts/train.py's baseline run so the
# comparison is apples-to-apples (same seed/split/optimizer/epochs/batch
# size/loss, same machine/run).
# ---------------------------------------------------------------------
RNG_SEED = 42
BATCH_SIZE = 16
LR = 0.0015
EPOCHS = 30  # see module docstring for why this isn't the notebook's 10
COLORED = False
CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 64
D_STATE = 64


def set_seed(seed=RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def s4d_loop_ops(seq_len, d_model=D_MODEL, d_state=D_STATE):
    """
    Combined op count of both stacked S4D layers (s4_1, s4_2), per the
    FLOPS comment in model/gclassifier.py:
        2 * (seq_len * (d_state // 2) * d_model * 8)
    The leading 2 accounts for the two stacked layers; the trailing 8 is
    ~2 complex MACs per state element per step, at ~4 real ops/complex MAC.
    This is the term that's linear in seq_len -- the one the CNN stem cuts.
    """
    return 2 * (seq_len * (d_state // 2) * d_model * 8)


def stem_conv_macs(model):
    """
    Sum over the CNN stem's conv layers of out_H*out_W*out_C*in_C*k*k.
    Returns 0 for the pure-S4D baseline, which has no conv stem.
    """
    stem = getattr(model, "cnn_stem", None)
    if stem is None:
        return 0
    total = 0
    in_hw = 64
    for conv in (stem.conv1, stem.conv2):
        if conv is None:
            continue
        # stride=2, kernel=3, padding=1 -> exact halving for even in_hw
        out_hw = in_hw // 2
        k = conv.kernel_size[0]
        total += out_hw * out_hw * conv.out_channels * conv.in_channels * k * k
        in_hw = out_hw
    return total


def total_est_ops(model, seq_len):
    """
    Total estimated ops per forward pass:
      - baseline: uproject Linear (seq_len * C * d_model) + S4D-loop ops
        + classifier head (d_model * num_classes)
      - hybrid: stem conv MACs (no separate uproject -- the stem's last
        conv already projects to d_model) + S4D-loop ops + classifier head
    """
    stem_macs = stem_conv_macs(model)
    if stem_macs == 0:
        proj_ops = seq_len * model.hilbert_channels * model.d_model
    else:
        proj_ops = stem_macs
    s4d_ops = s4d_loop_ops(seq_len, d_model=model.d_model)
    fc_ops = model.d_model * model.fc.out_features
    return proj_ops + s4d_ops + fc_ops, s4d_ops


def measure_inference_time(model, colored, n_samples=100):
    """Mean per-sample CPU inference time (eval mode, batch size 1)."""
    model_cpu = model.to("cpu")
    model_cpu.eval()
    c = 3 if colored else 1
    x = torch.randn(1, c, 64, 64)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model_cpu(x)
        times = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            model_cpu(x)
            times.append(time.perf_counter() - t0)
    model.to(DEVICE)
    return float(np.mean(times))


def run_experiment(model, name, train_loader, val_loader, test_loader, epochs):
    set_seed(RNG_SEED)  # re-seed before each run so init/shuffling is reproducible per model
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'='*70}\nTraining: {name}\n{'='*70}")
    t0 = time.time()
    hist = train(train_loader, val_loader, model, optimizer, loss_fn, epochs, DEVICE, verbose=True)
    train_time = time.time() - t0

    # Held-out test set evaluation
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs, return_logits=True)
            preds = torch.argmax(logits, dim=1)
            target = torch.argmax(labels, dim=1)
            correct += (preds == target).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_acc = correct / total
    cm = confusion_matrix(all_targets, all_preds)

    inf_time = measure_inference_time(model, colored=COLORED, n_samples=100)
    n_params = count_params(model)
    seq_len = getattr(model, "seq_len", 4096)
    total_ops, s4d_ops = total_est_ops(model, seq_len)

    print(f"{name}: test_acc={test_acc*100:.2f}%  params={n_params:,}  "
          f"S4D-loop ops={s4d_ops:,}  train_time={train_time:.1f}s  "
          f"inference/sample={inf_time*1000:.3f}ms")

    return {
        "name": name,
        "seq_len": seq_len,
        "params": n_params,
        "test_acc": test_acc,
        "s4d_loop_ops": s4d_ops,
        "total_est_ops": total_ops,
        "train_time_sec": train_time,
        "inference_time_sec_per_sample": inf_time,
        "history": hist,
        "confusion_matrix": cm.tolist(),
    }


def print_results_table(results):
    print("\n| Model | Seq Len | Params | Test Acc | S4D-loop ops | Total est. ops | "
          "Train time (s) | Inference/sample (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        train_time_str = f"{r['train_time_sec']:.1f}" if r['train_time_sec'] is not None else "n/a (loaded checkpoint)"
        print(f"| {r['name']} | {r['seq_len']} | {r['params']:,} | {r['test_acc']*100:.2f}% | "
              f"{r['s4d_loop_ops']:,} | {r['total_est_ops']:,} | {train_time_str} | "
              f"{r['inference_time_sec_per_sample']*1000:.3f} |")

def plot_training_curves(results, out_path="training_curves_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        if r["history"] is None:   # e.g. baseline loaded from checkpoint, not retrained
            continue
        axes[0].plot(r["history"]["loss"], label=r["name"])
        axes[1].plot(r["history"]["val_accuracy"], label=r["name"])
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_confusion_matrices(results, out_path="confusion_matrices_comparison.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{r['name']}\nTest Acc: {r['test_acc']*100:.2f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def main():
    set_seed(RNG_SEED)
    print(f"Using device: {DEVICE}")

    X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
    NUM_CLASSES = y_onehot.shape[1]

    x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
        X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    train_ds = TensorDataset(x_train, y_train_onehot)
    val_ds = TensorDataset(x_val, y_val_onehot)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    X_test, y_test_onehot, y_test = load_data(root="./data", download=True, train=False, colored=COLORED)
    test_ds = TensorDataset(X_test, y_test_onehot)
    test_loader = DataLoader(test_ds, batch_size=64)

    results = []

    # --- Baseline: load existing checkpoint instead of retraining ---
    baseline = GalaxyClassifierS4D(num_classes=NUM_CLASSES, colored=COLORED)
    baseline.load_state_dict(torch.load(
        os.path.join(_REPO_ROOT, "model_params", "galaxys4-30EPOCH-STANDARD.pth"),
        map_location=DEVICE,
    ))
    baseline = baseline.to(DEVICE)
    baseline.eval()

    # Evaluate on test set only (no training loop, no history)
    correct, total = 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = baseline(imgs, return_logits=True)
            preds = torch.argmax(logits, dim=1)
            target = torch.argmax(labels, dim=1)
            correct += (preds == target).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    baseline_test_acc = correct / total
    baseline_cm = confusion_matrix(all_targets, all_preds)
    seq_len = 4096
    total_ops, s4d_ops = total_est_ops(baseline, seq_len)

    results.append({
        "name": "S4D baseline (seq_len=4096)",
        "seq_len": seq_len,
        "params": count_params(baseline),
        "test_acc": baseline_test_acc,
        "s4d_loop_ops": s4d_ops,
        "total_est_ops": total_ops,
        "train_time_sec": None,   # not retrained this run
        "inference_time_sec_per_sample": measure_inference_time(baseline, colored=COLORED, n_samples=100),
        "history": None,          # no training curve to plot for this row
        "confusion_matrix": baseline_cm.tolist(),
    })

    # --- Hybrid, primary: CNN stem (16x) + S4D ---
    hybrid16 = GalaxyClassifierCNNS4D(num_classes=NUM_CLASSES, colored=COLORED, stem_reduction=16)
    results.append(run_experiment(
        hybrid16, "CNN stem (16x) + S4D (seq_len=256)",
        train_loader, val_loader, test_loader, EPOCHS,
    ))

    # --- Hybrid, fallback: CNN stem (4x) + S4D ---
    hybrid4 = GalaxyClassifierCNNS4D(num_classes=NUM_CLASSES, colored=COLORED, stem_reduction=4)
    results.append(run_experiment(
        hybrid4, "CNN stem (4x) + S4D (seq_len=1024)",
        train_loader, val_loader, test_loader, EPOCHS,
    ))

    print_results_table(results)

    with open("results_table.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_table.json")

    plot_training_curves(results)
    plot_confusion_matrices(results)


if __name__ == "__main__":
    main()
