# S4 Galaxy Classification - Base Code Repository

This repository provides starter code, utilities, and infrastructure for implementing S4-based galaxy morphology classification. It includes data loaders, model interfaces, visualization tools, and a reference S4D implementation.
  
**Requirements:** Python 3.11+, PyTorch 2.0+, CUDA (optional)

## Overview

This base repository contains:
- **Data loaders** for GalaxyMNIST dataset
- **Model scaffolding** with TODOs for implementation
- **Reference S4D layer** (fully implemented)
- **Utility functions** for Hilbert curves and sequence processing
- **Interactive GUI** for model exploration
- **Training infrastructure** with notebook and utilities

## Repository Structure

```
space-state-model/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── main.py                        # Interactive visualization tool
├── train.ipynb                    # Training notebook
│
└── model/                         # Model implementations
    ├── __init__.py
    ├── gclassifier.py             # Galaxy classifier (TODO: forward pass)
    ├── s4d.py                     # S4D reference implementation
    ├── hilbert.py                 # Hilbert curve (TODO: _d2xy method)
    ├── tlts.py                    # TakeLastTimestep (TODO: forward)
    ├── interface.py               # Unified model interface (M3/M4)
    ├── functions.py               # Utility functions
    └── gui.py                     # GUI components
```

## Installation

```bash
cd space-state-model
pip install -r requirements.txt
```

## Model Modules

### Core Components

**`model/gclassifier.py`** - Galaxy classifier architecture
- `GalaxyClassifierS4D` - Main model combining Hilbert scanning, S4 layers, classification head
- TODO: Complete `forward()` method

**`model/s4d.py`** - Diagonal S4 layer
- Fully implemented reference implementation
- Study for S4 architecture patterns, FFT convolution, diagonal parameterization

**`model/hilbert.py`** - Hilbert curve utilities
- `HilbertScan` - Converts 2D images to 1D sequences
- TODO: Complete `_d2xy()` method

**`model/tlts.py`** - Sequence pooling
- `TakeLastTimestep` - Extracts final timestep for classification
- TODO: Implement extraction logic

**`model/functions.py`** - Helper utilities
- Matrix operations, discretization methods

## Training

Interactive training notebook with step-by-step explanations:

```bash
jupyter notebook train.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model initialization
- Training loop with validation
- Logging and visualization
- TODO markers for required implementations

## Interactive Visualization Tool

Launch the interactive galaxy explorer GUI:

```bash
python main.py --python -m galaxy_s4_model.pth
```

Full usage:

```
usage: main.py [-h] (--python | --riscv) [--model-path MODEL_PATH] [--colored] [--data-dir DATA_DIR]

Interactive Galaxy Classification Visualization Tool

options:
  -h, --help            show this help message and exit
  --python, -p          Use Python model implementation
  --riscv               Use RISC-V model implementation
  --model-path MODEL_PATH, -m MODEL_PATH
                        Path to trained model file (default: galaxy_s4_model.pth)
  --colored, -c         Use colored (RGB) images instead of grayscale (default: grayscale)
  --data-dir DATA_DIR   Root directory for dataset (default: ./data)

Examples:
  main.py --python -m galaxy_model.pth
  main.py -p -m galaxy_model.pth --colored
  main.py --riscv
```

### Controls

- **LEFT Arrow** - Previous sample
- **RIGHT Arrow** - Next sample
- **R** - Random sample
- **Q** - Quit

## Implementation Tasks

Primary TODOs:

1. **`model/gclassifier.py`** - Complete `GalaxyClassifierS4D.forward()`
   - Connect Hilbert scanning, linear projection, S4 blocks, GELU activation, final timestep extraction, classification head
   - Handle tensor shapes: (B, C, 64, 64) → (B, 4)

2. **`model/hilbert.py`** - Implement `_d2xy()`
   - Convert 1D distance along Hilbert curve to 2D (x, y) coordinates
   - Recursive algorithm for arbitrary power-of-two grid sizes

3. **`model/tlts.py`** - Implement `TakeLastTimestep.forward()`
   - Extract final timestep from sequence tensor
   - Shape: (B, L, D) → (B, D)

4. **`train.ipynb`** - Fill TODO sections
   - Training loop implementation
   - Validation/test evaluation
   - Visualization functions

## Fixed Constraints

Do not modify these values (required for multi-milestone compatibility):

- `d_model = 64` - Hidden dimension
- `d_state = 64` - State space dimension
- `image_size = 64` - Image resolution
- `num_classes = 4` - Galaxy morphology classes

## Dependencies

Key packages:
- `torch` - Deep learning framework
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `pygame` - GUI framework
- `einops` - Tensor operations
- `galaxy_mnist` - Dataset loader

## Support

**Technical Questions:** s.taha.29208@khi.iba.edu.pk

**References:**
- Gu et al. (2022) - "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR)
- Gu et al. (2022) - "On the Parameterization and Initialization of Diagonal State Space Models" (NeurIPS)
