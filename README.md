# S4 Galaxy Classification - Team 0x43 Repository

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
├── README.md # Project overview, setup instructions, and usage guide
|
├── requirements.txt # Python dependencies required to run the project
│
├── main.py # Interactive visualization tool (e.g., exploring models/outputs)
│
├── utils.py # Helper functions used throughout the project
│
├── Hilbertplot.py # Create Hilbert Plot image into /images folder
│
├── images/ # Report figures (Hilbert scan, confusion matrix, training curves,
│ │ # recurrent-vs-causal-conv verification plot)
│
├── model/ # model implementations
│ ├── init.py # Marks this directory as a Python package
| |
│ ├── gclassifier.py # Galaxy classification model (Hilbert + S4D pipeline)
| |
│ ├── s4d_recurrent.py # S4D implementation (recurrent, diagonal -- used by gclassifier.py)
| |
│ ├── s4_conv.py # Convolution-based S4 implementation (Milestone 1, dense/non-diagonal)
| |
│ ├── s4_recurrent.py # Recurrent S4 implementation (Milestone 1, dense/non-diagonal)
| |
│ ├── hilbert.py # Hilbert curve mapping (2D image → 1D sequence)
| |
│ ├── tlts.py # TakeLastTimestep layer (extracts final sequence state)
| |
│ ├── interface.py # Unified interface for different S4 model variants
| |
│ ├── functions.py # Utility/helper functions used across models
| |
│ ├── verify_my_task.py # Script for verifying correctness of implementations
| |
│ └── gui.py # GUI components for visualization/debugging
│
├── export/ # Scripts for exporting and testing trained models
│ ├── export_weights.py # Saves trained model weights
│ │
│ ├── generate_test_data.py # Generates test inputs for evaluation
│ │
│ └── run_test.py # Runs inference using exported models
│
├── scripts/ # Training and execution scripts
│ ├── train.ipynb # Notebook-based training pipeline
│ │
│ └── train.py # Script-based training
│
├──  tests/ # Unit tests and validation scripts
| ├── test_forward.py # Tests forward pass of models
│ |
| | test_s4_equivalence.py # Verifies recurrent vs convolution S4 equivalence (Milestone 1, dense)
└─
```

## Installation

```bash
cd space-state-model
pip install -r requirements.txt
```

## Model Modules

### Core Components

**`model/gclassifier.py`** - Galaxy classifier architecture:
- `GalaxyClassifierS4D` - Main model combining Hilbert scanning, S4 layers, classification head
- Completed `forward()` method

**`model/s4d_recurrent.py`** - Diagonal S4 layer: 
- Fully implemented, recurrent forward pass (steps through the sequence instead of convolving)
- Same parameterization as the old FFT layer, so existing checkpoints load unchanged
- Study for S4 architecture patterns, ZOH discretization, diagonal parameterization

**`model/hilbert.py`** - Hilbert curve utilities: 
- `HilbertScan` - Converts 2D images to 1D sequences
- Completed `_d2xy()` method

**`model/tlts.py`** - Sequence pooling: 
- `TakeLastTimestep` - Extracts final timestep for classification
- Implemented extraction logic

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

## Fixed Constraints

These values are fixed for the required multi-milestone compatibility:

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
