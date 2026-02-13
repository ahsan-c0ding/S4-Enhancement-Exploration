"""
Galaxy Classification Model Package

This package provides a complete implementation of a Structured State Space (S4D)
model for galaxy morphology classification. It includes:

- GalaxyClassifierS4D: Main S4D-based classifier for galaxy images
- ModelInterface: Unified interface supporting Python and RISC-V implementations
- GalaxyExplorerGUI: Interactive visualization tool for model predictions
- functions: Utility module for data loading and model export

Components
----------
- s4d: Diagonal Structured State Space (S4D) layer implementation
- hilbert: Hilbert curve scanning for spatial locality preservation
- tlts: Utility for extracting final timestep from sequences

Example Usage
-------------
>>> from model import GalaxyClassifierS4D
>>> model = GalaxyClassifierS4D(colored=True, num_classes=4)
>>> predictions = model(images)
"""

## Expose the main model classes for easy import
from .gclassifier import GalaxyClassifierS4D

## Export 'functions' as a submodule
from . import functions
from .interface import ModelInterface
from .gui import GalaxyExplorerGUI

__all__ = [
    'GalaxyClassifierS4D',
    'functions',
    'ModelInterface',
    'GalaxyExplorerGUI'
]