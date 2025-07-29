"""
InterpretSR: Neural Networks with Symbolic Regression for Interpretable Machine Learning

InterpretSR combines PyTorch MLPs with PySR (Python Symbolic Regression) to automatically 
discover symbolic expressions that approximate learned neural network behavior.
"""

__version__ = "1.0.0"
__author__ = "InterpretSR Team"

from src.mlp_sr import MLP_SR
from src.utils import load_existing_weights, load_existing_weights_auto

__all__ = [
    "MLP_SR",
    "load_existing_weights", 
    "load_existing_weights_auto"
]