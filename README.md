# InterpretSR

InterpretSR is a Python library that combines neural networks with symbolic regression for interpretable machine learning. It wraps PyTorch MLPs with PySR (Python Symbolic Regression) to automatically discover symbolic expressions that approximate the learned neural network behavior.

## Features

- **Neural Network Wrapping**: Seamlessly wrap any PyTorch MLP with symbolic regression capabilities
- **Automatic Discovery**: Use PySR to find mathematical expressions that approximate neural network behavior  
- **Interpretability**: Convert black-box neural networks into human-readable symbolic expressions
- **Dynamic Switching**: Switch between neural network and symbolic equation modes during inference

## Installation

```bash
pip install interpretsr
```

## Quick Start

```python
import torch
import torch.nn as nn
from src.mlp_sr import MLP_SR

# Create a simple MLP
mlp = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Wrap with MLP_SR
wrapped_model = MLP_SR(mlp, "my_model")

# Train normally
inputs = torch.randn(100, 5)
outputs = wrapped_model(inputs)

# Discover symbolic expression
regressor = wrapped_model.interpret(inputs)

# Switch to symbolic equation
wrapped_model.switch_to_equation()
```

## Documentation

Full documentation is available at [ReadTheDocs](https://interpretsr.readthedocs.io/).

## License

MIT License