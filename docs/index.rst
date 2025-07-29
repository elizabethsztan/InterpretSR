InterpretSR Documentation
=========================

InterpretSR is a Python library that combines neural networks with symbolic regression for interpretable machine learning. 
The project wraps PyTorch MLPs with PySR (Python Symbolic Regression) to automatically discover symbolic expressions 
that approximate the learned neural network behavior.

Key Features
============

* **Neural Network Wrapping**: Seamlessly wrap any PyTorch MLP with symbolic regression capabilities
* **Automatic Discovery**: Use PySR to find mathematical expressions that approximate neural network behavior
* **Interpretability**: Convert black-box neural networks into human-readable symbolic expressions
* **Flexible Integration**: Maintain full PyTorch compatibility while adding interpretability features
* **Dynamic Switching**: Switch between neural network and symbolic equation modes during inference

Quick Start
===========

.. code-block:: python

   import torch
   import torch.nn as nn
   from InterpretSR.src.mlp_sr import MLP_SR
   
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

Installation
============

To install InterpretSR, clone the repository and install the dependencies:

.. code-block:: bash

   git clone <repository-url>
   cd InterpretSR_project
   source interpretsr_venv/bin/activate
   pip install -r requirements.txt

Or install via pip (coming soon):

.. code-block:: bash

   pip install interpretsr

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   getting_started
   api_reference
   examples
   contributing

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules

