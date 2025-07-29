API Reference
=============

This section contains the complete API reference for InterpretSR.

Core Classes
------------

MLP_SR
~~~~~~

.. autoclass:: mlp_sr.MLP_SR
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Weight Loading Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: utils.load_existing_weights

.. autofunction:: utils.load_existing_weights_auto

Configuration and Parameters
----------------------------

Default PySR Parameters
~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`MLP_SR.interpret` method uses the following default parameters for PySR:

- **binary_operators**: ``["+", "*"]``
- **unary_operators**: ``["inv(x) = 1/x", "sin", "exp"]``
- **extra_sympy_mappings**: ``{"inv": lambda x: 1/x}``
- **constraints**: ``{"sin": 3, "exp": 3}``
- **complexity_of_operators**: ``{"sin": 3, "exp": 3}``
- **niterations**: ``400``

You can override any of these parameters by passing them as keyword arguments to the :meth:`interpret` method.

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from InterpretSR.src.mlp_sr import MLP_SR
   import torch.nn as nn
   
   # Create and wrap MLP
   mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
   model = MLP_SR(mlp, "example_model")
   
   # Discover symbolic expressions
   regressor = model.interpret(input_data, niterations=1000)
   
   # Switch to symbolic mode
   model.switch_to_equation()

Custom PySR Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use custom operators and settings
   regressor = model.interpret(
       input_data,
       binary_operators=["+", "-", "*", "/"],
       unary_operators=["sin", "cos", "exp", "log"],
       niterations=800,
       complexity_of_operators={"sin": 2, "cos": 2, "exp": 4, "log": 4}
   )

Working with Pre-trained Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from InterpretSR.src.utils import load_existing_weights
   
   # Load weights adapted for MLP_SR wrapper
   adapted_weights = load_existing_weights("pretrained_model.pth")
   wrapped_model.load_state_dict(adapted_weights)