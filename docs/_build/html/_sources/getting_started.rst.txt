Getting Started
===============

This guide will help you get started with InterpretSR, a library that combines neural networks with symbolic regression for interpretable machine learning.

Installation
------------

First, make sure you have Python 3.8 or higher installed. Then you can install InterpretSR and its dependencies:

.. code-block:: bash

   pip install interpretsr

Or if you're installing from source:

.. code-block:: bash

   git clone <repository-url>
   cd InterpretSR_project
   python -m venv interpretsr_venv
   source interpretsr_venv/bin/activate  # On Windows: interpretsr_venv\Scripts\activate
   pip install -r requirements.txt

Basic Usage
-----------

The core concept of InterpretSR is wrapping existing PyTorch MLPs with the :class:`MLP_SR` class to add symbolic regression capabilities.

Creating an MLP_SR Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from InterpretSR.src.mlp_sr import MLP_SR
   
   # Create your standard PyTorch MLP
   mlp = nn.Sequential(
       nn.Linear(3, 64),
       nn.ReLU(),
       nn.Linear(64, 32),
       nn.ReLU(),
       nn.Linear(32, 1)
   )
   
   # Wrap it with MLP_SR
   interpretable_model = MLP_SR(mlp, name="my_model")

Training the Model
~~~~~~~~~~~~~~~~~~

The wrapped model behaves exactly like a standard PyTorch model during training:

.. code-block:: python

   # Generate some synthetic data
   X = torch.randn(1000, 3)
   y = X[:, 0]**2 + 3*torch.sin(X[:, 1]) - 2*X[:, 2] + torch.randn(1000, 1)*0.1
   
   # Standard PyTorch training loop
   optimizer = torch.optim.Adam(interpretable_model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   
   for epoch in range(100):
       optimizer.zero_grad()
       predictions = interpretable_model(X)
       loss = criterion(predictions, y)
       loss.backward()
       optimizer.step()

Discovering Symbolic Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, use the :meth:`interpret` method to discover symbolic expressions:

.. code-block:: python

   # Discover symbolic expressions
   regressor = interpretable_model.interpret(X, niterations=500)
   
   # View the best equation found
   print("Best equation:", regressor.get_best()['equation'])
   print("Best score:", regressor.get_best()['score'])

Switching to Symbolic Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have a symbolic expression, you can switch the model to use it instead of the neural network:

.. code-block:: python

   # Switch to using the symbolic equation
   success = interpretable_model.switch_to_equation()
   
   if success:
       # Now the model uses the symbolic equation for forward passes
       symbolic_predictions = interpretable_model(X)
       
       # Switch back to neural network if needed
       interpretable_model.switch_to_mlp()

Working with Different Complexities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySR finds multiple equations of varying complexity. You can explore different options:

.. code-block:: python

   # View all discovered equations
   print(regressor.equations_[['complexity', 'loss', 'score', 'equation']])
   
   # Switch to a specific complexity level
   interpretable_model.switch_to_equation(complexity=5)

Next Steps
----------

- Check out the :doc:`examples` for more detailed use cases
- Explore the :doc:`api_reference` for complete API documentation
- See the demo notebook in the repository for interactive examples