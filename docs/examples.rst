Examples
========

This section provides detailed examples showing how to use InterpretSR in various scenarios.

Example 1: Basic Symbolic Regression
------------------------------------

This example demonstrates the basic workflow of training a neural network and discovering its symbolic representation.

.. code-block:: python

   import torch
   import torch.nn as nn
   import numpy as np
   from InterpretSR.src.mlp_sr import MLP_SR
   
   # Generate synthetic data with known ground truth: y = x0^2 + 3*sin(x1) - 2*x2
   def generate_data(n_samples=1000):
       X = torch.randn(n_samples, 3)
       y = X[:, 0]**2 + 3*torch.sin(X[:, 1]) - 2*X[:, 2]
       y = y.unsqueeze(1) + torch.randn(n_samples, 1) * 0.1  # Add noise
       return X, y
   
   # Create training data
   X_train, y_train = generate_data(1000)
   X_test, y_test = generate_data(200)
   
   # Define MLP architecture
   mlp = nn.Sequential(
       nn.Linear(3, 64),
       nn.ReLU(),
       nn.Linear(64, 32),
       nn.ReLU(),
       nn.Linear(32, 1)
   )
   
   # Wrap with MLP_SR
   model = MLP_SR(mlp, "polynomial_sin_model")
   
   # Train the model
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   
   for epoch in range(200):
       optimizer.zero_grad()
       predictions = model(X_train)
       loss = criterion(predictions, y_train)
       loss.backward()
       optimizer.step()
       
       if epoch % 50 == 0:
           print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
   
   # Discover symbolic expression
   print("Discovering symbolic expression...")
   regressor = model.interpret(X_train, niterations=500)
   
   # Evaluate both neural network and symbolic predictions
   model.eval()
   with torch.no_grad():
       nn_predictions = model(X_test)
       nn_mse = criterion(nn_predictions, y_test)
       print(f"Neural Network Test MSE: {nn_mse.item():.6f}")
   
   # Switch to symbolic equation and evaluate
   success = model.switch_to_equation()
   if success:
       with torch.no_grad():
           symbolic_predictions = model(X_test)
           symbolic_mse = criterion(symbolic_predictions, y_test)
           print(f"Symbolic Equation Test MSE: {symbolic_mse.item():.6f}")
   
   print(f"Best equation: {regressor.get_best()['equation']}")

Example 2: Complex Model with Multiple MLPs
-------------------------------------------

This example shows how to use MLP_SR with more complex architectures that contain multiple MLP components.

.. code-block:: python

   import torch
   import torch.nn as nn
   from InterpretSR.src.mlp_sr import MLP_SR
   from InterpretSR.src.utils import load_existing_weights
   
   class ComplexModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Create separate MLPs for different parts of the model
           encoder_mlp = nn.Sequential(
               nn.Linear(10, 32),
               nn.ReLU(),
               nn.Linear(32, 16)
           )
           
           decoder_mlp = nn.Sequential(
               nn.Linear(16, 8),
               nn.ReLU(),  
               nn.Linear(8, 1)
           )
           
           # Wrap MLPs with MLP_SR
           self.encoder = MLP_SR(encoder_mlp, "encoder")
           self.decoder = MLP_SR(decoder_mlp, "decoder")
           
       def forward(self, x):
           encoded = self.encoder(x)
           output = self.decoder(encoded)
           return output
   
   # Create and train model
   model = ComplexModel()
   X = torch.randn(500, 10)
   y = torch.randn(500, 1)
   
   # Training loop (simplified)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   
   for epoch in range(100):
       optimizer.zero_grad()
       predictions = model(X)
       loss = criterion(predictions, y)
       loss.backward()
       optimizer.step()
   
   # Interpret different components separately
   print("Interpreting encoder...")
   encoder_regressor = model.encoder.interpret(X, niterations=300)
   
   # Get encoder output for decoder interpretation
   model.eval()
   with torch.no_grad():
       encoder_output = model.encoder(X)
   
   print("Interpreting decoder...")
   decoder_regressor = model.decoder.interpret(encoder_output, niterations=300)
   
   # Switch components to symbolic mode
   model.encoder.switch_to_equation()
   model.decoder.switch_to_equation()
   
   print(f"Encoder equation: {encoder_regressor.get_best()['equation']}")
   print(f"Decoder equation: {decoder_regressor.get_best()['equation']}")

Example 3: Hyperparameter Tuning for Symbolic Regression
--------------------------------------------------------

This example demonstrates how to tune PySR parameters for better symbolic discovery.

.. code-block:: python

   import torch
   import torch.nn as nn
   from InterpretSR.src.mlp_sr import MLP_SR
   
   # Create model and data
   mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))
   model = MLP_SR(mlp, "tuned_model")
   
   X = torch.randn(800, 4)
   y = torch.sin(X[:, 0]) * torch.exp(X[:, 1]) + X[:, 2] / (1 + X[:, 3]**2)
   y = y.unsqueeze(1)
   
   # Train model (simplified)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.MSELoss()
   
   for epoch in range(150):
       optimizer.zero_grad()
       predictions = model(X)
       loss = criterion(predictions, y)
       loss.backward()
       optimizer.step()
   
   # Try different PySR configurations
   configs = [
       {
           "name": "Simple operators",
           "params": {
               "binary_operators": ["+", "*"],
               "unary_operators": ["sin", "exp"],
               "niterations": 400
           }
       },
       {
           "name": "Extended operators", 
           "params": {
               "binary_operators": ["+", "-", "*", "/"],
               "unary_operators": ["sin", "cos", "exp", "log", "inv(x) = 1/x"],
               "niterations": 600,
               "complexity_of_operators": {"sin": 2, "cos": 2, "exp": 3, "log": 3}
           }
       },
       {
           "name": "High iterations",
           "params": {
               "binary_operators": ["+", "-", "*", "/"],
               "unary_operators": ["sin", "exp", "inv(x) = 1/x"],
               "niterations": 1000,
               "constraints": {"sin": 4, "exp": 4}
           }
       }
   ]
   
   best_score = float('inf')
   best_config = None
   best_regressor = None
   
   for config in configs:
       print(f"\\nTrying configuration: {config['name']}")
       regressor = model.interpret(X, **config['params'])
       
       best_eq = regressor.get_best()
       score = best_eq['loss']
       
       print(f"Best equation: {best_eq['equation']}")
       print(f"Score: {score:.6f}")
       
       if score < best_score:
           best_score = score
           best_config = config
           best_regressor = regressor
   
   print(f"\\nBest configuration: {best_config['name']}")
   print(f"Best equation: {best_regressor.get_best()['equation']}")
   print(f"Best score: {best_score:.6f}")

Example 4: Weight Loading and Model Adaptation
----------------------------------------------

This example shows how to load pre-trained weights into MLP_SR wrapped models.

.. code-block:: python

   import torch
   import torch.nn as nn
   from InterpretSR.src.mlp_sr import MLP_SR
   from InterpretSR.src.utils import load_existing_weights, load_existing_weights_auto
   
   # Original model architecture (for saving weights)
   class OriginalModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.mlp = nn.Sequential(
               nn.Linear(5, 32),
               nn.ReLU(),
               nn.Linear(32, 16),
               nn.ReLU(),
               nn.Linear(16, 1)
           )
           
       def forward(self, x):
           return self.mlp(x)
   
   # Model with MLP_SR wrapper  
   class InterpretableModel(nn.Module):
       def __init__(self):
           super().__init__()
           mlp = nn.Sequential(
               nn.Linear(5, 32),
               nn.ReLU(),
               nn.Linear(32, 16), 
               nn.ReLU(),
               nn.Linear(16, 1)
           )
           self.mlp = MLP_SR(mlp, "main_mlp")
           
       def forward(self, x):
           return self.mlp(x)
   
   # Simulate having a pre-trained model
   original_model = OriginalModel()
   X = torch.randn(100, 5)
   y = torch.randn(100, 1)
   
   # Train original model briefly
   optimizer = torch.optim.Adam(original_model.parameters(), lr=0.01)
   criterion = nn.MSELoss()
   for _ in range(50):
       optimizer.zero_grad()
       loss = criterion(original_model(X), y)
       loss.backward()
       optimizer.step()
   
   # Save the trained weights
   torch.save(original_model.state_dict(), "pretrained_model.pth")
   
   # Method 1: Manual mapping
   interpretable_model = InterpretableModel()
   adapted_weights = load_existing_weights(
       "pretrained_model.pth",
       mlp_mappings={"mlp.": "mlp.InterpretSR_MLP."}
   )
   interpretable_model.load_state_dict(adapted_weights)
   
   # Method 2: Automatic detection
   interpretable_model_auto = InterpretableModel()
   auto_weights = load_existing_weights_auto("pretrained_model.pth", interpretable_model_auto)
   interpretable_model_auto.load_state_dict(auto_weights)
   
   # Now you can use the pre-trained model for symbolic regression
   regressor = interpretable_model.mlp.interpret(X, niterations=300)
   print(f"Discovered equation: {regressor.get_best()['equation']}")

Running the Examples
-------------------

To run these examples:

1. Make sure you have InterpretSR installed with all dependencies
2. Copy the example code into a Python script or Jupyter notebook
3. Run the code - note that symbolic regression can take several minutes depending on the configuration
4. Experiment with different architectures, data, and PySR parameters

The examples are designed to be educational and demonstrate best practices for using InterpretSR in real-world scenarios.