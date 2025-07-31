"""
InterpretSR MLP_SR Module

This module provides a wrapper for PyTorch MLP models that adds symbolic regression
capabilities using PySR (Python Symbolic Regression).
"""

from pysr import *
import torch 
import torch.nn as nn
import time
import sympy
from sympy import lambdify

class MLP_SR(nn.Module):
    """
    A PyTorch module wrapper that adds symbolic regression capabilities to MLPs.
    
    This class wraps any PyTorch MLP (Multi-Layer Perceptron) and provides methods
    to discover symbolic expressions that approximate the learned neural network
    behavior using genetic algorithms supported by PySR.
    
    The wrapper maintains full compatibility with PyTorch's training pipeline while
    adding interpretability features through symbolic regression.
    
    Attributes:
        InterpretSR_MLP (nn.Module): The wrapped PyTorch MLP model
        mlp_name (str): Human-readable name for the MLP instance
        pysr_regressor (PySRRegressor): The fitted symbolic regression model
        
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from interpretsr.mlp_sr import MLP_SR
        >>> 
        >>> # Create a model
        >>> class SimpleModel(nn.Module):
                def __init__(self, input_dim, output_dim, hidden_dim = 64):
                    super(SimpleModel, self).__init__()
                    mlp = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, output_dim)
                    )
                    self.mlp = MLP_SR(mlp, mlp_name = "Sequential") # Wrap the mlp 
                    with MLP_SR and provide a label
        >>> model = SimpleModel(input_dim=5, output_dim=1) # Initialise the model
        >>> # Train the model normally
        >>> trained_model = training_function(model, dataloader, num_steps)
        >>> 
        >>> # Apply symbolic regression to the inputs and outputs of the MLP
        >>> regressor = wrapped_model.interpret(inputs)
        >>> 
        >>> # Switch to using the symbolic equation instead of the MLP in the forwards 
            pass of your deep learning model
        >>> trained_model.switch_to_equation()
        >>> # Switch back to using the MLP in the forwards pass
        >>> trained_model.switch_to_mlp()
    """
    
    def __init__(self, mlp: nn.Module, mlp_name: str = None):
        """
        Initialise the MLP_SR wrapper.
        
        Args:
            mlp (nn.Module): The PyTorch MLP model to wrap
            mlp_name (str, optional): Human-readable name for this MLP instance.
                                    If None, generates a unique name based on object ID.
        """
        super().__init__()
        self.InterpretSR_MLP = mlp
        self.mlp_name = mlp_name or f"mlp_{id(self)}"
        if not mlp_name: 
            print(f"➡️ No MLP name specified. MLP label is {self.mlp_name}.")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Automatically switches between MLP and symbolic equation based on current mode.
        When using symbolic equation mode, extracts only the required input variables
        and evaluates the discovered symbolic expression.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If symbolic equation requires variables not present in input
        """
        if hasattr(self, '_using_equation') and self._using_equation:
            # Extract only the variables used in the equation
            selected_inputs = []
            for idx in self._var_indices:
                if idx < x.shape[1]:
                    selected_inputs.append(x[:, idx])
                else:
                    raise ValueError(f"Equation requires variable x{idx} but input only has {x.shape[1]} dimensions")
            
            # Convert to numpy for the equation function, then back to torch
            numpy_inputs = [inp.detach().cpu().numpy() for inp in selected_inputs]
            result = self._equation_func(*numpy_inputs)
            
            # Convert back to torch tensor with same device/dtype as input
            result_tensor = torch.tensor(result, dtype=x.dtype, device=x.device)
            
            # Ensure result has correct shape (batch_size, output_dim)
            if result_tensor.dim() == 1:
                result_tensor = result_tensor.unsqueeze(1)
            
            return result_tensor
        else:
            return self.InterpretSR_MLP(x)

    def interpret(self, inputs, **kwargs):
        """
        Discover symbolic expressions that approximate the MLP's behavior.
        
        Uses PySR to find mathematical expressions that best fit the input-output relationship learned by the neural network.
        
        Args:
            inputs (torch.Tensor): Input data for symbolic regression fitting
            **kwargs: Parameters passed to PySRRegressor. Defaults:
                - binary_operators (list): ["+", "*"]
                - unary_operators (list): ["inv(x) = 1/x", "sin", "exp"]
                - niterations (int): 400
                - output_directory (str): "SR_output/{mlp_name}" # Where PySR outputs are 
                stored
                - run_id (str): "{timestamp}" # Where PySR outputs of a specific run 
                are stored
            To see more information on the possible inputs to the PySRRegressor, please see
            the PySR documentation.
                
        Returns:
            PySRRegressor: Fitted symbolic regression model
            
        Example:
            >>> regressor = model.interpret(train_inputs, niterations=1000)
            >>> print(regressor.get_best()['equation'])
        """
        timestamp = int(time.time())
        run_id = f"{timestamp}"
        output_name = f"SR_output/{self.mlp_name}"
        
        default_params = {
            "binary_operators": ["+", "*"],
            "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
            "extra_sympy_mappings": {"inv": lambda x: 1/x},
            "niterations": 400,
            "complexity_of_operators": {"sin": 3, "exp":3},
            "output_directory": output_name,
            "run_id": run_id
        }
        params = {**default_params, **kwargs}
        regressor = PySRRegressor(**params)
        self.InterpretSR_MLP.eval()
        with torch.no_grad():
            output = self.InterpretSR_MLP(inputs)
        regressor.fit(inputs.detach(), output.detach())

        print(f"❤️ SR on {self.mlp_name} complete.")
        print(f"💡Best equation found to be {regressor.get_best()['equation']}.")

        self.pysr_regressor = regressor
        return regressor
   
    def _get_equation(self, complexity: int = None):
        """
        Extract symbolic equation function from fitted regressor.
        
        Converts the symbolic expression from PySR into a callable function
        that can be used for prediction.
        
        Args:
            complexity (int, optional): Specific complexity level to retrieve.
                                      If None, returns the best overall equation.
                                      
        Returns:
            tuple or None: (equation_function, sorted_variables) if successful,
                          None if no equation found or complexity not available
                          
        Note:
            This is an internal method. Use switch_to_equation() for public API.
        """
        if not hasattr(self, 'pysr_regressor') or self.pysr_regressor is None:
            print("❗No equation found for this MLP yet. You need to first run .interpret to find the best equation to fit this MLP.")
            return None

        if complexity is None:
            best_str = self.pysr_regressor.get_best()["equation"] 
            expr = self.pysr_regressor.equations_.loc[self.pysr_regressor.equations_["equation"] == best_str, "sympy_format"].values[0]
        else:
            matching_rows = self.pysr_regressor.equations_[self.pysr_regressor.equations_["complexity"] == complexity]
            if matching_rows.empty:
                available_complexities = sorted(self.pysr_regressor.equations_["complexity"].unique())
                print(f"⚠️ Warning: No equation found with complexity {complexity}. Available complexities: {available_complexities}")
                return None
            expr = matching_rows["sympy_format"].values[0]

        vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))
        f = lambdify(vars_sorted, expr, "numpy")
        return f, vars_sorted

    def switch_to_equation(self, complexity: int = None):
        """
        Switch the forward pass from MLP to symbolic equation.
        
        After calling this method, the model will use the discovered symbolic
        expression instead of the neural network for forward passes. This maintains
        gradient flow for continued training of other model components.
        
        Args:
            complexity (int, optional): Specific complexity level to use.
                                      If None, uses the best overall equation.
            
        Example:
            >>> model.switch_to_equation(complexity=5)

        """
        result = self._get_equation(complexity)
        if result is None:
            return
            
        f, vars_sorted = result
        
        # Store original MLP for potential restoration
        if not hasattr(self, '_original_mlp'):
            self._original_mlp = self.InterpretSR_MLP
        
        # Convert variable names to indices (e.g., 'x0' -> 0, 'x4' -> 4)
        var_indices = []
        for var in vars_sorted:
            var_str = str(var)
            if var_str.startswith('x'):
                try:
                    idx = int(var_str[1:])
                    var_indices.append(idx)
                except ValueError:
                    print(f"⚠️ Warning: Could not parse variable {var_str}")
                    return
            else:
                print(f"⚠️ Warning: Unexpected variable format {var_str}")
                return
        
        self._var_indices = var_indices
        self._equation_func = f
        self._using_equation = True
        
        # Get the equation string for display
        if complexity is None:
            equation_str = self.pysr_regressor.get_best()["equation"]
        else:
            matching_rows = self.pysr_regressor.equations_[self.pysr_regressor.equations_["complexity"] == complexity]
            equation_str = matching_rows["equation"].values[0]
        
        print(f"✅ Successfully switched {self.mlp_name} to symbolic equation: {equation_str}")
        print(f"📊 Using variables: {[f'x{i}' for i in var_indices]}.")
   
    def switch_to_mlp(self):
        """
        Switch back to using the original MLP for forward passes.
        
        Restores the neural network as the primary forward pass mechanism,
        reverting any previous switch_to_equation() call.
        
        Returns:
            bool: True if switch was successful, False if no original MLP stored
            
        Example:
            >>> model.switch_to_equation()  # Use symbolic equation
            >>> # ... do some analysis ...
            >>> model.switch_to_mlp()       # Switch back to neural network
        """
        if hasattr(self, '_original_mlp'):
            self.InterpretSR_MLP = self._original_mlp
            self._using_equation = False
            print(f"✅ Switched {self.mlp_name} back to MLP")
            return True
        else:
            print("❗ No original MLP stored to switch back to")
            return False