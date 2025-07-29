from pysr import *
import torch 
import torch.nn as nn
import time
import sympy
from sympy import lambdify

class MLP_SR(nn.Module):
   def __init__(self, mlp: nn.Module, mlp_name: str = None):
       super().__init__()
       self.InterpretSR_MLP = mlp
       self.mlp_name = mlp_name or f"mlp_{id(self)}"
       if not mlp_name: 
          print(f"➡️ No MLP name specified. MLP label is {self.mlp_name}.")
    
   def forward(self, x):
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
    timestamp = int(time.time())
    run_id = f"{timestamp}"
    output_name = f"SR_output/{self.mlp_name}"
    
    default_params = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "constraints": {"sin": 3, "exp": 3},
        "complexity_of_operators": {"sin": 3, "exp": 3}, 
        "niterations": 400,
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
   
   def get_equation_(self, complexity: int = None):
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
       Maintains gradient flow for continued training of other model parts.
       """
       result = self.get_equation_(complexity)
       if result is None:
           return False
           
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
                   return False
           else:
               print(f"⚠️ Warning: Unexpected variable format {var_str}")
               return False
       
       self._var_indices = var_indices
       self._equation_func = f
       self._using_equation = True
       
       print(f"✅ Successfully switched {self.mlp_name} to symbolic equation.")
       print(f"📊 Using variables: {[f'x{i}' for i in var_indices]}.")
       return True
   
   def switch_to_mlp(self):
       """Switch back to using the original MLP."""
       if hasattr(self, '_original_mlp'):
           self.InterpretSR_MLP = self._original_mlp
           self._using_equation = False
           print(f"✅ Switched {self.mlp_name} back to MLP")
           return True
       else:
           print("❗ No original MLP stored to switch back to")
           return False