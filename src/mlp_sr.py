from pysr import *
import torch 
import torch.nn as nn
import time

class MLP_SR(nn.Module):
   def __init__(self, mlp: nn.Module, layer_name: str = None):
       super().__init__()
       self.InterpretSR_MLP = mlp
       self.layer_name = layer_name or f"layer_{id(self)}"
 
   def forward(self, x):
       x = self.InterpretSR_MLP(x)
       return x
 
   def interpret(self, inputs, **kwargs):
    timestamp = int(time.time())
    run_id = f"{self.layer_name}_{timestamp}"
    output_name = f"sr_output_{self.layer_name}_{timestamp}"
    
    default_params = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "constraints": {"sin": 3, "exp": 3},
        "complexity_of_operators": {"sin": 3, "exp": 3}, 
        "niterations": 400,
        "outputs": output_name,
        "run_id": run_id
    }
    params = {**default_params, **kwargs}
    regressor = PySRRegressor(**params)
    self.InterpretSR_MLP.eval()
    with torch.no_grad():
        output = self.InterpretSR_MLP(inputs)
    regressor.fit(inputs.detach(), output.detach())
    return regressor
   