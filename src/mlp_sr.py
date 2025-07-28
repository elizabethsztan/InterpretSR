from pysr import *
import torch 
import torch.nn as nn

class MLP_SR(nn.Module):
   def __init__(self, mlp: nn.Module):
       super().__init__()
       self.InterpretSR_MLP = mlp
 
   def forward(self, x):
       x = self.InterpretSR_MLP(x)
       return x
 
   def interpret(self, inputs, **kwargs):
    default_params = {
        "binary_operators": ["+", "*"],
        "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
        "extra_sympy_mappings": {"inv": lambda x: 1/x},
        "constraints": {"sin": 3, "exp": 3},
        "complexity_of_operators": {"sin": 3, "exp": 3}, 
        "niterations": 400
    }
    params = {**default_params, **kwargs}
    regressor = PySRRegressor(**params)
    self.mlp.eval()
    with torch.no_grad():
        output = self.InterpretSR_MLP(inputs)
    regressor.fit(inputs.detach(), output.detach())
    return regressor
   