#Interpretability toolkit 

import torch
import torch.nn as nn
import math
import time
from typing import Optional, Dict, Any
from pysr import PySRRegressor
from .mlp_sr import MLP_SR

class Pruning_MLP(MLP_SR):
    def __init__(self, mlp: nn.Module, initial_dim: int, target_dim: int, mlp_name: str = None):
        """
        Initialise the pruning wrapper that inherits MLP_SR functionality.
        
        Args:
            mlp (nn.Module): The PyTorch MLP model to wrap.
            initial_dim (int): Initial output dimensionality before pruning
            target_dim (int): Target output dimensionality after pruning
            mlp_name (str): Name for the MLP (used by MLP_SR for output directories)
        """
        # Initialize MLP_SR with the MLP
        super().__init__(mlp, mlp_name or f"pruned_mlp_{id(self)}")
        
        # Add pruning-specific attributes
        self.initial_dim = initial_dim
        self.current_dim = initial_dim 
        self.target_dim = target_dim
        self.pruning_schedule = None
        self.pruning_mask = torch.ones(self.current_dim, dtype=torch.bool)
    
    def forward(self, x):
        """Forward pass with pruning mask applied to MLP_SR output."""
        # Use parent's forward method (handles symbolic/MLP switching)
        output = super().forward(x)
        # Apply pruning mask
        return output * self.pruning_mask

    def set_schedule(self, total_epochs: int, decay_rate: str = 'cosine', end_epoch_frac: int = 0.5):
        
        prune_end_epoch = int(end_epoch_frac * total_epochs)
        prune_epochs = prune_end_epoch

        dims_to_prune = self.initial_dim - self.target_dim
        schedule_dict = {}

        #different pruning schedules
        #exponential decay
        if decay_rate == 'exp':
            decay_rate = 3.0
            max_decay = 1 - math.exp(-decay_rate)

            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                raw_decay = 1 - math.exp(-decay_rate * progress)
                decay_factor = raw_decay / max_decay

                dims_pruned = math.ceil(dims_to_prune * decay_factor)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #linear decay
        elif decay_rate == 'linear':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                dims_pruned = math.ceil(dims_to_prune * progress)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #cosine decay
        elif decay_rate == 'cosine':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                dims_pruned = math.ceil(dims_to_prune * (1 - cosine_decay))
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #keep target_dim for the last part of training
        for epoch in range(prune_end_epoch, total_epochs):
            schedule_dict[epoch] = self.target_dim

        self.pruning_schedule = schedule_dict

    def prune(self, epoch, sample_data):

        if epoch not in self.pruning_schedule:
            return
        if self.pruning_schedule is None:
            assert 'Pruning schedule is not set.'
            
        target_dims = self.pruning_schedule[epoch]
        
        with torch.no_grad():

            output_array = self.InterpretSR_MLP(sample_data)

            output_importance = output_array.std(dim=0)
            most_important = torch.argsort(output_importance)[-target_dims:]
            
            new_mask = torch.zeros_like(self.pruning_mask)
            new_mask[most_important] = True
            self.pruning_mask = new_mask
            self.current_dim = target_dims

    def get_active_dimensions(self):
        """Get indices of currently active (non-masked) dimensions."""
        return torch.where(self.pruning_mask)[0].tolist()

    def interpret(self, sample_data, parent_model=None, **pysr_kwargs):
        """
        Override MLP_SR's interpret to only run on active dimensions.
        
        Args:
            sample_data: Input data for symbolic regression
            parent_model (nn.Module, optional): The parent model containing this MLP_SR instance.
                                              If provided, will trace intermediate activations to get
                                              the actual inputs/outputs at this layer level.
            **pysr_kwargs: Additional arguments for PySR
            
        Returns:
            Dictionary mapping active dimension indices to PySR regressors
        """
        active_dims = self.get_active_dimensions()
        if not active_dims:
            print("No active dimensions to interpret!")
            return {}
        
        # Extract inputs and outputs at this layer level
        if parent_model is not None:
            # Use forward hooks to capture inputs/outputs at this specific layer
            layer_inputs = []
            layer_outputs = []
            
            def hook_fn(module, input, output):
                if module is self.InterpretSR_MLP:
                    layer_inputs.append(input[0].clone())
                    layer_outputs.append(output.clone())
            
            # Register forward hook
            hook = self.InterpretSR_MLP.register_forward_hook(hook_fn)
            
            # Run parent model to capture intermediate activations
            parent_model.eval()
            with torch.no_grad():
                _ = parent_model(sample_data)
            
            # Remove hook
            hook.remove()
            
            # Use captured intermediate data
            if layer_inputs and layer_outputs:
                inputs = layer_inputs[0]
                full_output = layer_outputs[0]
                active_output = full_output[:, self.pruning_mask]
            else:
                raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this MLP_SR instance.")
        else:
            # Original behavior - extract inputs and outputs for active dimensions only
            self.InterpretSR_MLP.eval()
            with torch.no_grad():
                if isinstance(sample_data, torch.Tensor):
                    inputs = sample_data.detach()
                    full_output = self.InterpretSR_MLP(sample_data)
                    active_output = full_output[:, self.pruning_mask]
                else:
                    # Handle DataLoader case
                    all_inputs, all_active_outputs = [], []
                    for batch in sample_data:
                        if isinstance(batch, (list, tuple)):
                            batch_inputs = batch[0]
                        else:
                            batch_inputs = batch
                        full_output = self.InterpretSR_MLP(batch_inputs)
                        active_output = full_output[:, self.pruning_mask]
                        all_inputs.append(batch_inputs)
                        all_active_outputs.append(active_output)
                    inputs = torch.cat(all_inputs)
                    active_output = torch.cat(all_active_outputs)

        timestamp = int(time.time())
        
        # Use same default parameters as MLP_SR
        default_params = {
            "binary_operators": ["+", "*"],
            "unary_operators": ["inv(x) = 1/x", "sin", "exp"],
            "extra_sympy_mappings": {"inv": lambda x: 1/x},
            "niterations": 400,
            "complexity_of_operators": {"sin": 3, "exp": 3},
            "output_directory": f"SR_output/{self.mlp_name}",
        }
        default_params.update(pysr_kwargs)

        # Run SR for each active dimension
        regressors = {}
        for i, dim_idx in enumerate(active_dims):
            print(f"🛠️ Running SR on active dimension {dim_idx} ({i+1}/{len(active_dims)})")
            
            run_id = f"dim{dim_idx}_{timestamp}"
            params = {**default_params, "run_id": run_id}
            
            regressor = PySRRegressor(**params)
            regressor.fit(inputs.detach().numpy(), active_output[:, i].detach().numpy())
            regressors[dim_idx] = regressor
            
            best_eq = regressor.get_best()['equation']
            print(f"💡Best equation for active dimension {dim_idx}: {best_eq}")
            
        # Store in the format expected by MLP_SR (replace entire dict, don't merge)
        self.pysr_regressor = regressors
        # Set output_dims for compatibility
        self.output_dims = self.initial_dim
        
        print(f"❤️ SR on {self.mlp_name} active dimensions complete.")
        return regressors

    def switch_to_equation(self, complexity: list = None):
        """
        Override MLP_SR's switch_to_equation to handle pruned dimensions correctly.
        Only active dimensions get symbolic equations, inactive ones remain zero.
        """
        if not hasattr(self, 'pysr_regressor') or not self.pysr_regressor:
            print("❗No equations found. You need to first run .interpret.")
            return
        
        active_dims = self.get_active_dimensions()
        if not active_dims:
            print("❗No active dimensions to switch to equations.")
            return
        
        # Store original MLP for potential restoration
        if not hasattr(self, '_original_mlp'):
            self._original_mlp = self.InterpretSR_MLP
        
        # Get equations for active dimensions only
        equation_funcs = {}
        equation_vars = {}
        equation_strs = {}
        
        for i, dim_idx in enumerate(active_dims):
            # Get complexity for this specific dimension
            dim_complexity = None
            if complexity is not None:
                if isinstance(complexity, list):
                    if i < len(complexity):
                        dim_complexity = complexity[i]
                else:
                    dim_complexity = complexity
            
            result = self._get_equation(dim_idx, dim_complexity)
            if result is None:
                print(f"⚠️ Failed to get equation for dimension {dim_idx}")
                return
                
            f, vars_sorted = result
            
            # Convert variable names to indices
            var_indices = []
            for var in vars_sorted:
                var_str = str(var)
                if var_str.startswith('x'):
                    try:
                        idx = int(var_str[1:])
                        var_indices.append(idx)
                    except ValueError:
                        print(f"⚠️ Warning: Could not parse variable {var_str} for dimension {dim_idx}")
                        return
            
            equation_funcs[dim_idx] = f
            equation_vars[dim_idx] = var_indices
            
            # Get equation string for display
            regressor = self.pysr_regressor[dim_idx]
            if dim_complexity is None:
                equation_strs[dim_idx] = regressor.get_best()["equation"]
            else:
                matching_rows = regressor.equations_[regressor.equations_["complexity"] == dim_complexity]
                equation_strs[dim_idx] = matching_rows["equation"].values[0]
        
        # Store the equation information
        self._equation_funcs = equation_funcs
        self._equation_vars = equation_vars
        self._using_equation = True
        
        # Print success messages
        print(f"✅ Successfully switched {self.mlp_name} to symbolic equations for {len(active_dims)} active dimensions:")
        for dim_idx in active_dims:
            print(f"   Dimension {dim_idx}: {equation_strs[dim_idx]}")
            print(f"   Variables: {[f'x{i}' for i in equation_vars[dim_idx]]}")
        
        print(f"🎯 Active dimensions {active_dims} now using symbolic equations.")
        print(f"🔒 Inactive dimensions will output zeros.")

    def forward(self, x):
        """
        Forward pass with pruning mask applied. 
        When using equations, inactive dimensions are zero, active ones use equations.
        """
        if not hasattr(self, '_using_equation') or not self._using_equation:
            # Use parent's forward method and apply pruning mask
            output = super().forward(x)
            return output * self.pruning_mask
        else:
            # Custom forward pass for equations with proper zero-padding
            batch_size = x.shape[0]
            # Initialize output tensor with zeros for all dimensions
            output = torch.zeros(batch_size, self.initial_dim, dtype=x.dtype, device=x.device)
            
            # Fill in active dimensions with symbolic equations
            active_dims = self.get_active_dimensions()
            for dim_idx in active_dims:
                if dim_idx in self._equation_funcs:
                    equation_func = self._equation_funcs[dim_idx]
                    var_indices = self._equation_vars[dim_idx]
                    
                    # Extract variables needed for this dimension
                    selected_inputs = []
                    for idx in var_indices:
                        if idx < x.shape[1]:
                            selected_inputs.append(x[:, idx])
                        else:
                            print(f"⚠️ Variable x{idx} not available for dimension {dim_idx}")
                            continue
                    
                    if len(selected_inputs) == len(var_indices):
                        # Convert to numpy for the equation function
                        numpy_inputs = [inp.detach().cpu().numpy() for inp in selected_inputs]
                        
                        try:
                            # Evaluate the equation for this dimension
                            result = equation_func(*numpy_inputs)
                            
                            # Convert back to torch tensor with same device/dtype as input
                            result_tensor = torch.tensor(result, dtype=x.dtype, device=x.device)
                            
                            # Ensure result is 1D (batch_size,)
                            if result_tensor.dim() == 0:
                                result_tensor = result_tensor.expand(batch_size)
                            elif result_tensor.dim() > 1:
                                result_tensor = result_tensor.flatten()
                            
                            output[:, dim_idx] = result_tensor
                        except Exception as e:
                            print(f"⚠️ Error evaluating equation for dimension {dim_idx}: {e}")
            
            # Apply pruning mask (though active dimensions should already be correct)
            return output * self.pruning_mask