#Interpretability toolkit 

import torch
import torch.nn as nn
import math

class Pruning_MLP(nn.Module):
    def __init__(self, mlp: nn.Module, initial_dim: int, target_dim: int):
        """
        Initialise the pruning wrapper.
        
        Args:
            mlp (nn.Module): The PyTorch MLP model to wrap.
        """
        super().__init__()
        self.pruning_mlp = mlp
        self.initial_dim = initial_dim # Output dimensionality before pruning
        self.current_dim = initial_dim 
        self.target_dim = target_dim # Output dimensionality after pruning finishes
        self.pruning_schedule = None
        self.pruning_mask = torch.ones(self.current_dim, dtype=torch.bool)
    
    def forward(self, x):
        return self.pruning_mlp(x) * self.pruning_mask

    def set_schedule(self, total_epochs: int, decay_rate: str, end_epoch_frac: int = 0.5):
        
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
            
        target_dims = self.pruning_schedule[epoch]
        
        with torch.no_grad():
            all_outputs = []
            
            # Check if sample_data is a DataLoader (for MLPs)
            if 'DataLoader' in str(type(sample_data)):
                # DataLoader case - for regular MLPs
                for batch in sample_data:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]  # Assuming (inputs, targets) format
                    else:
                        inputs = batch  # Just inputs
                    outputs = self.pruning_mlp(inputs)
                    all_outputs.append(outputs)
            else:
                # List of Data objects case - for GNNs
                for datapoint in sample_data:
                    # Check if it's a torch_geometric Data object by type name
                    if 'torch_geometric' in str(type(datapoint)):
                        outputs = self.pruning_mlp(datapoint.x, datapoint.edge_index)
                    else:
                        # Fallback for regular tensor data
                        outputs = self.pruning_mlp(datapoint)
                    all_outputs.append(outputs)
            
            output_array = torch.cat(all_outputs, dim=0)
            output_importance = output_array.std(dim=0)
            most_important = torch.argsort(output_importance)[-target_dims:]
            
            new_mask = torch.zeros_like(self.pruning_mask)
            new_mask[most_important] = True
            self.pruning_mask = new_mask
            self.current_dim = target_dims