"""
InterpretSR Utilities

This module provides utility functions for loading and adapting model weights
when using MLP_SR wrappers, enabling seamless weight transfer between standard
MLPs and their wrapped counterparts.
"""

import torch

def _load_existing_weights(file, mlp_mappings=None):
    """
    Load weights from a model and adapt them for architectures with MLP_SR wrappers.
    This function is used in the main weight loading function load_existing_weights_auto.
    
    Args:
        file: Path to the saved model weights
        mlp_mappings: Dict mapping original MLP paths to their wrapped versions.
                     If None, defaults to simple case: {"mlp.": "mlp.InterpretSR_MLP."}
    """
    original_state_dict = torch.load(file)
    new_state_dict = {}
    
    # Default mapping for backward compatibility
    if mlp_mappings is None:
        mlp_mappings = {"mlp.": "mlp.InterpretSR_MLP."}
    
    for key, value in original_state_dict.items():
        new_key = key
        
        # Apply mappings in order (longer prefixes first to avoid conflicts)
        for original_prefix, wrapped_prefix in sorted(mlp_mappings.items(), key=len, reverse=True):
            if key.startswith(original_prefix):
                new_key = key.replace(original_prefix, wrapped_prefix, 1)
                break
                
        new_state_dict[new_key] = value
    
    return new_state_dict

def load_existing_weights_auto(file, target_model):
    """
    Automatically detect MLP_SR wrappers in target model and create appropriate mappings.
    
    Args:
        file: Path to the saved model weights
        target_model: The model instance with MLP_SR wrappers to load weights into
        
    Returns:
        Dict of adapted weights that can be loaded into target_model

    Example:
    >>> # Loading trained model weights onto a model using InterpretSR
    >>> from interpretsr.utils import load_existing_weights_auto
    >>> model_with_MLP_SR = model() # Initialise your model with MLP_SR wrappings
    >>> weights = load_existing_weights_auto('original_model_weights.pth') # Adapt existing weights for InterpretSR compatibility
    >>> model_with_MLP_SR.load_state_dict(weights) # Load adapted weights onto the model
    """
    # Load source weights to check their structure
    source_state_dict = torch.load(file)
    target_state_dict = target_model.state_dict()
    
    # Check if source weights are already from wrapped models
    source_has_wrapped = any(".InterpretSR_MLP." in key for key in source_state_dict.keys())
    
    if source_has_wrapped:
        # Source is already wrapped, return as-is
        return source_state_dict
    
    # Find all MLP_SR wrapped parameters in target model
    wrapped_paths = {}
    for target_key in target_state_dict.keys():
        if ".InterpretSR_MLP." in target_key:
            # Extract the path before InterpretSR_MLP
            parts = target_key.split(".InterpretSR_MLP.")
            if len(parts) == 2:
                original_prefix = parts[0] + "."
                wrapped_prefix = parts[0] + ".InterpretSR_MLP."
                wrapped_paths[original_prefix] = wrapped_prefix
    
    # Use the detected mappings
    return _load_existing_weights(file, wrapped_paths)