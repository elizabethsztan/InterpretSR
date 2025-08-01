import sys
import os
import shutil
import pytest
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add the src directory to Python path for absolute imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from symtorch.mlp_sr import MLP_SR


def test_MLP_SR_wrapper():
    """
    Test that MLP_SR wrapper can successfully wrap a PyTorch Sequential model.
    """
    try:
        class SimpleModel(nn.Module):
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
                self.mlp = MLP_SR(mlp, mlp_name = "Sequential")
        model = SimpleModel(input_dim=5, output_dim=1)
        assert hasattr(model.mlp, 'InterpretSR_MLP'), "MLP_SR should have InterpretSR_MLP attribute"
        assert hasattr(model.mlp, 'interpret'), "MLP_SR should have interpret method"
    except Exception as e:
        pytest.fail(f"MLP_SR wrapper failed with error: {e}.")


class SimpleModel(nn.Module):
    """
    Simple model class for testing MLP_SR functionality.
    Uses a Sequential MLP wrapped with MLP_SR.
    """
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
        self.mlp = MLP_SR(mlp, mlp_name = "Sequential")

    def forward(self, x):
        x = self.mlp(x)
        return x


def train_model(model, dataloader, opt, criterion, epochs = 100):
    """
    Train a model for the specified number of epochs.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader for training data
        opt: Optimizer
        criterion: Loss function
        epochs: Number of training epochs
        
    Returns:
        tuple: (trained_model, loss_tracker)
    """
    loss_tracker = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
        
        loss_tracker.append(epoch_loss)
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}')
    return model, loss_tracker


# Global test data and model setup
np.random.seed(290402)  # For reproducible tests
torch.manual_seed(290402)

# Make the dataset 
x = np.array([np.random.uniform(0, 1, 1_000) for _ in range(5)]).T  
y = x[:, 0]**2 + 3*np.sin(x[:, 4]) - 4
noise = np.array([np.random.normal(0, 0.05*np.std(y)) for _ in range(len(y))])
y = y + noise 

# Split into train and test
X_train, _, y_train, _ = train_test_split(x, y.reshape(-1,1), test_size=0.2, random_state=290402)

# Create the model and set up training
model = SimpleModel(input_dim=x.shape[1], output_dim=1)
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)
dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Global variable to store trained model for subsequent tests
trained_model = None


def test_training_MLP_SR_model():
    """
    Test that a MLP_SR wrapped model can be trained successfully.
    """
    global trained_model
    try:
        trained_model, losses = train_model(model, dataloader, opt, criterion, 20)
        assert len(losses) == 20, "Should have loss for each epoch"
        assert all(isinstance(loss, float) for loss in losses), "All losses should be floats"
        
    except Exception as e:
        pytest.fail(f"MLP_SR model training failed with error {e}.")


def test_MLP_SR_interpret():
    """
    Test that the interpret method works on a trained MLP_SR model.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
        
    try:
        # Create input data for interpretation
        input_data = torch.FloatTensor(X_train[:100])  # Use subset for faster testing
        
        # Run interpretation with reduced iterations for testing
        regressors = trained_model.mlp.interpret(input_data, niterations=50)
        
        # For single output model, should return dictionary with one entry
        assert regressors is not None, "Regressors should not be None"
        assert isinstance(regressors, dict), "Should return dictionary of regressors"
        assert 0 in regressors, "Should have regressor for dimension 0"
        
        # Test the regressor for dimension 0
        regressor = regressors[0]
        assert hasattr(regressor, 'equations_'), "Regressor should have equations_ attribute"
        assert hasattr(regressor, 'get_best'), "Regressor should have get_best method"
        
        # Verify the MLP_SR object stored the regressors
        assert hasattr(trained_model.mlp, 'pysr_regressor'), "MLP_SR should store the regressors"
        assert 0 in trained_model.mlp.pysr_regressor, "MLP_SR should store regressor for dimension 0"
        assert trained_model.mlp.pysr_regressor[0] is regressor, "Stored regressor should match returned regressor"
        
    except Exception as e:
        pytest.fail(f"MLP_SR interpret method failed with error: {e}")
    finally:
        # Clean up SR output directory
        cleanup_sr_outputs()


def test_switch_to_equation():
    """
    Test that switch_to_equation method works correctly.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
        
    # Ensure we have a regressor first
    if not hasattr(trained_model.mlp, 'pysr_regressor') or not trained_model.mlp.pysr_regressor:
        input_data = torch.FloatTensor(X_train[:100])
        trained_model.mlp.interpret(input_data, niterations=50)
    
    try:
        # Test switching to equation
        trained_model.mlp.switch_to_equation()
        assert trained_model.mlp._using_equation, "Should be using equation mode after switch"
        
        # Verify internal state for multi-dimensional API
        assert hasattr(trained_model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert trained_model.mlp._using_equation, "Should be using equation mode"
        assert hasattr(trained_model.mlp, '_equation_funcs'), "Should have _equation_funcs attribute"
        assert hasattr(trained_model.mlp, '_equation_vars'), "Should have _equation_vars attribute"
        
        # For single output model, should have one equation function
        assert len(trained_model.mlp._equation_funcs) == 1, "Should have one equation function"
        assert 0 in trained_model.mlp._equation_funcs, "Should have equation function for dimension 0"
        assert 0 in trained_model.mlp._equation_vars, "Should have equation variables for dimension 0"
        
        # Test forward pass still works
        test_input = torch.FloatTensor(X_train[:10])
        output = trained_model.mlp(test_input)
        assert output is not None, "Forward pass should work in equation mode"
        assert output.shape[0] == 10, "Output should have correct batch size"
        assert output.shape[1] == 1, "Output should have correct number of dimensions"
        
    except Exception as e:
        pytest.fail(f"switch_to_equation failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_switch_to_mlp():
    """
    Test that switch_to_mlp method works correctly.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
        
    # Ensure we have a regressor first
    if not hasattr(trained_model.mlp, 'pysr_regressor') or trained_model.mlp.pysr_regressor is None:
        input_data = torch.FloatTensor(X_train[:100])
        trained_model.mlp.interpret(input_data, niterations=50)
    
    # Switch to equation mode first
    trained_model.mlp.switch_to_equation()
    
    try:
        # Test switching back to MLP
        success = trained_model.mlp.switch_to_mlp()
        assert success, "switch_to_mlp should return True on success"
        
        # Verify internal state
        assert hasattr(trained_model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert not trained_model.mlp._using_equation, "Should not be using equation mode"
        
        # Test forward pass still works
        test_input = torch.FloatTensor(X_train[:10])
        output = trained_model.mlp(test_input)
        assert output is not None, "Forward pass should work in MLP mode"
        assert output.shape[0] == 10, "Output should have correct batch size"
        
    except Exception as e:
        pytest.fail(f"switch_to_mlp failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_equation_actually_used_in_forward():
    """
    Test that switching to equation mode actually uses the symbolic equation 
    by manually setting a known equation and verifying the output.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input with enough dimensions - use simple equation sin(x0) + 2
        test_input = torch.FloatTensor([[0.5, 0.1, 0.2, 0.3, 0.4], 
                                       [1.0, 0.2, 0.3, 0.4, 0.5], 
                                       [1.57, 0.3, 0.4, 0.5, 0.6], 
                                       [3.14, 0.4, 0.5, 0.6, 0.7]])  # 5 dimensions
        
        # Manually set up the equation components for multi-dimensional API
        def test_equation(x0):
            return np.sin(x0) + 2
        
        # Manually set the equation in the MLP_SR object using new multi-dimensional API
        trained_model.mlp._equation_funcs = {0: test_equation}  # Dictionary for dimension 0
        trained_model.mlp._equation_vars = {0: [0]}  # Only use first input variable for dimension 0
        trained_model.mlp._using_equation = True
        
        # Get output using the equation
        equation_output = trained_model.mlp(test_input)
        
        # Calculate expected output manually
        expected_output = torch.tensor([[np.sin(0.5) + 2], 
                                       [np.sin(1.0) + 2], 
                                       [np.sin(1.57) + 2], 
                                       [np.sin(3.14) + 2]], dtype=torch.float32)
        
        # Verify outputs match (within floating point tolerance)
        diff = torch.abs(equation_output - expected_output)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-5, f"Equation output doesn't match expected (max diff: {max_diff})"
        print(f"✅ Equation mode correctly computes sin(x0) + 2 (max diff: {max_diff:.8f})")
        
    except Exception as e:
        pytest.fail(f"test_equation_actually_used_in_forward failed with error: {e}")
    finally:
        # Reset to MLP mode
        if hasattr(trained_model.mlp, '_using_equation'):
            trained_model.mlp._using_equation = False


def test_mlp_actually_used_after_switch_back():
    """
    Test that switching back to MLP mode actually uses the original MLP
    by comparing with a separate MLP loaded with the same weights.
    """
    global trained_model
    if trained_model is None:
        pytest.skip("No trained model available - training test may have failed")
    
    try:
        # Create test input
        test_input = torch.FloatTensor(X_train[:10])
        
        # Ensure we're in MLP mode
        trained_model.mlp.switch_to_mlp()
        trained_model.mlp._using_equation = False
        
        # Get output from the MLP_SR in MLP mode
        mlp_sr_output = trained_model.mlp(test_input).clone().detach()
        
        # Create a separate regular MLP with same architecture
        separate_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Copy weights from the MLP_SR's internal MLP to the separate MLP
        separate_mlp.load_state_dict(trained_model.mlp.InterpretSR_MLP.state_dict())
        
        # Set to eval mode to match the trained model
        separate_mlp.eval()
        
        # Get output from the separate MLP
        with torch.no_grad():
            separate_mlp_output = separate_mlp(test_input)
        
        # Outputs should be identical
        diff = torch.abs(mlp_sr_output - separate_mlp_output)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-6, f"MLP_SR and separate MLP outputs differ (max diff: {max_diff})"
        print(f"✅ MLP mode uses actual MLP (max diff: {max_diff:.8f})")
        
    except Exception as e:
        pytest.fail(f"test_mlp_actually_used_after_switch_back failed with error: {e}")


class DualMLPModel(nn.Module):
    """
    Model with two MLPs: one regular and one wrapped with MLP_SR.
    Used to test training after switching to symbolic equations.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DualMLPModel, self).__init__()
        
        # Regular MLP (not wrapped)
        self.regular_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # MLP wrapped with MLP_SR
        sr_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sr_mlp = MLP_SR(sr_mlp, mlp_name="SRSequential")
        
    def forward(self, x):
        # Combine outputs from both MLPs
        regular_out = self.regular_mlp(x)
        sr_out = self.sr_mlp(x)
        return regular_out + sr_out


def test_training_after_switch_to_equation():
    """
    Test that a model can still train after switching one MLP component to symbolic equation.
    """
    try:
        # Create dual MLP model
        dual_model = DualMLPModel(input_dim=5, output_dim=1)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dual_model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train initially for a few epochs
        print("Training dual model initially...")
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=10)
        
        # Run interpret on the SR-wrapped MLP component
        print("Running interpretation on SR-wrapped MLP...")
        input_data = torch.FloatTensor(X_train[:100])
        regressor = dual_model.sr_mlp.interpret(input_data, niterations=30)
        
        assert regressor is not None, "Interpretation should succeed"
        assert hasattr(dual_model.sr_mlp, 'pysr_regressor'), "Should have regressor stored"
        
        # Switch to equation mode
        print("Switching SR-wrapped MLP to equation mode...")
        dual_model.sr_mlp.switch_to_equation()
        assert dual_model.sr_mlp._using_equation, "Should be in equation mode"
        
        # Continue training after switch - this is the key test
        print("Training dual model after equation switch...")
        dual_model, post_switch_losses = train_model(dual_model, dataloader, optimizer, criterion, epochs=5)
        
        # Verify training completed successfully
        assert len(post_switch_losses) == 5, "Should complete all post-switch epochs"
        assert all(isinstance(loss, float) for loss in post_switch_losses), "All losses should be valid floats"
        
        # Test that forward passes still work
        test_input = torch.FloatTensor(X_train[:10])
        output = dual_model(test_input)
        assert output is not None, "Forward pass should work after equation switch"
        assert output.shape == (10, 1), "Output should have correct shape"
        
        print("✅ Successfully trained model after switching to symbolic equation")
        
    except Exception as e:
        pytest.fail(f"Training after switch to equation failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_equation_parameters_fixed_during_training():
    """
    Test that symbolic equation parameters remain fixed during training.
    The equation itself should not change, only other model components should train.
    """
    try:
        # Create dual MLP model
        dual_model = DualMLPModel(input_dim=5, output_dim=1)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dual_model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train initially
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=5)
        
        # Run interpret and switch to equation
        input_data = torch.FloatTensor(X_train[:100])
        _ = dual_model.sr_mlp.interpret(input_data, niterations=30)
        dual_model.sr_mlp.switch_to_equation()
        
        # Get equation functions and variables before training (multi-dimensional API)
        equation_funcs_before = dual_model.sr_mlp._equation_funcs.copy()
        equation_vars_before = dual_model.sr_mlp._equation_vars.copy()
        
        # Test the equation output before training
        test_input = torch.FloatTensor([[0.5, 0.3, 0.7, 0.1, 0.9]])
        with torch.no_grad():
            equation_output_before = dual_model.sr_mlp(test_input).clone()
        
        # Train more after switching to equation
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=3)
        
        # Check that equation functions and variables haven't changed
        equation_funcs_after = dual_model.sr_mlp._equation_funcs
        equation_vars_after = dual_model.sr_mlp._equation_vars
        
        # For single output model, check dimension 0
        assert equation_funcs_before[0] is equation_funcs_after[0], "Equation function should be the same object"
        assert equation_vars_before[0] == equation_vars_after[0], "Variable indices should remain unchanged"
        
        # Test that equation gives same output for same input
        with torch.no_grad():
            equation_output_after = dual_model.sr_mlp(test_input)
        
        diff = torch.abs(equation_output_before - equation_output_after)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-6, f"Equation output should be identical (diff: {max_diff})"
        
        print("✅ Confirmed: Symbolic equation parameters remain fixed during training")
        print(f"   Equation function: {equation_funcs_before[0]}")
        print(f"   Variables used: {[f'x{i}' for i in equation_vars_before[0]]}")
        print(f"   Output consistency: max diff = {max_diff:.8f}")
        
    except Exception as e:
        pytest.fail(f"Equation parameter fixity test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_gradient_flow_through_other_components():
    """
    Test that gradients still flow through other model components when one uses symbolic equation.
    The regular MLP should continue to train while the equation component remains fixed.
    """
    try:
        # Create dual MLP model
        dual_model = DualMLPModel(input_dim=5, output_dim=1)
        
        # Set up training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(dual_model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train initially
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=5)
        
        # Run interpret and switch to equation
        input_data = torch.FloatTensor(X_train[:100])
        _ = dual_model.sr_mlp.interpret(input_data, niterations=30)
        dual_model.sr_mlp.switch_to_equation()
        
        # Get regular MLP parameters before additional training
        regular_mlp_params_before = {}
        for name, param in dual_model.regular_mlp.named_parameters():
            regular_mlp_params_before[name] = param.clone().detach()
        
        # Train more - regular MLP should change, equation should not
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=3)
        
        # Check that regular MLP parameters have changed (indicating gradient flow)
        regular_mlp_changed = False
        for name, param in dual_model.regular_mlp.named_parameters():
            param_before = regular_mlp_params_before[name]
            diff = torch.abs(param - param_before)
            max_diff = torch.max(diff)
            if max_diff > 1e-6:
                regular_mlp_changed = True
                print(f"   Regular MLP {name}: max parameter change = {max_diff:.6f}")
                break
        
        assert regular_mlp_changed, "Regular MLP parameters should change during training"
        
        # Verify equation component does NOT maintain gradients
        # (The symbolic equation is not differentiable in PyTorch's autograd sense)
        test_input = torch.FloatTensor(X_train[:10])
        test_input.requires_grad_(True)
        
        # Forward pass through equation component only
        equation_output = dual_model.sr_mlp(test_input)
        
        # The equation output should not have gradients
        assert not equation_output.requires_grad, "Equation output should not require gradients"
        assert equation_output.grad_fn is None, "Equation output should not have grad_fn"
        
        # Try to backward through the equation - this should fail
        try:
            loss = torch.sum(equation_output)
            loss.backward()
            gradient_flows = True
        except RuntimeError:
            gradient_flows = False
        
        assert not gradient_flows, "Gradients should NOT flow through symbolic equation"
        
        print("✅ Confirmed: Gradients flow correctly in mixed MLP/equation model")
        print("   - Regular MLP parameters change during training")
        print("   - Equation parameters remain fixed")
        print("   - Gradients do NOT flow through symbolic equation (as expected)")
        
    except Exception as e:
        pytest.fail(f"Gradient flow test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


class MultiOutputModel(nn.Module):
    """
    Model with multiple outputs for testing multi-dimensional symbolic regression.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(MultiOutputModel, self).__init__()
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.mlp = MLP_SR(mlp, mlp_name="MultiOutput")

    def forward(self, x):
        return self.mlp(x)


def create_multi_output_synthetic_data(n_samples=500, input_dim=4, output_dim=3):
    """Create synthetic data with multiple outputs and known relationships."""
    np.random.seed(42)
    
    x = np.random.uniform(-1, 1, (n_samples, input_dim))
    y = np.zeros((n_samples, output_dim))
    
    # Simple known relationships for easier symbolic regression
    for i in range(output_dim):
        if i == 0:
            y[:, i] = x[:, 0] + x[:, min(1, input_dim-1)]  # Linear sum
        elif i == 1:
            y[:, i] = x[:, 0] * x[:, min(2, input_dim-1)]  # Product 
        elif i == 2:
            y[:, i] = x[:, min(1, input_dim-1)]**2 + 0.5   # Quadratic plus constant
        else:
            # For additional dimensions, create simple linear combinations
            y[:, i] = np.sum(x[:, :min(i+1, input_dim)], axis=1) * 0.1
    
    # Add small amount of noise
    noise = np.random.normal(0, 0.01, y.shape)
    y = y + noise
    
    return x, y


def test_multi_dimensional_interpret_all_outputs():
    """
    Test that interpret() works correctly when applied to all output dimensions.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=300, input_dim=4, output_dim=3)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(30):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test interpret on all dimensions
        input_data = X_tensor[:150]  # Use subset for faster testing
        regressors = model.mlp.interpret(input_data, niterations=30)
        
        # Verify we got a dictionary of regressors
        assert isinstance(regressors, dict), "Should return dictionary of regressors for multi-dim"
        assert len(regressors) == 3, "Should have regressors for all 3 output dimensions"
        assert set(regressors.keys()) == {0, 1, 2}, "Should have regressors for dimensions 0, 1, 2"
        
        # Verify each regressor has the expected attributes
        for dim, regressor in regressors.items():
            assert regressor is not None, f"Regressor for dimension {dim} should not be None"
            assert hasattr(regressor, 'equations_'), f"Regressor {dim} should have equations_ attribute"
            assert hasattr(regressor, 'get_best'), f"Regressor {dim} should have get_best method"
            
            # Test that we can get the best equation
            best_eq = regressor.get_best()['equation']
            assert isinstance(best_eq, str), f"Best equation for dimension {dim} should be a string"
            assert len(best_eq) > 0, f"Best equation for dimension {dim} should not be empty"
        
        # Verify the MLP_SR object stored all regressors
        assert hasattr(model.mlp, 'pysr_regressor'), "MLP_SR should store regressors"
        for dim in [0, 1, 2]:
            assert dim in model.mlp.pysr_regressor, f"MLP_SR should store regressor for dimension {dim}"
        
        print("✅ Multi-dimensional interpret (all outputs) test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional interpret (all outputs) failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_interpret_specific_output():
    """
    Test that interpret() works correctly when applied to a specific output dimension.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=300, input_dim=4, output_dim=3)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(30):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test interpret on specific dimension (dimension 1)
        input_data = X_tensor[:150]
        regressor = model.mlp.interpret(input_data, output_dim=1, niterations=30)
        
        # Verify we got a single regressor (not a dictionary)
        assert not isinstance(regressor, dict), "Should return single regressor for specific dimension"
        assert regressor is not None, "Regressor should not be None"
        assert hasattr(regressor, 'equations_'), "Regressor should have equations_ attribute"
        assert hasattr(regressor, 'get_best'), "Regressor should have get_best method"
        
        # Test that we can get the best equation
        best_eq = regressor.get_best()['equation']
        assert isinstance(best_eq, str), "Best equation should be a string"
        assert len(best_eq) > 0, "Best equation should not be empty"
        
        # Verify the MLP_SR object stored the regressor for dimension 1
        assert hasattr(model.mlp, 'pysr_regressor'), "MLP_SR should store regressors"
        assert 1 in model.mlp.pysr_regressor, "MLP_SR should store regressor for dimension 1"
        assert model.mlp.pysr_regressor[1] is regressor, "Stored regressor should match returned regressor"
        
        print("✅ Multi-dimensional interpret (specific output) test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional interpret (specific output) failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_forward_pass_consistency():
    """
    Test that forward passes work correctly with multi-dimensional models.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=4, output_dim=3)
        
        # Create and train model
        model = MultiOutputModel(input_dim=4, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Test forward pass before interpretation
        test_input = X_tensor[:10]
        output_before = model(test_input).clone().detach()
        assert output_before.shape == (10, 3), "Output should have correct shape (batch_size, output_dim)"
        
        # Run interpretation
        model.mlp.interpret(X_tensor[:100], niterations=20)
        
        # Test forward pass after interpretation (should still work)
        output_after = model(test_input)
        assert output_after.shape == (10, 3), "Output should maintain correct shape after interpretation"
        
        # Outputs should be very similar (model weights shouldn't change during interpretation)
        diff = torch.abs(output_before - output_after)
        max_diff = torch.max(diff)
        assert max_diff < 1e-5, f"Forward pass should be consistent before/after interpretation (max diff: {max_diff})"
        
        print("✅ Multi-dimensional forward pass consistency test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional forward pass consistency test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_mixed_training():
    """
    Test training a model that combines multi-dimensional MLP_SR with other components.
    """
    try:
        # Create a model that combines multi-output MLP_SR with another component
        class MixedMultiModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MixedMultiModel, self).__init__()
                
                # Multi-output MLP_SR component
                sr_mlp = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_dim)
                )
                self.sr_mlp = MLP_SR(sr_mlp, mlp_name="MultiMixed")
                
                # Regular linear layer
                self.linear = nn.Linear(input_dim, output_dim)
                
            def forward(self, x):
                # Combine outputs from both components
                sr_out = self.sr_mlp(x)
                linear_out = self.linear(x)
                return sr_out + linear_out * 0.1  # Weight the linear component less
        
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=3, output_dim=2)
        
        # Create and train mixed model
        model = MixedMultiModel(input_dim=3, output_dim=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Initial training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        initial_loss = loss.item()
        
        # Run interpretation on the MLP_SR component
        model.sr_mlp.interpret(X_tensor[:100], niterations=20)
        
        # Continue training after interpretation
        for epoch in range(10):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Verify training worked (model should still be trainable)
        assert isinstance(initial_loss, float), "Initial loss should be a float"
        assert isinstance(final_loss, float), "Final loss should be a float"
        assert not np.isnan(final_loss), "Final loss should not be NaN"
        assert not np.isinf(final_loss), "Final loss should not be infinite"
        
        # Test forward pass
        test_input = X_tensor[:5]
        output = model(test_input)
        assert output.shape == (5, 2), "Output should have correct shape"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        
        print("✅ Multi-dimensional mixed training test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional mixed training test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_switch_to_equation():
    """
    Test that switch_to_equation works correctly with multi-dimensional models.
    """
    try:
        # Create multi-output data
        x_data, y_data = create_multi_output_synthetic_data(n_samples=200, input_dim=3, output_dim=2)
        
        # Create and train model
        model = MultiOutputModel(input_dim=3, output_dim=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Run interpretation on all dimensions
        input_data = X_tensor[:100]
        regressors = model.mlp.interpret(input_data, niterations=20)
        
        assert isinstance(regressors, dict), "Should return dictionary for multi-dim"
        assert len(regressors) == 2, "Should have 2 regressors"
        
        # Test forward pass before switching
        test_input = X_tensor[:5]
        output_before = model(test_input).clone().detach()
        assert output_before.shape == (5, 2), "Output should have correct shape"
        
        # Test switch_to_equation
        model.mlp.switch_to_equation()
        
        # Verify we're in equation mode
        assert hasattr(model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert model.mlp._using_equation, "Should be in equation mode"
        assert hasattr(model.mlp, '_equation_funcs'), "Should have _equation_funcs attribute"
        assert hasattr(model.mlp, '_equation_vars'), "Should have _equation_vars attribute"
        assert len(model.mlp._equation_funcs) == 2, "Should have equation functions for both dimensions"
        
        # Test forward pass after switching
        output_after = model(test_input)
        assert output_after.shape == (5, 2), "Output should maintain correct shape after switch"
        assert not torch.isnan(output_after).any(), "Output should not contain NaN values"
        
        # Test switch back to MLP
        success = model.mlp.switch_to_mlp()
        assert success, "Switch back should succeed"
        assert not model.mlp._using_equation, "Should not be in equation mode after switch back"
        
        # Test forward pass after switching back
        output_back = model(test_input)
        assert output_back.shape == (5, 2), "Output should maintain correct shape after switch back"
        
        print("✅ Multi-dimensional switch_to_equation test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional switch_to_equation test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def test_multi_dimensional_switch_to_equation_missing_dims():
    """
    Test that switch_to_equation correctly handles missing dimensions.
    """
    try:
        # Create multi-output data with 3 dimensions
        x_data, y_data = create_multi_output_synthetic_data(n_samples=150, input_dim=3, output_dim=3)
        
        # Create and train model with 3 outputs
        model = MultiOutputModel(input_dim=3, output_dim=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(x_data)
        y_tensor = torch.FloatTensor(y_data)
        
        # Quick training
        for epoch in range(15):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Run interpretation on only 2 out of 3 dimensions
        input_data = X_tensor[:75]
        model.mlp.interpret(input_data, output_dim=0, niterations=15)
        model.mlp.interpret(input_data, output_dim=1, niterations=15)
        
        # Manually remove one dimension to simulate missing scenario
        if 2 in model.mlp.pysr_regressor:
            del model.mlp.pysr_regressor[2]
        
        # Try to switch_to_equation (should fail gracefully)
        model.mlp.switch_to_equation()
        
        # Should still be in MLP mode
        if hasattr(model.mlp, '_using_equation'):
            assert not model.mlp._using_equation, "Should not switch to equation mode with missing dimensions"
        
        # Forward pass should still work normally
        test_input = X_tensor[:3]
        output = model(test_input)
        assert output.shape == (3, 3), "Forward pass should work normally with missing equations"
        
        print("✅ Multi-dimensional missing dimensions test passed")
        
    except Exception as e:
        pytest.fail(f"Multi-dimensional missing dimensions test failed with error: {e}")
    finally:
        cleanup_sr_outputs()


def cleanup_sr_outputs():
    """
    Clean up SR output files and directories created during testing.
    """
    if os.path.exists('SR_output'):
        shutil.rmtree('SR_output')
    
    # Clean up any other potential output files
    for file in os.listdir('.'):
        if file.startswith('hall_of_fame') or file.endswith('.pkl'):
            try:
                os.remove(file)
            except OSError:
                pass


# Cleanup fixture to ensure files are cleaned up after all tests
@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """
    Fixture to clean up output files after all tests complete.
    """
    yield
    cleanup_sr_outputs()