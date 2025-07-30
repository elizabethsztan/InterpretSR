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

from interpretsr.mlp_sr import MLP_SR


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
        regressor = trained_model.mlp.interpret(input_data, niterations=50)
        
        # Verify regressor was created and has expected attributes
        assert regressor is not None, "Regressor should not be None"
        assert hasattr(regressor, 'equations_'), "Regressor should have equations_ attribute"
        assert hasattr(regressor, 'get_best'), "Regressor should have get_best method"
        
        # Verify the MLP_SR object stored the regressor
        assert hasattr(trained_model.mlp, 'pysr_regressor'), "MLP_SR should store the regressor"
        assert trained_model.mlp.pysr_regressor is regressor, "Stored regressor should match returned regressor"
        
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
    if not hasattr(trained_model.mlp, 'pysr_regressor') or trained_model.mlp.pysr_regressor is None:
        input_data = torch.FloatTensor(X_train[:100])
        trained_model.mlp.interpret(input_data, niterations=50)
    
    try:
        # Test switching to equation
        trained_model.mlp.switch_to_equation()
        assert trained_model.mlp._using_equation, "Should be using equation mode after switch"
        
        # Verify internal state
        assert hasattr(trained_model.mlp, '_using_equation'), "Should have _using_equation attribute"
        assert trained_model.mlp._using_equation, "Should be using equation mode"
        assert hasattr(trained_model.mlp, '_equation_func'), "Should have _equation_func attribute"
        assert hasattr(trained_model.mlp, '_var_indices'), "Should have _var_indices attribute"
        
        # Test forward pass still works
        test_input = torch.FloatTensor(X_train[:10])
        output = trained_model.mlp(test_input)
        assert output is not None, "Forward pass should work in equation mode"
        assert output.shape[0] == 10, "Output should have correct batch size"
        
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
        # Create test input - use first column for simple equation sin(x0) + 2
        test_input = torch.FloatTensor([[0.5], [1.0], [1.57], [3.14]])  # Some test values
        
        # Manually set up the equation components
        def test_equation(x0):
            return np.sin(x0) + 2
        
        # Manually set the equation in the MLP_SR object
        trained_model.mlp._equation_func = test_equation
        trained_model.mlp._var_indices = [0]  # Only use first input variable
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
        
        # Get equation function and test inputs before training
        equation_func_before = dual_model.sr_mlp._equation_func
        var_indices_before = dual_model.sr_mlp._var_indices.copy()
        
        # Test the equation output before training
        test_input = torch.FloatTensor([[0.5, 0.3, 0.7, 0.1, 0.9]])
        with torch.no_grad():
            equation_output_before = dual_model.sr_mlp(test_input).clone()
        
        # Train more after switching to equation
        dual_model, _ = train_model(dual_model, dataloader, optimizer, criterion, epochs=3)
        
        # Check that equation function and variables haven't changed
        equation_func_after = dual_model.sr_mlp._equation_func
        var_indices_after = dual_model.sr_mlp._var_indices
        
        assert equation_func_before is equation_func_after, "Equation function should be the same object"
        assert var_indices_before == var_indices_after, "Variable indices should remain unchanged"
        
        # Test that equation gives same output for same input
        with torch.no_grad():
            equation_output_after = dual_model.sr_mlp(test_input)
        
        diff = torch.abs(equation_output_before - equation_output_after)
        max_diff = torch.max(diff)
        
        assert max_diff < 1e-6, f"Equation output should be identical (diff: {max_diff})"
        
        print("✅ Confirmed: Symbolic equation parameters remain fixed during training")
        print(f"   Equation function: {equation_func_before}")
        print(f"   Variables used: {[f'x{i}' for i in var_indices_before]}")
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