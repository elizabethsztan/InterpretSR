import torch
import torch.nn as nn
import sys
import os
import pytest

# Add the src directory to Python path for absolute imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from interpretsr.mlp_sr import MLP_SR
from interpretsr.utils import *

class SimpleOriginal(nn.Module):
    """
    Simple class with just one MLP.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))

class SimpleWithSR(nn.Module):
    """ 
    Simple class wrapped with the MLP_SR wrapper.
    """
    def __init__(self):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
        self.mlp = MLP_SR(mlp)

class ComplexOriginal(nn.Module):
    """
    More complex class with several MLPs. 
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(20, 10), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

class ComplexWithSR(nn.Module):
    """
    More complex class where only some of the MLPs are wrapped in the MLP_SR wrapper.
    """
    def __init__(self):
        super().__init__()
        # Only wrap encoder and classifier with MLP_SR, leave decoder as-is
        encoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        self.encoder = MLP_SR(encoder)
        self.decoder = nn.Sequential(nn.Linear(20, 10), nn.ReLU())
        classifier = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        self.classifier = MLP_SR(classifier)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # input: 1x28x28 → output: 8x28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16x14x14 after pooling
        self.pool = nn.MaxPool2d(2, 2)  # downsample by 2
        self.mlp = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.Linear(64, 10))

class SimpleCNNWithSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # input: 1x28x28 → output: 8x28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16x14x14 after pooling
        self.pool = nn.MaxPool2d(2, 2)  # downsample by 2
        mlp = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.Linear(64, 10))
        self.mlp = MLP_SR(mlp)


def test_weight_loading_simple():
    """
    Test loading weights from an unwrapped simple model to a MLP_SR-wrapped simple model.
    """
    simple_model = SimpleOriginal()
    torch.save(simple_model.state_dict(), 'test_simple.pth')
    simple_model_SR = SimpleWithSR()
    weights = load_existing_weights_auto('test_simple.pth', simple_model_SR)

    try:
        simple_model_SR.load_state_dict(weights)
        os.remove('test_simple.pth')
    except Exception as e:
        pytest.fail(f"Weight loading failed with error: {e}")
        os.remove('test_simple.pth')


def test_weight_loading_complex():
    """
    Test loading weights from an unwrapped complex model to a MLP_SR-wrapped complex model.
    """
    complex_model = ComplexOriginal()
    torch.save(complex_model.state_dict(), 'test_complex.pth')
    complex_model_SR = ComplexWithSR()
    weights = load_existing_weights_auto('test_complex.pth', complex_model_SR)

    try:
        complex_model_SR.load_state_dict(weights)
        os.remove('test_complex.pth')
    except Exception as e:
        pytest.fail(f"Weight loading failed with error: {e}")
        os.remove('test_complex.pth')
    
    
def test_weight_loading_simple_SR():
    """
    Test loading weights from a MLP_SR-wrapped simple model to a MLP_SR-wrapped simple model.
    """
    simple_model_0 = SimpleWithSR()
    torch.save(simple_model_0.state_dict(), 'test_simple.pth')
    simple_model_1 = SimpleWithSR()
    weights = load_existing_weights_auto('test_simple.pth', simple_model_1)
    
    try:
        simple_model_1.load_state_dict(weights)
        os.remove('test_simple.pth')
    except Exception as e:
        pytest.fail(f"Weight loading failed with error: {e}")
        os.remove('test_simple.pth')

def test_weight_loading_complex_SR():
    """
    Test loading weights from a MLP_SR-wrapped complex model to a MLP_SR-wrapped complex model.
    """
    complex_model_0 = ComplexWithSR()
    torch.save(complex_model_0.state_dict(), 'test_complex.pth')
    complex_model_1 = ComplexWithSR()
    weights = load_existing_weights_auto('test_complex.pth', complex_model_1)
    
    try:
        complex_model_1.load_state_dict(weights)
        os.remove('test_complex.pth')
    except Exception as e:
        pytest.fail(f"Weight loading failed with error: {e}")
        os.remove('test_complex.pth')
    
    
def test_weight_loading_CNN():
    """
    Test loading weights from an unwrapped CNN model to a MLP_SR-wrapped CNN model.
    """
    cnn_model = SimpleCNN()
    torch.save(cnn_model.state_dict(), 'test_cnn.pth')
    complex_cnn_SR = SimpleCNNWithSR()
    weights = load_existing_weights_auto('test_cnn.pth', complex_cnn_SR)

    try:
        complex_cnn_SR.load_state_dict(weights)
        os.remove('test_cnn.pth')
    except Exception as e:
        pytest.fail(f"Weight loading failed with error: {e}")
        os.remove('test_cnn.pth')
    
    
