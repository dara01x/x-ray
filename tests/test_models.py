"""
Tests for model architectures.
"""

import sys
import os
import pytest
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ChestXrayModel, create_model


class TestChestXrayModel:
    """Test cases for ChestXrayModel."""
    
    def test_model_creation(self, sample_config):
        """Test model creation with default parameters."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'head')
    
    def test_model_forward_pass(self, sample_config, sample_tensor):
        """Test forward pass through the model."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_tensor)
        
        assert output.shape == (1, 4)  # batch_size=1, num_classes=4
    
    def test_model_output_range(self, sample_config, sample_tensor):
        """Test that model outputs reasonable values."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_tensor)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output)
        
        # Check that probabilities are in valid range
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
    
    def test_model_training_mode(self, sample_config, sample_tensor):
        """Test model in training mode."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        model.train()
        
        # Use batch size > 1 to avoid BatchNorm issues
        batch_tensor = torch.randn(2, 1, 224, 224)
        output = model(batch_tensor)
        assert output.shape == (2, 4)
        
        # Test that gradients can be computed
        loss = torch.nn.BCEWithLogitsLoss()(output, torch.rand(2, 4))
        loss.backward()
        
        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_create_model_function(self, sample_config):
        """Test the create_model function."""
        model = create_model(sample_config)
        assert isinstance(model, ChestXrayModel)
        assert model.head[-1].out_features == sample_config['model']['num_classes']
    
    def test_model_parameter_count(self, sample_config):
        """Test that model has reasonable number of parameters."""
        model = create_model(sample_config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model should have a reasonable number of parameters
        assert total_params > 1000000  # At least 1M parameters
        assert trainable_params > 0    # Should have trainable parameters
    
    def test_model_feature_extraction(self, sample_config, sample_tensor):
        """Test feature extraction capability."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        model.eval()
        
        with torch.no_grad():
            features = model.get_feature_maps(sample_tensor)
        
        # Features should have the right shape for DenseNet
        assert len(features.shape) == 4  # Should be (batch, channels, height, width)
        assert features.shape[0] == 1    # Batch size
    
    def test_model_with_different_configs(self):
        """Test model with different configuration parameters."""
        # Test with different number of classes
        model1 = ChestXrayModel(num_classes=14, dropout_rate=0.6)
        model2 = ChestXrayModel(num_classes=5, dropout_rate=0.3)
        
        assert model1.head[-1].out_features == 14
        assert model2.head[-1].out_features == 5
        
        # Test forward pass with different inputs
        input1 = torch.randn(2, 1, 224, 224)
        input2 = torch.randn(3, 1, 224, 224)
        
        with torch.no_grad():
            output1 = model1(input1)
            output2 = model2(input2)
        
        assert output1.shape == (2, 14)
        assert output2.shape == (3, 5)
