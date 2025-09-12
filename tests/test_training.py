"""
Tests for training utilities.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training import FocalLoss, Trainer, create_trainer
from models import ChestXrayModel


class TestFocalLoss:
    """Test cases for FocalLoss."""
    
    def test_focal_loss_creation(self):
        """Test FocalLoss creation with default parameters."""
        loss_fn = FocalLoss()
        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == 'mean'
    
    def test_focal_loss_forward(self, sample_labels):
        """Test FocalLoss forward pass."""
        loss_fn = FocalLoss()
        inputs = torch.randn(1, 4, requires_grad=True)
        targets = sample_labels
        
        loss = loss_fn(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0  # Loss should be non-negative
        
        # Test that gradients can be computed
        loss.backward()
        assert inputs.grad is not None
    
    def test_focal_loss_reduction_modes(self, sample_labels):
        """Test different reduction modes."""
        inputs = torch.randn(2, 4)
        targets = torch.rand(2, 4)
        
        # Test mean reduction
        loss_mean = FocalLoss(reduction='mean')(inputs, targets)
        assert loss_mean.dim() == 0  # Should be scalar
        
        # Test sum reduction
        loss_sum = FocalLoss(reduction='sum')(inputs, targets)
        assert loss_sum.dim() == 0  # Should be scalar
        
        # Test no reduction
        loss_none = FocalLoss(reduction='none')(inputs, targets)
        assert loss_none.shape == targets.shape  # Should match input shape
    
    def test_focal_loss_parameters(self):
        """Test FocalLoss with different parameters."""
        # Test with different alpha and gamma values
        loss1 = FocalLoss(alpha=0.5, gamma=1.0)
        loss2 = FocalLoss(alpha=0.75, gamma=3.0)
        
        inputs = torch.randn(1, 4)
        targets = torch.rand(1, 4)
        
        loss_val1 = loss1(inputs, targets)
        loss_val2 = loss2(inputs, targets)
        
        # Both should produce valid losses
        assert loss_val1.item() >= 0.0
        assert loss_val2.item() >= 0.0


class TestTrainer:
    """Test cases for Trainer class."""
    
    def test_trainer_creation(self, sample_config):
        """Test Trainer creation."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        device = torch.device('cpu')
        
        trainer = create_trainer(model, sample_config, device)
        
        assert isinstance(trainer, Trainer)
        assert trainer.model == model
        assert trainer.device == device
        assert isinstance(trainer.criterion, FocalLoss)
    
    def test_trainer_optimizer_setup(self, sample_config):
        """Test that optimizer is properly configured."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        device = torch.device('cpu')
        
        trainer = create_trainer(model, sample_config, device)
        
        # Check that optimizer has two parameter groups (backbone and head)
        assert len(trainer.optimizer.param_groups) == 2
        
        # Check learning rates
        backbone_lr = trainer.optimizer.param_groups[0]['lr']
        head_lr = trainer.optimizer.param_groups[1]['lr']
        
        assert backbone_lr == sample_config['training']['backbone_lr']
        assert head_lr == sample_config['training']['head_lr']
    
    def test_trainer_scheduler_setup(self, sample_config):
        """Test scheduler setup."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        device = torch.device('cpu')
        
        trainer = create_trainer(model, sample_config, device)
        
        # Create a mock data loader for scheduler setup
        class MockDataLoader:
            def __len__(self):
                return 10
        
        mock_loader = MockDataLoader()
        trainer.setup_scheduler(mock_loader)
        
        assert hasattr(trainer, 'scheduler')
        assert trainer.scheduler is not None
    
    def test_trainer_save_checkpoint(self, sample_config, temp_dir):
        """Test checkpoint saving functionality."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        device = torch.device('cpu')
        
        trainer = create_trainer(model, sample_config, device)
        
        # Setup scheduler for checkpoint saving
        class MockDataLoader:
            def __len__(self):
                return 5
        
        trainer.setup_scheduler(MockDataLoader())
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        metrics = {'macro_auc': 0.85, 'val_loss': 0.5}
        
        trainer.save_checkpoint(1, metrics, checkpoint_path)
        
        # Verify checkpoint was saved
        assert os.path.exists(checkpoint_path)
        
        # Load and verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'metrics' in checkpoint
        assert checkpoint['epoch'] == 1
        assert checkpoint['metrics'] == metrics
    
    def test_trainer_mixed_precision_setup(self, sample_config):
        """Test mixed precision training setup."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        device = torch.device('cpu')
        
        # Enable mixed precision in config
        sample_config['hardware']['mixed_precision'] = True
        
        trainer = create_trainer(model, sample_config, device)
        
        assert hasattr(trainer, 'scaler')
        assert trainer.scaler is not None


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_training_step_integration(self, sample_config, sample_tensor, sample_labels):
        """Test a single training step."""
        model = ChestXrayModel(num_classes=4, dropout_rate=0.5)
        device = torch.device('cpu')
        
        trainer = create_trainer(model, sample_config, device)
        
        # Setup scheduler
        class MockDataLoader:
            def __len__(self):
                return 5
        
        trainer.setup_scheduler(MockDataLoader())
        
        # Simulate one training step with batch size > 1
        model.train()
        trainer.optimizer.zero_grad()
        
        # Use batch size > 1 to avoid BatchNorm issues
        batch_tensor = torch.randn(2, 1, 224, 224)
        batch_labels = torch.rand(2, 4)
        
        outputs = model(batch_tensor)
        loss = trainer.criterion(outputs, batch_labels)
        
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        
        # Verify that loss is computed and gradients exist
        assert loss.item() >= 0.0
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
