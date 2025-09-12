"""
Training utilities and loss functions for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Any
import time
import os


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-label classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Computed focal loss
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class Trainer:
    """Main trainer class for chest X-ray disease classification."""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize criterion
        if config['training']['loss']['type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=config['training']['loss']['alpha'],
                gamma=config['training']['loss']['gamma']
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize optimizer with discriminative learning rates
        self.optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': config['training']['backbone_lr']},
            {'params': model.head.parameters(), 'lr': config['training']['head_lr']}
        ], weight_decay=config['training']['weight_decay'])
        
        # Initialize mixed precision scaler (updated for newer PyTorch)
        try:
            from torch.amp import GradScaler as NewGradScaler
            self.scaler = NewGradScaler('cuda', enabled=config['hardware']['mixed_precision'])
        except ImportError:
            # Fallback for older PyTorch versions
            self.scaler = GradScaler(enabled=config['hardware']['mixed_precision'])
        
        # Training state
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def setup_scheduler(self, train_loader):
        """Setup the learning rate scheduler."""
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config['training']['num_epochs']
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.config['training']['backbone_lr'], self.config['training']['head_lr']],
            total_steps=total_steps,
            pct_start=self.config['training']['scheduler']['pct_start'],
            div_factor=self.config['training']['scheduler']['div_factor'],
            final_div_factor=self.config['training']['scheduler']['final_div_factor']
        )
    
    def train_one_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        running_loss = 0.0
        num_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue
            
            images, labels = batch_data
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            with autocast(enabled=self.config['hardware']['mixed_precision']):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Step scheduler
            self.scheduler.step()
            
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            # Update progress bar
            if batch_idx % self.config['monitoring']['log_frequency'] == 0:
                current_lr = self.scheduler.get_last_lr()[1]  # Head LR
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.1e}'
                })
        
        progress_bar.close()
        return running_loss / num_samples if num_samples > 0 else 0.0
    
    def validate(self, val_loader, evaluator) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            evaluator: Evaluator instance
            
        Returns:
            Tuple of (validation_loss, metrics_dict)
        """
        return evaluator.evaluate(self.model, self.criterion, val_loader, self.device)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], checkpoint_path: str):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            checkpoint_path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric_value': self.best_metric,
            'metrics': metrics,
            'config': self.config,
            'train_losses_history': self.train_losses,
            'val_losses_history': self.val_losses,
            'val_metrics_history': self.val_metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_loader, val_loader, evaluator):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            evaluator: Evaluator instance
        """
        # Setup scheduler
        self.setup_scheduler(train_loader)
        
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        print(f"Monitoring: {self.config['monitoring']['metric_to_monitor']}")
        
        # Create output directories
        os.makedirs(self.config['output']['checkpoints_dir'], exist_ok=True)
        
        for epoch in range(self.config['training']['num_epochs']):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            print("-" * 50)
            
            # Training phase
            train_loss = self.train_one_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_metrics = self.validate(val_loader, evaluator)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            for metric_name, metric_value in val_metrics.items():
                print(f"Val {metric_name}: {metric_value:.4f}")
            
            # Check for improvement
            current_metric = val_metrics.get(self.config['monitoring']['metric_to_monitor'], 0.0)
            improved = False
            
            if self.config['monitoring']['monitor_mode'] == 'max':
                improved = current_metric > self.best_metric
            else:
                improved = current_metric < self.best_metric
            
            if improved:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                
                # Save best model
                checkpoint_path = os.path.join(
                    self.config['output']['checkpoints_dir'],
                    'best_model.pth'
                )
                self.save_checkpoint(epoch + 1, val_metrics, checkpoint_path)
                print(f"New best {self.config['monitoring']['metric_to_monitor']}: {self.best_metric:.4f}")
            else:
                self.epochs_no_improve += 1
                print(f"No improvement for {self.epochs_no_improve} epochs")
            
            # Early stopping
            if self.epochs_no_improve >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {self.config['training']['early_stopping_patience']} epochs without improvement")
                break
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch time: {epoch_time:.2f}s")
        
        print(f"\nTraining completed. Best {self.config['monitoring']['metric_to_monitor']}: {self.best_metric:.4f}")


def create_trainer(model: nn.Module, config: Dict, device: torch.device) -> Trainer:
    """
    Create a trainer instance.
    
    Args:
        model: Model to train
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Trainer instance
    """
    return Trainer(model, config, device)
