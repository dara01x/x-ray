"""
Model architectures for chest X-ray disease classification.
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv
from typing import Optional


class ChestXrayModel(nn.Module):
    """
    Multi-label chest X-ray disease classification model.
    
    Uses TorchXRayVision DenseNet121 as backbone with a custom classification head.
    """
    
    def __init__(self, num_classes: int = 14, dropout_rate: float = 0.6, 
                 pretrained_weights: str = "densenet121-res224-all"):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of disease classes to predict
            dropout_rate: Dropout rate for regularization
            pretrained_weights: Pretrained weights identifier for TorchXRayVision
        """
        super(ChestXrayModel, self).__init__()
        
        # Load pretrained backbone
        self.backbone = xrv.models.DenseNet(weights=pretrained_weights)
        self.backbone.op_threshs = None
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier.in_features
        
        # Replace the classifier with identity (we'll add our own head)
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        return self.head(features)
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps from the backbone for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps from the last convolutional layer
        """
        return self.backbone.features(x)


def create_model(config: dict) -> ChestXrayModel:
    """
    Create a model instance from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    return ChestXrayModel(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained_weights=config['model']['pretrained_weights']
    )


def load_model_weights(model: ChestXrayModel, checkpoint_path: str, 
                      device: torch.device) -> ChestXrayModel:
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle potential key naming differences
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Rename keys if necessary (for backward compatibility)
    renamed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('base_model.classifier', 'head').replace('base_model', 'backbone')
        renamed_state_dict[new_key] = value
    
    model.load_state_dict(renamed_state_dict, strict=False)
    return model
