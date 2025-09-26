"""
Utility functions for the radiology AI project.
"""

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('x_ray_ai')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_{timestamp}.log')
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def create_output_directories(config: Dict[str, Any]):
    """
    Create output directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['output']['model_dir'],
        config['output']['results_dir'],
        config['output']['logs_dir'],
        config['output']['checkpoints_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(train_losses: List[float], val_losses: List[float],
                         val_metrics: List[Dict[str, float]], 
                         metric_name: str = 'macro_auc',
                         save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        val_metrics: Validation metrics
        metric_name: Metric to plot
        save_path: Optional save path
    """
    epochs = range(1, len(train_losses) + 1)
    metric_values = [metrics.get(metric_name, 0) for metrics in val_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(epochs, metric_values, 'g-', label=f'Validation {metric_name}')
    ax2.set_title(f'Validation {metric_name}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def plot_disease_distribution(df, disease_labels: List[str], 
                            save_path: Optional[str] = None):
    """
    Plot distribution of diseases in the dataset.
    
    Args:
        df: DataFrame with disease labels
        disease_labels: List of disease names
        save_path: Optional save path
    """
    disease_counts = df[disease_labels].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(disease_counts)), disease_counts.values)
    plt.title('Disease Distribution in Dataset')
    plt.xlabel('Disease')
    plt.ylabel('Number of Cases')
    plt.xticks(range(len(disease_counts)), disease_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, disease_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Disease distribution plot saved to {save_path}")
    
    plt.show()


def calculate_class_weights(df, disease_labels: List[str]) -> Dict[str, float]:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        df: DataFrame with disease labels
        disease_labels: List of disease names
        
    Returns:
        Dictionary of class weights
    """
    weights = {}
    total_samples = len(df)
    
    for disease in disease_labels:
        positive_samples = df[disease].sum()
        negative_samples = total_samples - positive_samples
        
        if positive_samples > 0:
            # Inverse frequency weighting
            pos_weight = negative_samples / positive_samples
            weights[disease] = pos_weight
        else:
            weights[disease] = 1.0
    
    return weights


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 224, 224)):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    total_params = count_parameters(model)
    print(f"Model Summary:")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Try to print model structure
    try:
        print("\nModel Architecture:")
        print(model)
    except Exception as e:
        print(f"Could not print model architecture: {e}")


def check_gpu_memory():
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
    else:
        print("CUDA not available")


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
