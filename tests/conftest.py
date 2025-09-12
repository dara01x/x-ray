"""
Test configuration and utilities.
"""

import pytest
import tempfile
import os
import numpy as np
import torch
from typing import Dict, Any


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        'project_name': 'test-radiology-ai',
        'data': {
            'data_dir': './test_data',
            'csv_file': 'test_data.csv',
            'valid_views': ['PA', 'AP']
        },
        'diseases': {
            'labels': ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration']
        },
        'model': {
            'backbone': 'densenet121',
            'num_classes': 4,
            'dropout_rate': 0.5,
            'pretrained_weights': 'densenet121-res224-all'
        },
        'training': {
            'batch_size': 8,
            'val_batch_size': 16,
            'num_epochs': 2,
            'early_stopping_patience': 5,
            'backbone_lr': 1e-5,
            'head_lr': 1e-3,
            'weight_decay': 1e-5,
            'loss': {
                'type': 'focal',
                'alpha': 0.25,
                'gamma': 2.0
            },
            'scheduler': {
                'type': 'onecycle',
                'pct_start': 0.3,
                'div_factor': 25,
                'final_div_factor': 1e4
            }
        },
        'preprocessing': {
            'image_size': 224,
            'clahe': {
                'clip_limit': 2.0,
                'tile_grid_size': [8, 8]
            }
        },
        'augmentation': {
            'horizontal_flip': 0.5,
            'rotation_limit': 15,
            'rotation_prob': 0.7
        },
        'validation': {
            'test_size': 0.2,
            'random_state': 42
        },
        'hardware': {
            'num_workers': 0,  # Use 0 for testing to avoid multiprocessing issues
            'pin_memory': False,
            'mixed_precision': False
        },
        'monitoring': {
            'metric_to_monitor': 'macro_auc',
            'monitor_mode': 'max',
            'log_frequency': 5
        },
        'output': {
            'model_dir': './test_outputs/models',
            'results_dir': './test_outputs/results',
            'logs_dir': './test_outputs/logs',
            'checkpoints_dir': './test_outputs/checkpoints'
        }
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image():
    """Create a sample grayscale image for testing."""
    # Create a random 224x224 grayscale image
    image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    return image


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(1, 1, 224, 224)


@pytest.fixture
def sample_labels():
    """Create sample multi-label targets."""
    return torch.rand(1, 4)  # 4 diseases


@pytest.fixture
def mock_dataframe():
    """Create a mock dataframe for testing."""
    import pandas as pd
    
    data = {
        'Image Index': [f'test_image_{i:03d}.png' for i in range(100)],
        'Patient ID': [f'patient_{i//10:02d}' for i in range(100)],
        'View Position': ['PA'] * 50 + ['AP'] * 50,
        'Finding Labels': ['No Finding'] * 80 + ['Atelectasis'] * 10 + ['Cardiomegaly'] * 5 + ['Atelectasis|Effusion'] * 5,
        'full_path': [f'/fake/path/test_image_{i:03d}.png' for i in range(100)]
    }
    
    df = pd.DataFrame(data)
    
    # Add disease columns
    diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration']
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
    
    return df


def create_mock_image_files(base_dir: str, num_images: int = 10):
    """Create mock image files for testing."""
    os.makedirs(base_dir, exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # Create a simple test image
        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
        # Save as PNG
        import cv2
        image_path = os.path.join(base_dir, f'test_image_{i:03d}.png')
        cv2.imwrite(image_path, image)
        image_paths.append(image_path)
    
    return image_paths


def cleanup_test_files(file_paths):
    """Clean up test files."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
