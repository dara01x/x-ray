#!/usr/bin/env python3
"""
Script to download and integrate the new merged model from Kaggle.
This will help move your trained model from Kaggle to this repository.
"""

import os
import sys
import shutil
from pathlib import Path
import torch
import json

def setup_directories():
    """Create necessary directories for the new model."""
    dirs_to_create = [
        "outputs/models/new_merged_model",
        "outputs/checkpoints/new_merged_model", 
        "outputs/results/new_merged_model",
        "data/new_model_data"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def download_kaggle_dataset(username, dataset_name, download_path="./downloads"):
    """
    Download dataset from Kaggle using the Kaggle API.
    
    Args:
        username: Your Kaggle username
        dataset_name: Name of your dataset/model on Kaggle
        download_path: Local path to download to
    """
    try:
        import kaggle
        
        # Create download directory
        Path(download_path).mkdir(parents=True, exist_ok=True)
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            f"{username}/{dataset_name}",
            path=download_path,
            unzip=True
        )
        print(f"✓ Downloaded Kaggle dataset to {download_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        print("\nTo use Kaggle API, you need to:")
        print("1. Go to Kaggle.com → Account → Create New API Token")
        print("2. Download kaggle.json file")
        print("3. Place it in ~/.kaggle/ (or C:\\Users\\{username}\\.kaggle\\ on Windows)")
        print("4. Run: kaggle datasets list (to test)")
        return False

def analyze_downloaded_model(download_path):
    """Analyze the downloaded model files and structure."""
    download_dir = Path(download_path)
    
    if not download_dir.exists():
        print(f"❌ Download directory {download_path} does not exist")
        return None
    
    # Find model files
    model_files = []
    for ext in ['*.pth', '*.pt', '*.ckpt', '*.pkl', '*.h5', '*.onnx']:
        model_files.extend(download_dir.glob(f"**/{ext}"))
    
    # Find config/metadata files
    config_files = []
    for ext in ['*.json', '*.yaml', '*.yml', '*.txt', '*.md']:
        config_files.extend(download_dir.glob(f"**/{ext}"))
    
    analysis = {
        'model_files': [str(f) for f in model_files],
        'config_files': [str(f) for f in config_files],
        'all_files': [str(f) for f in download_dir.rglob('*') if f.is_file()]
    }
    
    print("\n📊 Analysis of Downloaded Files:")
    print(f"Model files found: {len(analysis['model_files'])}")
    for f in analysis['model_files']:
        print(f"  - {f}")
    
    print(f"\nConfig files found: {len(analysis['config_files'])}")
    for f in analysis['config_files']:
        print(f"  - {f}")
    
    return analysis

def inspect_pytorch_model(model_path):
    """Inspect PyTorch model to understand its structure."""
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n🔍 Inspecting PyTorch model: {model_path}")
        
        # Print checkpoint keys
        if isinstance(checkpoint, dict):
            print("Checkpoint keys:", list(checkpoint.keys()))
            
            # Try to get model architecture info
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            else:
                model_state = checkpoint
                
            # Print some layer names to understand architecture
            if isinstance(model_state, dict):
                layer_names = list(model_state.keys())[:10]  # First 10 layers
                print("Sample layer names:", layer_names)
                
                # Try to infer model type
                if any('densenet' in name.lower() for name in layer_names):
                    print("🎯 Detected: Likely DenseNet-based model")
                elif any('resnet' in name.lower() for name in layer_names):
                    print("🎯 Detected: Likely ResNet-based model")
                elif any('efficientnet' in name.lower() for name in layer_names):
                    print("🎯 Detected: Likely EfficientNet-based model")
                
                # Check output layer
                output_layers = [name for name in layer_names if 'classifier' in name or 'fc' in name or 'head' in name]
                if output_layers:
                    print("Output layers:", output_layers)
                    
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        return None

def create_model_integration_template():
    """Create a template for integrating the new model."""
    
    template_code = '''"""
New Merged Model Integration
Generated template for integrating your new merged model.
"""

import torch
import torch.nn as nn
from pathlib import Path

class NewMergedModel(nn.Module):
    """
    Template for your new merged model.
    Update this class based on your model's architecture.
    """
    
    def __init__(self, num_classes=14, **kwargs):
        super().__init__()
        # TODO: Define your model architecture here
        # This is a placeholder - replace with your actual model structure
        
        self.backbone = None  # Replace with your backbone
        self.classifier = nn.Linear(1024, num_classes)  # Adjust input size
        
    def forward(self, x):
        # TODO: Implement forward pass
        features = self.backbone(x)
        return self.classifier(features)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load model from your Kaggle checkpoint."""
        model = cls(**kwargs)
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        return model

def load_new_model(checkpoint_path, device='cpu'):
    """
    Load your new merged model.
    Update this function based on your model's requirements.
    """
    model = NewMergedModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model

# Example usage
if __name__ == "__main__":
    # Load your model
    checkpoint_path = "path/to/your/model.pth"  # Update this
    model = load_new_model(checkpoint_path)
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
'''

    with open("src/models/new_merged_model.py", "w") as f:
        f.write(template_code)
    
    print("✓ Created model integration template at src/models/new_merged_model.py")

def main():
    """Main function to guide the model integration process."""
    
    print("🚀 New Merged Model Integration Helper")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    print("\n📥 Model Download Options:")
    print("1. Download from Kaggle using API")
    print("2. Manual download (you'll need to download files manually)")
    print("3. Skip download (files already available locally)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🔧 Setting up Kaggle API download...")
        username = input("Enter your Kaggle username: ").strip()
        dataset_name = input("Enter your dataset/model name: ").strip()
        
        if download_kaggle_dataset(username, dataset_name):
            analysis = analyze_downloaded_model("./downloads")
    
    elif choice == "2":
        print("\n📋 Manual Download Instructions:")
        print("1. Go to your Kaggle dataset/notebook")
        print("2. Download all model files (.pth, .pt, etc.)")
        print("3. Download any config/metadata files")
        print("4. Place them in ./downloads/ folder")
        print("5. Run this script again with option 3")
        
    elif choice == "3":
        download_path = input("Enter path to your model files (default: ./downloads): ").strip()
        if not download_path:
            download_path = "./downloads"
        
        analysis = analyze_downloaded_model(download_path)
        
        if analysis and analysis['model_files']:
            # Inspect the first model file
            model_file = analysis['model_files'][0]
            checkpoint = inspect_pytorch_model(model_file)
    
    # Create integration template
    create_model_integration_template()
    
    print("\n✅ Setup Complete!")
    print("\n📝 Next Steps:")
    print("1. Update src/models/new_merged_model.py with your actual model architecture")
    print("2. Update configs/config.yaml to use your new model")
    print("3. Test the model loading with: python -c 'from src.models.new_merged_model import load_new_model; load_new_model(\"path/to/model.pth\")'")
    print("4. Update training/evaluation scripts if needed")
    
if __name__ == "__main__":
    main()