#!/usr/bin/env python3
"""
Quick test script to verify project setup and run basic functionality tests.
"""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all main modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Test if CUDA is available
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CUDA not available, using CPU")
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import albumentations as A
        print(f"✅ Albumentations {A.__version__}")
    except ImportError as e:
        print(f"❌ Albumentations import failed: {e}")
        return False
    
    print("✅ All core imports successful!\n")
    return True


def test_project_structure():
    """Test that project structure is correct."""
    print("📁 Testing project structure...")
    
    required_dirs = [
        'src', 'src/models', 'src/data', 'src/training', 
        'src/evaluation', 'src/utils', 'scripts', 'tests', 
        'configs', 'outputs'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ Missing: {directory}/")
            return False
    
    required_files = [
        'requirements.txt', 'setup.py', 'README.md', 
        'configs/config.yaml', 'scripts/train.py',
        'scripts/evaluate.py', 'scripts/inference.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ Missing: {file}")
            return False
    
    print("✅ Project structure is correct!\n")
    return True


def test_config_loading():
    """Test configuration loading."""
    print("⚙️ Testing configuration loading...")
    
    try:
        from utils import load_config
        config = load_config('configs/config.yaml')
        
        # Check required sections
        required_sections = ['data', 'model', 'training', 'diseases']
        for section in required_sections:
            if section in config:
                print(f"✅ Config section: {section}")
            else:
                print(f"❌ Missing config section: {section}")
                return False
        
        # Check diseases list
        diseases = config.get('diseases', {}).get('labels', [])
        if len(diseases) == 14:
            print(f"✅ Disease labels: {len(diseases)} classes")
        else:
            print(f"❌ Expected 14 disease classes, found {len(diseases)}")
            return False
        
        print("✅ Configuration loading successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("🧠 Testing model creation...")
    
    try:
        import torch
        from models import ChestXrayModel
        
        # Create model
        model = ChestXrayModel(num_classes=14, dropout_rate=0.6)
        model.eval()  # Set to eval mode to avoid BatchNorm issues with single sample
        print("✅ Model created successfully")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 224, 224)
        
        with torch.no_grad():
            output = model(test_input)
        
        if output.shape == (1, 14):
            print(f"✅ Forward pass successful: {output.shape}")
        else:
            print(f"❌ Unexpected output shape: {output.shape}")
            return False
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print("✅ Model creation test successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_data_transforms():
    """Test data transformation pipeline."""
    print("🖼️ Testing data transforms...")
    
    try:
        import numpy as np
        import torch
        from data import create_transforms
        from utils import load_config
        
        config = load_config('configs/config.yaml')
        train_transform, val_transform = create_transforms(config)
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        
        # Test transforms
        train_result = train_transform(image=dummy_image)
        val_result = val_transform(image=dummy_image)
        
        train_tensor = train_result['image']
        val_tensor = val_result['image']
        
        expected_shape = (1, 224, 224)  # C, H, W
        
        if train_tensor.shape == expected_shape:
            print("✅ Training transform successful")
        else:
            print(f"❌ Training transform shape: {train_tensor.shape}, expected {expected_shape}")
            return False
        
        if val_tensor.shape == expected_shape:
            print("✅ Validation transform successful")
        else:
            print(f"❌ Validation transform shape: {val_tensor.shape}, expected {expected_shape}")
            return False
        
        print("✅ Data transforms test successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Data transforms test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Radiology AI Project Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_project_structure,
        test_config_loading,
        test_model_creation,
        test_data_transforms
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed\n")
        except Exception as e:
            print(f"❌ Test crashed: {e}\n")
            traceback.print_exc()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is successful!")
        print("\n📋 Next steps:")
        print("1. Download the NIH Chest X-ray dataset")
        print("2. Update data paths in configs/config.yaml")
        print("3. Run: python scripts/train.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
