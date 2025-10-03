#!/usr/bin/env python3
"""
Utility script to prepare the web application
Converts CSV thresholds to JSON and checks setup
"""

import os
import json
import sys
import pandas as pd
import traceback

def convert_thresholds_to_json():
    """Convert CSV thresholds to JSON format."""
    csv_path = 'kaggle_outputs/optimal_thresholds_ensemble_final.csv'
    json_path = 'outputs/optimal_thresholds.json'
    
    try:
        if not os.path.exists(csv_path):
            print(f"⚠️ Thresholds CSV not found at {csv_path}")
            return False
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Convert to dictionary
        thresholds = {}
        for _, row in df.iterrows():
            thresholds[row['Disease']] = float(row['Optimal_Threshold'])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # Save as JSON
        with open(json_path, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        print(f"✅ Thresholds converted and saved to {json_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting thresholds: {e}")
        traceback.print_exc()
        return False

def check_model_files():
    """Check if model files are available."""
    files_to_check = [
        'kaggle_outputs/model.pth.tar',
        'outputs/models/best_model.pth',
        'configs/config.yaml'
    ]
    
    available_files = []
    missing_files = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            available_files.append(file_path)
            print(f"✅ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"⚠️ Missing: {file_path}")
    
    return available_files, missing_files

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask',
        'flask_cors', 
        'torch',
        'torchvision',
        'torchxrayvision',
        'opencv-python',
        'albumentations',
        'numpy',
        'pandas',
        'PIL',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'yaml':
                import yaml
            elif package == 'flask_cors':
                from flask_cors import CORS
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    return missing_packages

def create_demo_image():
    """Create a demo X-ray image for testing."""
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple demo image
        demo_dir = 'demo_images'
        os.makedirs(demo_dir, exist_ok=True)
        
        # Create a 224x224 grayscale image with some patterns
        img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        
        # Add some chest-like patterns
        center_x, center_y = 112, 112
        for i in range(224):
            for j in range(224):
                distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if distance < 80:
                    img_array[i, j] = min(255, img_array[i, j] + 30)
        
        # Save as PNG
        demo_path = os.path.join(demo_dir, 'demo_chest_xray.png')
        Image.fromarray(img_array, mode='L').save(demo_path)
        
        print(f"✅ Demo image created: {demo_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating demo image: {e}")
        return False

def main():
    """Main setup function."""
    print("🔬 Chest X-ray AI Web Application Setup")
    print("=" * 50)
    
    # Convert thresholds
    print("\n📊 Converting thresholds to JSON...")
    convert_thresholds_to_json()
    
    # Check model files
    print("\n🧠 Checking model files...")
    available, missing = check_model_files()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install them with: pip install " + " ".join(missing_deps))
    else:
        print("\n✅ All dependencies installed")
    
    # Create demo image
    print("\n🖼️ Creating demo image...")
    create_demo_image()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    if available:
        print(f"✅ Model files available: {len(available)}")
        if 'kaggle_outputs/model.pth.tar' in available:
            print("   → Ensemble model ready")
        elif 'outputs/models/best_model.pth' in available:
            print("   → Single model ready")
    else:
        print("⚠️ No model files found - will run in demo mode")
    
    if not missing_deps:
        print("✅ All dependencies satisfied")
        print("🚀 Ready to start web application!")
        print("\nTo start the server, run:")
        print("   python app.py")
        print("\nThen open: http://localhost:5000")
    else:
        print("❌ Missing dependencies - install them first")
    
    print("\n💡 USAGE TIPS:")
    print("   • Upload PNG, JPG, or JPEG images")
    print("   • Maximum file size: 16MB")
    print("   • Results are for educational purposes only")
    print("   • Always consult healthcare professionals")

if __name__ == "__main__":
    main()