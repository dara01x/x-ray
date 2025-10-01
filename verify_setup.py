#!/usr/bin/env python3
"""
Quick test script to verify ensemble model setup
"""

import os
from pathlib import Path

def check_kaggle_files():
    """Check if Kaggle files are present."""
    kaggle_dir = Path("./kaggle_outputs")
    
    required_files = {
        'champion_checkpoint': 'best_model_all_out_v1.pth',
        'champion_thresholds': 'optimal_thresholds_all_out_v1.json',
        'arnoweng_checkpoint': 'model.pth.tar',
        'ensemble_thresholds': 'optimal_thresholds_ensemble_final_v1.json',
        'ensemble_metrics': 'final_metrics_ensemble_final_v1.json',
        'classification_report': 'classification_report_ensemble_final_v1.txt'
    }
    
    print("🔍 Checking for required Kaggle files...")
    print("=" * 50)
    
    all_present = True
    for key, filename in required_files.items():
        filepath = kaggle_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {filename} - MISSING")
            all_present = False
    
    return all_present

def check_dependencies():
    """Check if required packages are installed."""
    print("\n🔧 Checking dependencies...")
    print("=" * 50)
    
    packages = [
        'torch', 'torchvision', 'torchxrayvision', 
        'albumentations', 'cv2', 'PIL', 'sklearn', 
        'pandas', 'numpy', 'matplotlib', 'yaml'
    ]
    
    missing = []
    for package in packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    return len(missing) == 0

def main():
    print("🚀 Ensemble Model Setup Verification")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check Kaggle files
    files_ok = check_kaggle_files()
    
    print("\n" + "=" * 60)
    print("📋 SETUP STATUS")
    print("=" * 60)
    
    if deps_ok:
        print("✅ Dependencies: All installed")
    else:
        print("❌ Dependencies: Some missing - run: pip install -r requirements_ensemble.txt")
    
    if files_ok:
        print("✅ Kaggle Files: All present")
        print("\n🎉 Setup complete! You can now run:")
        print("   python scripts/ensemble_inference.py --image your_image.png --kaggle-outputs ./kaggle_outputs")
    else:
        print("❌ Kaggle Files: Some missing")
        print("\n📥 Next steps:")
        print("1. Download the missing files from your Kaggle notebook")
        print("2. Place them in ./kaggle_outputs/ directory")  
        print("3. Run this script again to verify")
        print("4. See ./kaggle_outputs/DOWNLOAD_INSTRUCTIONS.txt for detailed help")
    
    print("\n🔍 For detailed setup help, run:")
    print("   python setup_ensemble.py --help")

if __name__ == "__main__":
    main()