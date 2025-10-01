#!/usr/bin/env python3
"""
Kaggle Model Files Downloader
Helps download your ensemble model files from Kaggle
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_kaggle_api():
    """Check if Kaggle API is available and configured."""
    try:
        import kaggle
        # Test API by listing competitions (should work if configured)
        result = subprocess.run(['kaggle', 'competitions', 'list', '--page-size', '1'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Kaggle API is configured and working")
            return True
        else:
            print(f"❌ Kaggle API error: {result.stderr}")
            return False
    except ImportError:
        print("❌ Kaggle API not installed")
        return False
    except Exception as e:
        print(f"❌ Kaggle API error: {e}")
        return False

def setup_kaggle_api():
    """Guide user through Kaggle API setup."""
    print("\n🔧 KAGGLE API SETUP GUIDE")
    print("=" * 50)
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print("5. Place it in your home directory:")
    
    home_dir = Path.home()
    kaggle_dir = home_dir / '.kaggle'
    print(f"   {kaggle_dir}")
    
    print("6. Install Kaggle API if not already installed:")
    print("   pip install kaggle")
    
    # Create .kaggle directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    print(f"\n📁 Created directory: {kaggle_dir}")
    
    # Check if kaggle.json exists
    kaggle_json = kaggle_dir / 'kaggle.json'
    if kaggle_json.exists():
        print(f"✅ Found kaggle.json at {kaggle_json}")
        
        # Set permissions on Unix systems
        if os.name != 'nt':
            os.chmod(kaggle_json, 0o600)
            print("✅ Set proper permissions for kaggle.json")
    else:
        print(f"❌ kaggle.json not found at {kaggle_json}")
        print("   Please download it from Kaggle and place it there")

def manual_download_guide():
    """Provide manual download instructions."""
    print("\n📋 MANUAL DOWNLOAD GUIDE")
    print("=" * 50)
    
    files_needed = [
        ("best_model_all_out_v1.pth", "Your champion model checkpoint", "~100-300MB"),
        ("optimal_thresholds_all_out_v1.json", "Champion model thresholds", "~1KB"),
        ("model.pth.tar", "Arnoweng CheXNet model checkpoint", "~100-300MB"),
        ("optimal_thresholds_ensemble_final_v1.json", "Ensemble optimal thresholds", "~1KB"),
        ("final_metrics_ensemble_final_v1.json", "Complete performance metrics", "~2KB"),
        ("classification_report_ensemble_final_v1.txt", "Detailed classification report", "~3KB")
    ]
    
    print("Download these files from your Kaggle notebook:")
    print()
    
    for i, (filename, description, size) in enumerate(files_needed, 1):
        print(f"{i}. {filename}")
        print(f"   Description: {description}")
        print(f"   Expected size: {size}")
        print()
    
    kaggle_outputs = Path("./kaggle_outputs").absolute()
    print(f"Save all files to: {kaggle_outputs}")
    print()
    print("HOW TO DOWNLOAD FROM KAGGLE:")
    print("1. Go to your ab-ensemble.ipynb notebook on Kaggle")
    print("2. Make sure it has been 'Committed' (saved with output)")
    print("3. Look for the 'Output' section at the bottom")
    print("4. Click the download button for each file")
    print("5. Or download all at once if available")

def check_downloaded_files():
    """Check which files have been downloaded."""
    kaggle_outputs = Path("./kaggle_outputs")
    
    required_files = {
        'best_model_all_out_v1.pth': 'Champion model checkpoint',
        'optimal_thresholds_all_out_v1.json': 'Champion thresholds',
        'model.pth.tar': 'Arnoweng model checkpoint',
        'optimal_thresholds_ensemble_final_v1.json': 'Ensemble thresholds',
        'final_metrics_ensemble_final_v1.json': 'Performance metrics',
        'classification_report_ensemble_final_v1.txt': 'Classification report'
    }
    
    print("\n🔍 CHECKING DOWNLOADED FILES")
    print("=" * 50)
    
    found_files = 0
    total_size = 0
    
    for filename, description in required_files.items():
        filepath = kaggle_outputs / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"✅ {filename} ({size_mb:.1f} MB)")
            found_files += 1
        else:
            print(f"❌ {filename} - MISSING")
    
    print(f"\n📊 Status: {found_files}/{len(required_files)} files found")
    print(f"📦 Total size: {total_size:.1f} MB")
    
    return found_files == len(required_files)

def test_ensemble_setup():
    """Test if ensemble can be loaded."""
    try:
        # Try to import the ensemble model
        sys.path.append('.')
        from src.models.ensemble_model import load_ensemble_model
        
        kaggle_outputs = Path("./kaggle_outputs")
        champion_checkpoint = kaggle_outputs / "best_model_all_out_v1.pth"
        arnoweng_checkpoint = kaggle_outputs / "model.pth.tar"
        ensemble_thresholds = kaggle_outputs / "optimal_thresholds_ensemble_final_v1.json"
        
        if all(p.exists() for p in [champion_checkpoint, arnoweng_checkpoint]):
            print("\n🧪 TESTING ENSEMBLE MODEL LOADING")
            print("=" * 50)
            
            ensemble = load_ensemble_model(
                champion_checkpoint=str(champion_checkpoint),
                arnoweng_checkpoint=str(arnoweng_checkpoint),
                ensemble_thresholds=str(ensemble_thresholds) if ensemble_thresholds.exists() else None
            )
            
            model_info = ensemble.get_model_info()
            print("✅ Ensemble model loaded successfully!")
            print(f"   Device: {model_info['device']}")
            print(f"   Total parameters: {model_info['total_params']:,}")
            print(f"   Ensemble type: {model_info['ensemble_type']}")
            
            return True
        else:
            print("\n⚠️ Model files not complete - skipping load test")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing ensemble: {e}")
        return False

def main():
    print("🚀 KAGGLE ENSEMBLE MODEL SETUP")
    print("=" * 60)
    
    # Check current file status
    all_files_present = check_downloaded_files()
    
    if all_files_present:
        print("\n🎉 All files found! Testing model loading...")
        if test_ensemble_setup():
            print("\n✅ SETUP COMPLETE!")
            print("\n🎯 You can now run:")
            print("   python scripts/ensemble_inference.py --image your_image.png --kaggle-outputs ./kaggle_outputs --visualize")
            return
    
    # Guide through download process
    print(f"\n📥 DOWNLOAD OPTIONS:")
    print("1. Manual download from Kaggle (Recommended)")
    print("2. Setup Kaggle API for potential future automation")
    print("3. Check files again")
    
    while True:
        choice = input("\nEnter your choice (1/2/3) or 'q' to quit: ").strip().lower()
        
        if choice == '1':
            manual_download_guide()
            input("\nPress Enter after you've downloaded the files...")
            if check_downloaded_files():
                test_ensemble_setup()
            break
        elif choice == '2':
            setup_kaggle_api()
            print("\nAfter setting up Kaggle API, you'll still need to manually download")
            print("the files from your notebook output section.")
            break
        elif choice == '3':
            if check_downloaded_files():
                test_ensemble_setup()
            break
        elif choice == 'q':
            print("Exiting setup...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 'q'")

if __name__ == "__main__":
    main()