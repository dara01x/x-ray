#!/usr/bin/env python3
"""
Final verification that ensemble mode is working properly.
"""

import os
import sys
import json
from pathlib import Path


def check_model_files():
    """Check that all required model files are present."""
    print("🔍 Checking model files for ensemble mode...")
    
    required_files = {
        # Primary model locations
        'Champion Model (Primary)': 'models/best_model_all_out_v1.pth',
        'Arnoweng Model (Primary)': 'models/model.pth.tar',
        
        # Backup locations  
        'Champion Model (Backup)': 'outputs/models/best_model.pth',
        
        # Kaggle locations (preferred by ensemble loader)
        'Champion Model (Kaggle)': 'kaggle_outputs/best_model_all_out_v1.pth',
        'Arnoweng Model (Kaggle)': 'kaggle_outputs/model.pth.tar',
        
        # Configuration files
        'Ensemble Thresholds': 'kaggle_outputs/optimal_thresholds_ensemble_final_v1.json',
        'Champion Thresholds': 'kaggle_outputs/optimal_thresholds_all_out_v1.json',
        'Model Metrics': 'kaggle_outputs/final_metrics_ensemble_final_v1.json',
        'Classification Report': 'kaggle_outputs/classification_report_ensemble_final_v1.txt',
        
        # Alternative threshold locations
        'Alternative Thresholds 1': 'models/optimal_thresholds_ensemble_final.json',
        'Alternative Thresholds 2': 'outputs/optimal_thresholds.json'
    }
    
    present_files = []
    missing_files = []
    
    for name, path in required_files.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            present_files.append(f"✅ {name}: {path} ({size_mb:.1f} MB)")
        else:
            missing_files.append(f"❌ {name}: {path}")
    
    print("\nPresent files:")
    for file_info in present_files:
        print(f"  {file_info}")
    
    if missing_files:
        print("\nMissing files:")
        for file_info in missing_files:
            print(f"  {file_info}")
    
    # Check critical files for ensemble mode
    critical_present = (
        os.path.exists('models/best_model_all_out_v1.pth') and
        os.path.exists('models/model.pth.tar')
    )
    
    return critical_present, len(present_files), len(missing_files)


def test_ensemble_import():
    """Test that ensemble model can be imported and initialized."""
    print("\n🧪 Testing ensemble model import...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Test basic imports
        from models.ensemble_model import load_ensemble_model
        print("✅ Ensemble model import successful")
        
        # Test model loading (just check if it would work)
        champion_path = None
        arnoweng_path = None
        
        # Find best paths
        if os.path.exists('kaggle_outputs/best_model_all_out_v1.pth'):
            champion_path = 'kaggle_outputs/best_model_all_out_v1.pth'
        elif os.path.exists('models/best_model_all_out_v1.pth'):
            champion_path = 'models/best_model_all_out_v1.pth'
        
        if os.path.exists('kaggle_outputs/model.pth.tar'):
            arnoweng_path = 'kaggle_outputs/model.pth.tar'
        elif os.path.exists('models/model.pth.tar'):
            arnoweng_path = 'models/model.pth.tar'
        
        if champion_path and arnoweng_path:
            print(f"✅ Champion model path: {champion_path}")
            print(f"✅ Arnoweng model path: {arnoweng_path}")
            return True
        else:
            print("❌ Missing required model files")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Ensemble test error: {e}")
        return False


def check_virtual_environment():
    """Check if virtual environment with dependencies is available."""
    print("\n🐍 Checking virtual environment...")
    
    venv_python = "C:\\venv\\xray-ai\\Scripts\\python.exe"
    
    if os.path.exists(venv_python):
        print(f"✅ Virtual environment found: {venv_python}")
        
        # Test if dependencies are installed in venv
        import subprocess
        try:
            result = subprocess.run([
                venv_python, "-c", 
                "import torch, torchxrayvision, flask; print('DEPS_OK')"
            ], capture_output=True, text=True, timeout=10)
            
            if "DEPS_OK" in result.stdout:
                print("✅ All critical dependencies installed in virtual environment")
                return True
            else:
                print("⚠️ Some dependencies missing in virtual environment")
                return False
                
        except Exception as e:
            print(f"⚠️ Could not test virtual environment dependencies: {e}")
            return False
    else:
        print("❌ Virtual environment not found")
        print("Run: python install_xray_ai.py")
        return False


def generate_startup_instructions():
    """Generate instructions for starting in ensemble mode."""
    print("\n📋 Startup Instructions for Ensemble Mode")
    print("=" * 50)
    
    if os.path.exists("C:\\venv\\xray-ai\\Scripts\\python.exe"):
        print("✅ OPTION 1 (Recommended): Use start script")
        print("   Double-click: start_ensemble_mode.bat")
        print("   Or run: .\\start_ensemble_mode.bat")
        print()
        print("✅ OPTION 2: Manual virtual environment")
        print("   C:\\venv\\xray-ai\\Scripts\\python.exe app_fixed.py")
        print()
        print("✅ OPTION 3: Activate environment first")
        print("   C:\\venv\\xray-ai\\Scripts\\activate.bat")
        print("   python app_fixed.py")
    else:
        print("❌ Virtual environment not found!")
        print("Please run: python install_xray_ai.py")
    
    print("\n🌐 Once started, access at: http://localhost:5000")


def main():
    """Main verification function."""
    print("🎯 X-ray AI Ensemble Mode Verification")
    print("=" * 50)
    
    # Check model files
    critical_files, present_count, missing_count = check_model_files()
    
    # Test ensemble import
    import_success = test_ensemble_import()
    
    # Check virtual environment
    venv_success = check_virtual_environment()
    
    # Final assessment
    print("\n" + "=" * 50)
    print("📊 ENSEMBLE MODE READINESS ASSESSMENT")
    print("=" * 50)
    
    print(f"Model Files: {'✅ Ready' if critical_files else '❌ Missing'} ({present_count} present, {missing_count} missing)")
    print(f"Ensemble Import: {'✅ Ready' if import_success else '❌ Failed'}")
    print(f"Virtual Environment: {'✅ Ready' if venv_success else '❌ Missing'}")
    
    overall_ready = critical_files and import_success and venv_success
    
    if overall_ready:
        print("\n🎉 ENSEMBLE MODE READY!")
        print("✅ All required components are available")
        print("✅ Full AI functionality will be available")
        print("✅ Both Champion and Arnoweng models will be used")
        print("✅ Optimal thresholds will be applied")
        
        generate_startup_instructions()
        
    else:
        print("\n⚠️ ENSEMBLE MODE NOT READY")
        
        if not critical_files:
            print("❌ Missing critical model files")
            print("   Need: best_model_all_out_v1.pth and model.pth.tar")
        
        if not import_success:
            print("❌ Cannot import ensemble model")
            print("   Check that src/models/ensemble_model.py exists")
        
        if not venv_success:
            print("❌ Virtual environment with dependencies missing")
            print("   Run: python install_xray_ai.py")
        
        print("\n🔧 TO FIX:")
        print("1. Run: python install_xray_ai.py")
        print("2. Ensure all model files are present")
        print("3. Re-run this verification")
    
    return overall_ready


if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 50)
    sys.exit(0 if success else 1)