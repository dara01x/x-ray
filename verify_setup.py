#!/usr/bin/env python3
"""
Comprehensive X-ray AI Setup Verification
Checks dependencies, model files, and provides detailed guidance.
"""

import os
import sys
import platform
import importlib
import json
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n🔍 {title}")
    print("-" * 50)


def check_python_environment():
    """Check Python environment details."""
    print_section("Python Environment")
    
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"Virtual Environment: {'✅ Active' if in_venv else '❌ Not detected'}")
    
    if in_venv:
        print(f"Virtual Environment Path: {sys.prefix}")
    else:
        print("⚠️ Recommendation: Use a virtual environment for better dependency management")
    
    # Check Python version compatibility
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 8):
        print("❌ Python 3.8+ required for this application")
        return False
    elif major == 3 and minor >= 13:
        print("⚠️ Python 3.13+ detected - some packages may have compatibility issues")
    else:
        print("✅ Python version is compatible")
    
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print_section("Dependencies Check")
    
    # Define package groups
    package_groups = {
        "Core Web Framework": {
            "packages": ["flask", "flask_cors", "werkzeug"],
            "critical": True
        },
        "Deep Learning": {
            "packages": ["torch", "torchvision"],
            "critical": True
        },
        "Medical AI": {
            "packages": ["torchxrayvision"],
            "critical": True
        },
        "Numerical Computing": {
            "packages": ["numpy", "pandas", "scipy"],
            "critical": True
        },
        "Machine Learning": {
            "packages": ["sklearn"],
            "critical": True
        },
        "Image Processing": {
            "packages": ["cv2", "PIL", "albumentations", "skimage"],
            "critical": True
        },
        "Configuration & Utils": {
            "packages": ["yaml", "requests", "tqdm"],
            "critical": True
        },
        "Visualization": {
            "packages": ["matplotlib", "seaborn"],
            "critical": False
        },
        "Medical Imaging": {
            "packages": ["pydicom"],
            "critical": False
        },
        "Testing": {
            "packages": ["pytest"],
            "critical": False
        }
    }
    
    overall_status = True
    missing_critical = []
    missing_optional = []
    
    for group_name, group_info in package_groups.items():
        print(f"\n📦 {group_name}:")
        packages = group_info["packages"]
        critical = group_info["critical"]
        
        group_status = True
        for package in packages:
            try:
                # Special import handling for some packages
                if package == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif package == 'PIL':
                    import PIL
                    version = PIL.__version__
                elif package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif package == 'yaml':
                    import yaml
                    version = getattr(yaml, '__version__', 'unknown')
                elif package == 'skimage':
                    import skimage
                    version = getattr(skimage, '__version__', 'unknown')
                elif package == 'flask_cors':
                    import flask_cors
                    version = getattr(flask_cors, '__version__', 'unknown')
                else:
                    mod = importlib.import_module(package)
                    version = getattr(mod, '__version__', 'unknown')
                
                print(f"  ✅ {package} ({version})")
                
            except ImportError as e:
                print(f"  ❌ {package} - MISSING")
                group_status = False
                if critical:
                    missing_critical.append(package)
                    overall_status = False
                else:
                    missing_optional.append(package)
        
        if group_status:
            print(f"  🎉 {group_name} - All packages available")
        else:
            status_emoji = "❌" if critical else "⚠️"
            print(f"  {status_emoji} {group_name} - Some packages missing")
    
    # Summary
    print_section("Dependencies Summary")
    if overall_status:
        print("✅ All critical dependencies are installed")
    else:
        print("❌ Some critical dependencies are missing")
        print(f"Missing critical packages: {missing_critical}")
    
    if missing_optional:
        print(f"⚠️ Missing optional packages: {missing_optional}")
    
    return overall_status, missing_critical, missing_optional


def check_model_files():
    """Check for trained model files."""
    print_section("Model Files Check")
    
    # Define model file locations with priorities
    model_files = {
        'Champion Model': {
            'paths': [
                'models/best_model_all_out_v1.pth',
                'kaggle_outputs/best_model_all_out_v1.pth',
                'outputs/models/best_model.pth'
            ],
            'description': 'Primary AI model for disease classification',
            'critical': True
        },
        'Arnoweng Model': {
            'paths': [
                'models/model.pth.tar',
                'kaggle_outputs/model.pth.tar'
            ],
            'description': 'Secondary model for ensemble predictions',
            'critical': True
        },
        'Optimal Thresholds': {
            'paths': [
                'kaggle_outputs/optimal_thresholds_ensemble_final_v1.json',
                'kaggle_outputs/optimal_thresholds_all_out_v1.json',
                'outputs/optimal_thresholds.json',
                'models/optimal_thresholds_ensemble_final.json'
            ],
            'description': 'Optimized prediction thresholds',
            'critical': False
        },
        'Model Metrics': {
            'paths': [
                'kaggle_outputs/final_metrics_ensemble_final_v1.json',
                'models/final_metrics_ensemble_final.json'
            ],
            'description': 'Model performance metrics',
            'critical': False
        },
        'Classification Report': {
            'paths': [
                'kaggle_outputs/classification_report_ensemble_final_v1.txt',
                'models/classification_report_ensemble_final.txt'
            ],
            'description': 'Detailed classification performance report',
            'critical': False
        }
    }
    
    found_files = {}
    missing_critical = []
    functionality_level = "none"
    
    for file_type, file_info in model_files.items():
        paths = file_info['paths']
        description = file_info['description']
        critical = file_info['critical']
        
        found = False
        found_path = None
        
        for path in paths:
            if os.path.exists(path):
                found = True
                found_path = path
                break
        
        if found and found_path:
            size_mb = os.path.getsize(found_path) / (1024 * 1024)
            print(f"✅ {file_type}: {found_path} ({size_mb:.1f} MB)")
            found_files[file_type] = found_path
        else:
            status_emoji = "❌" if critical else "⚠️"
            print(f"{status_emoji} {file_type}: Missing")
            print(f"    Searched: {', '.join(paths)}")
            if critical:
                missing_critical.append(file_type)
    
    # Determine functionality level
    has_champion = 'Champion Model' in found_files
    has_arnoweng = 'Arnoweng Model' in found_files
    
    if has_champion and has_arnoweng:
        functionality_level = "full_ensemble"
        print("\n🎉 Full ensemble functionality available!")
    elif has_champion:
        functionality_level = "single_model"
        print("\n✅ Single model functionality available")
    elif has_arnoweng:
        functionality_level = "arnoweng_only"
        print("\n⚠️ Only Arnoweng model available (limited functionality)")
    else:
        functionality_level = "demo"
        print("\n❌ No trained models found - will run in demo mode")
    
    return functionality_level, found_files, missing_critical


def check_directory_structure():
    """Check if required directories exist."""
    print_section("Directory Structure")
    
    required_dirs = [
        ('configs', 'Configuration files', True),
        ('src', 'Source code', True),
        ('scripts', 'Utility scripts', True),
        ('templates', 'Web templates', True),
        ('static', 'Static web assets', True),
        ('uploads', 'File upload directory', False),
        ('outputs', 'Output directory', False),
        ('outputs/models', 'Model storage', False),
        ('outputs/logs', 'Log files', False),
        ('models', 'Model files', False),
        ('kaggle_outputs', 'Kaggle model files', False)
    ]
    
    all_critical_exist = True
    
    for dir_path, description, critical in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ {dir_path} - {description}")
        else:
            status_emoji = "❌" if critical else "⚠️"
            print(f"{status_emoji} {dir_path} - {description} (Missing)")
            if critical:
                all_critical_exist = False
    
    return all_critical_exist


def test_basic_functionality():
    """Test basic application functionality."""
    print_section("Basic Functionality Test")
    
    try:
        # Test imports
        print("Testing core imports...")
        import sys
        sys.path.append('src')
        
        from utils import load_config, get_device
        print("✅ Core utilities imported successfully")
        
        # Test Flask app import
        from app_fixed import app
        print("✅ Flask application imported successfully")
        
        # Test device detection
        device = get_device()
        print(f"✅ Device detection: {device}")
        
        # Test configuration loading
        if os.path.exists('configs/config.yaml'):
            config = load_config('configs/config.yaml')
            print("✅ Configuration loaded successfully")
        else:
            print("⚠️ Configuration file not found")
        
        print("🎉 Basic functionality test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def generate_installation_guidance(deps_status, missing_critical_deps, functionality_level):
    """Generate installation guidance based on current status."""
    print_section("Installation Guidance")
    
    if deps_status and functionality_level in ["full_ensemble", "single_model"]:
        print("🎉 INSTALLATION STATUS: COMPLETE")
        print("✅ All dependencies installed")
        print("✅ Trained models available")
        print("✅ Full AI functionality ready")
        print("\n🚀 Ready to run:")
        print("   python app_fixed.py")
        print("   Access at: http://localhost:5000")
        return
    
    print("⚠️ INSTALLATION STATUS: INCOMPLETE")
    
    # Dependencies issues
    if not deps_status:
        print("\n❌ DEPENDENCY ISSUES:")
        print(f"Missing critical packages: {missing_critical_deps}")
        print("\n🔧 To fix dependencies:")
        print("   Option 1 (Recommended): python install_xray_ai.py")
        print("   Option 2 (Manual): pip install -r requirements.txt")
        print("   Option 3 (Minimal): pip install -r requirements_minimal.txt")
        
        if platform.system() == "Windows":
            print("   For Windows path issues: python scripts/windows_path_fix.py")
    
    # Model files issues
    if functionality_level == "demo":
        print("\n❌ MODEL FILES MISSING:")
        print("The application will run in demo mode with limited functionality:")
        print("  ✅ Web interface will work")
        print("  ✅ File uploads will work")
        print("  ❌ AI predictions will NOT work")
        print("  ❌ Disease classification will NOT work")
        print("  ❌ Medical insights will NOT work")
        
        print("\n📥 TO ENABLE FULL AI FUNCTIONALITY:")
        print("1. Obtain trained model files from original training environment")
        print("2. Place files in correct directories:")
        print("   - Champion model → models/best_model_all_out_v1.pth")
        print("   - Arnoweng model → models/model.pth.tar")
        print("   - Or place in kaggle_outputs/ directory")
        print("3. Re-run this verification script")
    
    elif functionality_level in ["single_model", "arnoweng_only"]:
        print(f"\n⚠️ PARTIAL MODEL AVAILABILITY ({functionality_level}):")
        print("Some AI functionality will be available, but not full ensemble prediction")
        print("For best results, obtain both Champion and Arnoweng model files")


def main():
    """Main verification function."""
    print_header("X-RAY AI SETUP VERIFICATION")
    print("This script checks your installation and provides guidance for any issues.")
    
    # Check Python environment
    python_ok = check_python_environment()
    
    # Check dependencies
    deps_ok, missing_critical_deps, missing_optional_deps = check_dependencies()
    
    # Check model files
    functionality_level, found_files, missing_critical_models = check_model_files()
    
    # Check directory structure
    dirs_ok = check_directory_structure()
    
    # Test basic functionality
    if python_ok and deps_ok:
        func_test_ok = test_basic_functionality()
    else:
        func_test_ok = False
        print_section("Basic Functionality Test")
        print("⏭️ Skipping functionality test due to missing dependencies")
    
    # Generate guidance
    generate_installation_guidance(deps_ok, missing_critical_deps, functionality_level)
    
    # Final summary
    print_header("FINAL SUMMARY")
    
    print(f"Python Environment: {'✅ OK' if python_ok else '❌ Issues'}")
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ Missing packages'}")
    print(f"Directory Structure: {'✅ OK' if dirs_ok else '⚠️ Some missing'}")
    print(f"Model Files: {functionality_level}")
    print(f"Basic Functionality: {'✅ OK' if func_test_ok else '❌ Failed'}")
    
    if python_ok and deps_ok and functionality_level != "demo":
        print("\n🎉 READY TO USE!")
        print("Run: python app_fixed.py")
    elif python_ok and deps_ok:
        print("\n⚠️ DEMO MODE READY")
        print("Run: python app_fixed.py (limited functionality)")
    else:
        print("\n❌ INSTALLATION NEEDED")
        print("Run: python install_xray_ai.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()