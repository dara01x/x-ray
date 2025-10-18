#!/usr/bin/env python3
"""
X-ray AI Environment Setup Script
Comprehensive installer that handles all common installation challenges.
"""

import os
import sys
import platform
import subprocess
import tempfile
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import tarfile


class XrayAIInstaller:
    """Comprehensive installer for X-ray AI application."""
    
    def __init__(self):
        self.platform = platform.system()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.errors = []
        self.warnings = []
        self.install_log = []
        
    def log(self, message, level="INFO"):
        """Log installation progress."""
        print(f"[{level}] {message}")
        self.install_log.append(f"[{level}] {message}")
        
        if level == "ERROR":
            self.errors.append(message)
        elif level == "WARNING":
            self.warnings.append(message)
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        self.log("Checking Python version...")
        
        major, minor = sys.version_info.major, sys.version_info.minor
        self.log(f"Python version: {major}.{minor}.{sys.version_info.micro}")
        
        if major < 3 or (major == 3 and minor < 8):
            self.log("Python 3.8+ required", "ERROR")
            return False
        elif major == 3 and minor >= 13:
            self.log("Python 3.13+ detected - some packages may have compatibility issues", "WARNING")
        
        return True
    
    def check_windows_path_support(self):
        """Check and handle Windows long path issues."""
        if self.platform != "Windows":
            return True
            
        self.log("Checking Windows long path support...")
        
        try:
            # Import here to avoid issues on non-Windows systems
            import winreg
            
            key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ)
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            winreg.CloseKey(key)
            
            if value == 1:
                self.log("Long paths enabled ✅")
                return True
            else:
                self.log("Long paths disabled - will use short path solution", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"Could not check long path support: {e}", "WARNING")
            return False
    
    def create_virtual_environment(self, path=None):
        """Create virtual environment in optimal location."""
        self.log("Creating virtual environment...")
        
        if path is None:
            if self.platform == "Windows":
                # Use short path on Windows
                path = "C:\\venv\\xray-ai"
            else:
                path = "./venv"
        
        try:
            # Remove existing venv if it exists
            if os.path.exists(path):
                self.log(f"Removing existing virtual environment at {path}")
                shutil.rmtree(path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create virtual environment
            subprocess.run([sys.executable, "-m", "venv", path], check=True)
            self.log(f"Virtual environment created at: {path}")
            
            # Get activation script path
            if self.platform == "Windows":
                activate_script = os.path.join(path, "Scripts", "activate.bat")
                python_exe = os.path.join(path, "Scripts", "python.exe")
            else:
                activate_script = os.path.join(path, "bin", "activate")
                python_exe = os.path.join(path, "bin", "python")
            
            return path, activate_script, python_exe
            
        except Exception as e:
            self.log(f"Failed to create virtual environment: {e}", "ERROR")
            return None, None, None
    
    def install_packages_ordered(self, python_exe):
        """Install packages in optimal order to avoid conflicts."""
        self.log("Installing packages in optimal order...")
        
        # Define installation groups in order
        install_groups = [
            # Group 1: Core web framework
            {
                "name": "Web Framework",
                "packages": [
                    "Flask>=2.0.0,<3.0.0",
                    "Flask-CORS>=3.0.0,<5.0.0", 
                    "Werkzeug>=2.0.0,<3.0.0"
                ]
            },
            # Group 2: Basic numerical processing
            {
                "name": "Numerical Libraries", 
                "packages": [
                    "numpy>=1.21.0",
                    "setuptools>=60.0.0"
                ]
            },
            # Group 3: PyTorch (special handling)
            {
                "name": "PyTorch",
                "packages": [],  # Handled separately
                "special": "pytorch"
            },
            # Group 4: Data processing
            {
                "name": "Data Processing",
                "packages": [
                    "pandas>=1.3.0",
                    "scikit-learn>=1.0.0",
                    "scipy>=1.7.0"
                ]
            },
            # Group 5: Medical AI
            {
                "name": "Medical AI Libraries",
                "packages": [
                    "torchxrayvision>=1.0.0"
                ]
            },
            # Group 6: Image processing
            {
                "name": "Image Processing",
                "packages": [
                    "opencv-python>=4.5.0",
                    "Pillow>=8.0.0",
                    "albumentations>=1.3.0",
                    "scikit-image>=0.19.0"
                ]
            },
            # Group 7: Utilities
            {
                "name": "Utilities",
                "packages": [
                    "PyYAML>=6.0",
                    "requests>=2.25.0",
                    "tqdm>=4.60.0",
                    "typing-extensions>=3.10.0"
                ]
            },
            # Group 8: Visualization
            {
                "name": "Visualization",
                "packages": [
                    "matplotlib>=3.5.0",
                    "seaborn>=0.11.0"
                ]
            },
            # Group 9: Medical imaging
            {
                "name": "Medical Imaging",
                "packages": [
                    "pydicom>=2.0.0"
                ]
            },
            # Group 10: Testing (optional)
            {
                "name": "Testing",
                "packages": [
                    "pytest>=7.0.0",
                    "pytest-cov>=4.0.0"
                ],
                "optional": True
            }
        ]
        
        for group in install_groups:
            if group.get("special") == "pytorch":
                success = self.install_pytorch(python_exe)
                if not success:
                    return False
            else:
                success = self.install_package_group(python_exe, group)
                if not success and not group.get("optional", False):
                    return False
        
        return True
    
    def install_pytorch(self, python_exe):
        """Install PyTorch with CPU backend for compatibility."""
        self.log("Installing PyTorch (CPU version for compatibility)...")
        
        try:
            cmd = [
                python_exe, "-m", "pip", "install",
                "torch", "torchvision",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.log("PyTorch installed successfully ✅")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to install PyTorch: {e}", "ERROR")
            self.log(f"Error output: {e.stderr}", "ERROR")
            return False
    
    def install_package_group(self, python_exe, group):
        """Install a group of packages."""
        name = group["name"]
        packages = group["packages"]
        optional = group.get("optional", False)
        
        if not packages:
            return True
            
        self.log(f"Installing {name}...")
        
        try:
            cmd = [python_exe, "-m", "pip", "install"] + packages
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.log(f"{name} installed successfully ✅")
            return True
            
        except subprocess.CalledProcessError as e:
            level = "WARNING" if optional else "ERROR"
            self.log(f"Failed to install {name}: {e}", level)
            if e.stderr:
                self.log(f"Error output: {e.stderr}", level)
            return optional  # Return True for optional packages
    
    def verify_installation(self, python_exe):
        """Verify that all critical packages are installed."""
        self.log("Verifying installation...")
        
        critical_packages = [
            "flask", "torch", "torchvision", "numpy", "pandas", 
            "cv2", "PIL", "sklearn", "yaml", "albumentations"
        ]
        
        verification_script = f"""
import sys
import importlib

results = {{}}
for package in {critical_packages}:
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
            importlib.import_module(package)
        results[package] = True
    except ImportError as e:
        results[package] = False

print('VERIFICATION_RESULTS:' + str(results))
"""
        
        try:
            result = subprocess.run(
                [python_exe, "-c", verification_script],
                capture_output=True, text=True, check=True
            )
            
            # Parse results
            output = result.stdout
            if "VERIFICATION_RESULTS:" in output:
                results_str = output.split("VERIFICATION_RESULTS:")[1].strip()
                results = eval(results_str)
                
                success_count = sum(results.values())
                total_count = len(results)
                
                self.log(f"Package verification: {success_count}/{total_count} packages working")
                
                failed_packages = [pkg for pkg, status in results.items() if not status]
                if failed_packages:
                    self.log(f"Failed packages: {failed_packages}", "WARNING")
                
                return success_count >= total_count * 0.8  # 80% success rate
            else:
                self.log("Could not parse verification results", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Verification failed: {e}", "ERROR")
            return False
    
    def setup_directories(self):
        """Create necessary directories."""
        self.log("Setting up directories...")
        
        directories = [
            "uploads",
            "outputs",
            "outputs/models",
            "outputs/logs", 
            "outputs/checkpoints",
            "outputs/results",
            "outputs/web_results"
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.log(f"Created directory: {directory}")
            except Exception as e:
                self.log(f"Failed to create directory {directory}: {e}", "WARNING")
    
    def check_model_files(self):
        """Check for model files and provide guidance."""
        self.log("Checking for trained model files...")
        
        model_files = {
            'Champion model (primary)': 'models/best_model_all_out_v1.pth',
            'Champion model (backup)': 'outputs/models/best_model.pth',
            'Champion model (kaggle)': 'kaggle_outputs/best_model_all_out_v1.pth',
            'Arnoweng model (primary)': 'models/model.pth.tar',
            'Arnoweng model (kaggle)': 'kaggle_outputs/model.pth.tar',
            'Thresholds (kaggle)': 'kaggle_outputs/optimal_thresholds_ensemble_final_v1.json',
            'Metrics (kaggle)': 'kaggle_outputs/final_metrics_ensemble_final_v1.json'
        }
        
        found_files = []
        missing_files = []
        
        for name, path in model_files.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                found_files.append(f"{name}: {path} ({size_mb:.1f} MB)")
            else:
                missing_files.append(f"{name}: {path}")
        
        if found_files:
            self.log("Found model files:")
            for file_info in found_files:
                self.log(f"  ✅ {file_info}")
        
        if missing_files:
            self.log("Missing model files:", "WARNING")
            for file_info in missing_files:
                self.log(f"  ❌ {file_info}", "WARNING")
        
        return len(found_files) > 0
    
    def create_activation_script(self, venv_path, python_exe):
        """Create easy activation script."""
        self.log("Creating activation script...")
        
        if self.platform == "Windows":
            script_name = "activate_xray_ai.bat"
            content = f"""@echo off
echo Activating X-ray AI environment...
call "{os.path.join(venv_path, 'Scripts', 'activate.bat')}"
echo Environment activated! You can now run:
echo   python app_fixed.py
echo   python verify_setup.py
"""
        else:
            script_name = "activate_xray_ai.sh"
            content = f"""#!/bin/bash
echo "Activating X-ray AI environment..."
source "{os.path.join(venv_path, 'bin', 'activate')}"
echo "Environment activated! You can now run:"
echo "  python app_fixed.py"
echo "  python verify_setup.py"
"""
        
        try:
            with open(script_name, 'w') as f:
                f.write(content)
            
            if self.platform != "Windows":
                os.chmod(script_name, 0o755)
            
            self.log(f"Activation script created: {script_name}")
            return script_name
            
        except Exception as e:
            self.log(f"Failed to create activation script: {e}", "WARNING")
            return None
    
    def run_quick_test(self, python_exe):
        """Run a quick test to verify the setup."""
        self.log("Running quick functionality test...")
        
        test_script = """
try:
    import sys
    sys.path.append('src')
    
    # Test basic imports
    import torch
    import flask
    import numpy as np
    import cv2
    
    # Test Flask app import
    from app_fixed import app
    
    print("QUICK_TEST: SUCCESS - Basic functionality working")
    
except ImportError as e:
    print(f"QUICK_TEST: IMPORT_ERROR - {e}")
except Exception as e:
    print(f"QUICK_TEST: ERROR - {e}")
"""
        
        try:
            result = subprocess.run(
                [python_exe, "-c", test_script],
                capture_output=True, text=True, timeout=30
            )
            
            if "QUICK_TEST: SUCCESS" in result.stdout:
                self.log("Quick test passed ✅")
                return True
            else:
                self.log(f"Quick test failed: {result.stdout}", "WARNING")
                return False
                
        except Exception as e:
            self.log(f"Quick test error: {e}", "WARNING")
            return False
    
    def generate_summary_report(self, venv_path, activation_script):
        """Generate installation summary report."""
        has_models = self.check_model_files()
        
        report = f"""
{'='*60}
X-RAY AI INSTALLATION SUMMARY
{'='*60}

Installation Status: {'✅ SUCCESS' if not self.errors else '❌ PARTIAL'}
Python Version: {self.python_version}
Platform: {self.platform}
Virtual Environment: {venv_path}

Package Installation: {'✅ Completed' if not self.errors else '❌ Had Issues'}
Model Files: {'✅ Found' if has_models else '❌ Missing'}

FUNCTIONALITY LEVEL:
"""
        
        if not self.errors and has_models:
            report += """✅ FULL FUNCTIONALITY AVAILABLE
  - Web interface ✅
  - File uploads ✅  
  - AI predictions ✅
  - Disease classification ✅
  - Medical insights ✅
"""
        elif not self.errors:
            report += """⚠️ PARTIAL FUNCTIONALITY (Demo Mode)
  - Web interface ✅
  - File uploads ✅
  - AI predictions ❌ (No model files)
  - Disease classification ❌ (No model files)
  - Medical insights ❌ (No model files)
"""
        else:
            report += """❌ LIMITED FUNCTIONALITY
  - Installation had errors
  - Check error messages above
"""
        
        report += f"""
NEXT STEPS:
1. Activate environment: {activation_script or 'See venv activation'}
2. Test setup: python verify_setup.py
3. Start application: python app_fixed.py
4. Access at: http://localhost:5000

"""
        
        if not has_models:
            report += """TO ENABLE FULL AI FUNCTIONALITY:
1. Obtain trained model files from original training environment
2. Place files in correct directories:
   - models/best_model_all_out_v1.pth (Champion model)
   - models/model.pth.tar (Arnoweng model)
   - Or in kaggle_outputs/ directory
3. Re-run: python verify_setup.py

"""
        
        if self.errors:
            report += f"""
ERRORS ENCOUNTERED ({len(self.errors)}):
"""
            for error in self.errors:
                report += f"  ❌ {error}\n"
        
        if self.warnings:
            report += f"""
WARNINGS ({len(self.warnings)}):
"""
            for warning in self.warnings:
                report += f"  ⚠️ {warning}\n"
        
        report += f"\n{'='*60}\n"
        
        return report
    
    def install(self, venv_path=None, skip_tests=False):
        """Run complete installation process."""
        self.log("Starting X-ray AI installation...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check Windows path support
        self.check_windows_path_support()
        
        # Create virtual environment
        venv_path, activate_script, python_exe = self.create_virtual_environment(venv_path)
        if not venv_path:
            return False
        
        # Install packages
        if not self.install_packages_ordered(python_exe):
            self.log("Package installation had errors", "ERROR")
        
        # Verify installation
        if not skip_tests:
            if not self.verify_installation(python_exe):
                self.log("Installation verification failed", "WARNING")
            
            # Run quick test
            self.run_quick_test(python_exe)
        
        # Setup directories
        self.setup_directories()
        
        # Create activation script
        activation_script = self.create_activation_script(venv_path, python_exe)
        
        # Generate and display summary
        summary = self.generate_summary_report(venv_path, activation_script)
        print(summary)
        
        # Save installation log
        log_file = "installation_log.txt"
        with open(log_file, 'w') as f:
            f.write(summary)
            f.write("\n\nDetailed Installation Log:\n")
            f.write("=" * 40 + "\n")
            for entry in self.install_log:
                f.write(entry + "\n")
        
        self.log(f"Installation log saved to: {log_file}")
        
        return len(self.errors) == 0


def main():
    """Main installation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="X-ray AI Environment Setup")
    parser.add_argument("--venv-path", help="Custom virtual environment path")
    parser.add_argument("--skip-tests", action="store_true", help="Skip verification tests")
    parser.add_argument("--windows-fix", action="store_true", help="Run Windows path fix first")
    
    args = parser.parse_args()
    
    # Run Windows path fix if requested
    if args.windows_fix and platform.system() == "Windows":
        print("Running Windows path fix...")
        try:
            subprocess.run([sys.executable, "scripts/windows_path_fix.py", "--diagnose"])
        except FileNotFoundError:
            print("Windows path fix script not found - continuing with installation")
    
    # Run installation
    installer = XrayAIInstaller()
    success = installer.install(args.venv_path, args.skip_tests)
    
    if success:
        print("\n🎉 Installation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Installation completed with errors. Check the log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()