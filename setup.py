#!/usr/bin/env python3
"""
X-Ray AI - Unified Setup Script
Handles both direct dependency installation and setuptools package installation
"""

import subprocess
import sys
import os

def run_command(description, command, check=True):
    """Run command with progress indication"""
    try:
        print(f"[INFO] {description}...")
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error in {description}: {e.stderr if e.stderr else str(e)}")
        if check:
            sys.exit(1)

def direct_install():
    """Install dependencies directly (end-user mode)"""
    print("X-RAY AI - ONE-CLICK SETUP")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        sys.exit(1)
    
    print(f"[SUCCESS] Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install PyTorch CPU
    print("\n[INFO] Installing dependencies...")
    torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    try:
        run_command("Installing PyTorch CPU", torch_cmd)
    except SystemExit:
        print("[WARNING] PyTorch installation failed, trying without index URL...")
        try:
            run_command("Installing PyTorch (fallback)", "pip install torch torchvision torchaudio")
        except SystemExit:
            print("[ERROR] Failed to install PyTorch")
            sys.exit(1)
    
    # Install other dependencies
    requirements = [
        "flask>=2.0.1",
        "pillow>=8.0.0", 
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "torchxrayvision>=1.2.0",
        "requests>=2.25.0"
    ]
    
    for req in requirements:
        run_command(f"Installing {req.split('>=')[0]}", f"pip install {req}")
    
    print("\n[INFO] Installing medical AI packages...")
    run_command("Installing TorchXRayVision", "pip install torchxrayvision")
    run_command("Installing additional dependencies", "pip install -r requirements.txt", check=False)
    
    print("\n[SUCCESS] SETUP COMPLETE!")
    print("[INFO] TO START: python app.py")
    print("[INFO] THEN OPEN: http://localhost:5000")

def setup_mode():
    """Handle setuptools installation (developer mode)"""
    try:
        # Import setuptools and run setup
        from setuptools import setup, find_packages
        
        setup(
            name="x-ray-ai",
            version="1.0.0",
            description="Professional X-Ray Disease Classification AI",
            long_description=open("README.md", encoding="utf-8").read(),
            long_description_content_type="text/markdown",
            author="X-Ray AI Team", 
            packages=find_packages(),
            python_requires=">=3.8",
            install_requires=[
                "torch>=1.9.0",
                "torchvision>=0.10.0", 
                "flask>=2.0.1",
                "pillow>=8.0.0",
                "numpy>=1.21.0",
                "opencv-python>=4.5.0",
                "matplotlib>=3.3.0",
                "scikit-learn>=1.0.0",
                "pandas>=1.3.0",
                "torchxrayvision>=1.2.0",
                "requests>=2.25.0"
            ],
            extras_require={
                "dev": ["pytest>=6.0", "black", "flake8"]
            },
            entry_points={
                "console_scripts": [
                    "xray-ai=app:main"
                ]
            },
            classifiers=[
                "Development Status :: 4 - Beta",
                "Intended Audience :: Healthcare Industry",
                "License :: OSI Approved :: MIT License", 
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Medical Science Apps."
            ]
        )
        
    except ImportError:
        print("[ERROR] setuptools not available for package installation")
        print("[HINT] Try: pip install setuptools")
        sys.exit(1)

def main():
    """Main setup logic with automatic mode detection"""
    
    # Detect if running through setuptools (has specific setuptools arguments)
    setuptools_args = {
        'install', 'develop', 'build', 'sdist', 'bdist', 'bdist_wheel',
        'clean', 'check', 'upload', 'register', 'test', 'egg_info',
        'build_py', 'build_ext', 'build_clib', 'build_scripts',
        '--help-commands', '--name', '--version', '--fullname',
        '--author', '--author-email', '--maintainer', '--maintainer-email',
        '--contact', '--contact-email', '--url', '--license',
        '--description', '--long-description', '--platforms', '--classifiers',
        '--keywords', '--provides', '--requires', '--obsoletes'
    }
    
    # Check if any setuptools arguments are present
    if len(sys.argv) > 1:
        has_setuptools_args = any(arg in setuptools_args for arg in sys.argv[1:])
        
        # Special test mode
        if sys.argv[1] == '--test':
            print("SUCCESS: setup.py is working correctly!")
            print("[INFO] Mode detection: Direct install")
            print("[USAGE]")
            print("   python setup.py          # Install dependencies")  
            print("   python setup.py install  # Install as package")
            print("   python setup.py --test   # Test mode detection")
            return
            
        if has_setuptools_args:
            # Setuptools mode
            setup_mode()
            return
    
    # Default: Direct install mode
    direct_install()

if __name__ == "__main__":
    main()