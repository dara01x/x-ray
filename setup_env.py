#!/usr/bin/env python3
"""
Setup script for development environment.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Radiology AI Development Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = r"venv\Scripts\activate"
        pip_cmd = r"venv\Scripts\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch first (CPU version for compatibility)
    if not run_command(f"{pip_cmd} install torch torchvision --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch"):
        print("⚠️  PyTorch installation failed, trying default installation...")
        run_command(f"{pip_cmd} install torch torchvision", "Installing PyTorch (fallback)")
    
    # Install other requirements
    run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements")
    
    # Install the package in development mode
    run_command(f"{pip_cmd} install -e .", "Installing package in development mode")
    
    # Install development dependencies
    run_command(f"{pip_cmd} install pytest pytest-cov black isort flake8", "Installing development tools")
    
    # Create output directories
    directories = ["outputs/models", "outputs/results", "outputs/logs", "outputs/checkpoints"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print(f"1. Activate the virtual environment:")
    if os.name == 'nt':
        print(f"   {activate_cmd}")
    else:
        print(f"   {activate_cmd}")
    print("2. Download the NIH Chest X-ray dataset to the 'data' directory")
    print("3. Update the configuration in 'configs/config.yaml'")
    print("4. Run tests: pytest tests/ -v")
    print("5. Start training: python scripts/train.py")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🚀 CUDA available! GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n💻 CUDA not available, using CPU")
    except ImportError:
        print("\n⚠️  Could not check CUDA availability")


if __name__ == "__main__":
    main()
