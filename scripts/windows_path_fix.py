#!/usr/bin/env python3
"""
Windows Long Path Support Utility
Addresses the Windows 260-character path limit issue that affects PyTorch installation.
"""

import os
import sys
import platform
import subprocess
import tempfile
from pathlib import Path
import winreg


def check_long_path_enabled():
    """Check if Windows Long Path support is enabled."""
    try:
        if platform.system() != "Windows":
            return True, "Not a Windows system"
        
        # Check registry key
        key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        winreg.CloseKey(key)
        
        if value == 1:
            return True, "Long paths are enabled"
        else:
            return False, "Long paths are disabled"
            
    except FileNotFoundError:
        return False, "LongPathsEnabled registry key not found"
    except PermissionError:
        return False, "Permission denied - run as administrator"
    except Exception as e:
        return False, f"Error checking registry: {e}"


def enable_long_paths():
    """Enable Windows Long Path support (requires administrator privileges)."""
    try:
        if platform.system() != "Windows":
            print("⚠️ Not a Windows system - no action needed")
            return True
        
        # Check current status first
        enabled, message = check_long_path_enabled()
        if enabled:
            print(f"✅ Long paths already enabled: {message}")
            return True
        
        print("🔧 Attempting to enable Windows Long Path support...")
        
        # Try to enable via registry
        key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_WRITE)
        winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        
        print("✅ Long Path support enabled successfully!")
        print("⚠️ Note: You may need to restart your command prompt or system for changes to take effect.")
        return True
        
    except PermissionError:
        print("❌ Permission denied. Please run as administrator to enable Long Path support.")
        print("\n📋 Manual steps:")
        print("1. Open PowerShell as Administrator")
        print("2. Run: Set-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1")
        print("3. Restart your command prompt")
        return False
    except Exception as e:
        print(f"❌ Failed to enable long paths: {e}")
        return False


def get_short_path_location():
    """Get a suitable short path location for virtual environments."""
    candidates = [
        "C:\\venv",
        "C:\\python-envs", 
        "C:\\temp\\venv",
        os.path.join(tempfile.gettempdir(), "venv")
    ]
    
    for path in candidates:
        try:
            # Check if path is short enough and writable
            if len(path) < 50:  # Leave room for venv name and packages
                test_dir = os.path.join(path, "test")
                os.makedirs(test_dir, exist_ok=True)
                os.rmdir(test_dir)
                return path
        except (OSError, PermissionError):
            continue
    
    # Fallback to temp directory
    return os.path.join(tempfile.gettempdir(), "venv")


def create_short_path_venv(env_name="xray-ai"):
    """Create a virtual environment in a short path location."""
    short_path = get_short_path_location()
    venv_path = os.path.join(short_path, env_name)
    
    print(f"🔧 Creating virtual environment at: {venv_path}")
    print(f"📏 Path length: {len(venv_path)} characters")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(short_path, exist_ok=True)
        
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        
        print(f"✅ Virtual environment created successfully!")
        print(f"\n🚀 To activate this environment:")
        
        if platform.system() == "Windows":
            activate_script = os.path.join(venv_path, "Scripts", "Activate.ps1")
            print(f"   {activate_script}")
            print(f"   # Or in Command Prompt:")
            print(f"   {os.path.join(venv_path, 'Scripts', 'activate.bat')}")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")
            print(f"   source {activate_script}")
        
        return venv_path, activate_script
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return None, None


def install_pytorch_cpu():
    """Install PyTorch CPU version to avoid long path issues."""
    print("🔧 Installing PyTorch CPU version (shorter paths, faster install)...")
    
    try:
        # Install PyTorch CPU version
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
        
        subprocess.run(cmd, check=True)
        print("✅ PyTorch CPU installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False


def diagnose_path_issues():
    """Diagnose potential path-related installation issues."""
    print("🔍 Diagnosing Windows path issues...")
    print("=" * 50)
    
    # Check system
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python executable path length: {len(sys.executable)} characters")
    
    # Check long path support
    enabled, message = check_long_path_enabled()
    print(f"Long Path Support: {'✅ Enabled' if enabled else '❌ Disabled'} - {message}")
    
    # Check current working directory
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    print(f"Current directory path length: {len(cwd)} characters")
    
    # Check pip cache location
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"], 
                              capture_output=True, text=True, check=True)
        cache_dir = result.stdout.strip()
        print(f"Pip cache directory: {cache_dir}")
        print(f"Pip cache path length: {len(cache_dir)} characters")
    except Exception:
        print("Pip cache directory: Unable to determine")
    
    # Check temp directory
    temp_dir = tempfile.gettempdir()
    print(f"Temp directory: {temp_dir}")
    print(f"Temp directory path length: {len(temp_dir)} characters")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("=" * 50)
    
    if not enabled:
        print("❗ Enable Windows Long Path support (requires admin rights)")
        print("❗ Or use short path virtual environment")
    
    if len(cwd) > 100:
        print("❗ Current directory path is long - consider moving to shorter path")
    
    suggested_path = get_short_path_location()
    print(f"✅ Suggested short path for virtual environment: {suggested_path}")


def main():
    """Main function to handle Windows path issues."""
    print("🪟 Windows Long Path Support Utility")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser(description="Fix Windows long path issues for PyTorch installation")
    parser.add_argument("--check", action="store_true", help="Check current long path status")
    parser.add_argument("--enable", action="store_true", help="Enable long path support (requires admin)")
    parser.add_argument("--create-venv", action="store_true", help="Create virtual environment in short path")
    parser.add_argument("--install-pytorch", action="store_true", help="Install PyTorch CPU version")
    parser.add_argument("--diagnose", action="store_true", help="Diagnose path issues")
    parser.add_argument("--env-name", default="xray-ai", help="Virtual environment name")
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_path_issues()
    
    elif args.check:
        enabled, message = check_long_path_enabled()
        print(f"Long Path Support: {'✅ Enabled' if enabled else '❌ Disabled'}")
        print(f"Details: {message}")
    
    elif args.enable:
        enable_long_paths()
    
    elif args.create_venv:
        venv_path, activate_script = create_short_path_venv(args.env_name)
        if venv_path:
            print(f"\n📋 Next steps:")
            print(f"1. Activate the environment: {activate_script}")
            print(f"2. Install PyTorch: python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print(f"3. Install other requirements: pip install -r requirements.txt")
    
    elif args.install_pytorch:
        install_pytorch_cpu()
    
    else:
        # Default: run diagnosis and provide guidance
        diagnose_path_issues()
        print("\n🚀 Quick fix options:")
        print("=" * 50)
        print("1. Enable long paths (admin required):")
        print(f"   python {__file__} --enable")
        print("\n2. Create short-path virtual environment:")
        print(f"   python {__file__} --create-venv")
        print("\n3. Install PyTorch CPU (in activated environment):")
        print(f"   python {__file__} --install-pytorch")


if __name__ == "__main__":
    main()