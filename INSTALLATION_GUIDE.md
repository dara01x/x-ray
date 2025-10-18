# X-ray AI Installation Guide

## 🎯 Complete Installation Instructions

This guide provides step-by-step instructions for installing the X-ray AI application, addressing all common challenges encountered during setup.

---

## 🚀 Quick Installation (Recommended)

### Method 1: Automatic Installer
```bash
# Clone repository
git clone https://github.com/dara01x/x-ray.git
cd x-ray

# Run automatic installer
python install_xray_ai.py

# Verify installation
python verify_setup.py

# Start application
python app_fixed.py
```

---

## 🛠️ Manual Installation

### Step 1: Prerequisites
- Python 3.8+ (3.9-3.12 recommended)
- pip (Python package manager)
- 4GB+ RAM, 2GB+ storage

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv xray-env

# Activate environment
# Windows:
xray-env\Scripts\activate
# macOS/Linux:
source xray-env/bin/activate
```

### Step 3: Install Dependencies
```bash
# Option A: Full installation
pip install -r requirements.txt

# Option B: Minimal installation (if full fails)
pip install -r requirements_minimal.txt

# Option C: Ordered installation (for conflicts)
python install_xray_ai.py --skip-tests
```

### Step 4: Verify Installation
```bash
python verify_setup.py
```

---

## 🚨 Common Installation Challenges & Solutions

### Challenge 1: Windows Long Path Support

**Problem**: Errors like `OSError: [Errno 2] No such file or directory` with very long file paths.

**Solutions**:
```bash
# Option 1: Diagnose and fix
python scripts/windows_path_fix.py --diagnose
python scripts/windows_path_fix.py --enable

# Option 2: Use short path virtual environment
python scripts/windows_path_fix.py --create-venv

# Option 3: Manual fix (requires admin)
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

### Challenge 2: Python 3.13+ Compatibility

**Problem**: Package compilation errors with newer Python versions.

**Solution**:
```bash
# Install PyTorch CPU first
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install minimal requirements
pip install -r requirements_minimal.txt
```

### Challenge 3: Dependency Conflicts

**Problem**: Package version conflicts during installation.

**Solution**:
```bash
# Use automatic installer (handles order)
python install_xray_ai.py

# Or manual ordered installation:
# 1. Core web framework
pip install Flask>=2.0.0,<3.0.0 Flask-CORS>=3.0.0 Werkzeug>=2.0.0,<3.0.0

# 2. Basic numerical
pip install numpy>=1.21.0 setuptools>=60.0.0

# 3. PyTorch
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Continue with remaining packages...
```

### Challenge 4: Demo Mode vs Full Functionality

**Problem**: Application starts but shows "Demo Mode" with no AI predictions.

**Understanding**:
- ✅ **Demo Mode**: Web interface works, but NO AI analysis
- ✅ **Full Mode**: Complete AI functionality with trained models

**Solution for Full Functionality**:
1. Obtain trained model files from original training environment
2. Place files in correct locations:
   ```
   models/best_model_all_out_v1.pth      # Champion model
   models/model.pth.tar                   # Arnoweng model
   
   # OR in kaggle_outputs/ directory:
   kaggle_outputs/best_model_all_out_v1.pth
   kaggle_outputs/model.pth.tar
   kaggle_outputs/optimal_thresholds_ensemble_final_v1.json
   ```
3. Re-run verification: `python verify_setup.py`

---

## 🔍 Installation Verification

### Check Installation Status
```bash
python verify_setup.py
```

**Expected Output for Success**:
```
✅ Python Environment: OK
✅ Dependencies: OK  
✅ Directory Structure: OK
✅ Model Files: full_ensemble / single_model / demo
✅ Basic Functionality: OK

🎉 READY TO USE!
```

### Check Application Status
```bash
# Start application
python app_fixed.py

# Check status via web API
curl http://localhost:5000/api/status
```

---

## 📊 Understanding Functionality Levels

### Full Functionality ✅
- **Requirements**: All dependencies + trained models
- **Capabilities**: 
  - Web interface ✅
  - File uploads ✅
  - AI predictions ✅
  - Disease classification ✅
  - Medical insights ✅

### Demo Mode ⚠️
- **Requirements**: Dependencies only (no models)
- **Capabilities**:
  - Web interface ✅
  - File uploads ✅
  - AI predictions ❌
  - Disease classification ❌
  - Medical insights ❌

### Error State ❌
- **Problem**: Missing dependencies or installation errors
- **Solution**: Re-run installer or check error messages

---

## 🔧 Advanced Configuration

### Custom Virtual Environment Location
```bash
python install_xray_ai.py --venv-path C:\custom\path\xray-ai
```

### Windows Path Issues Diagnosis
```bash
python scripts/windows_path_fix.py --diagnose
```

### Skip Tests During Installation
```bash
python install_xray_ai.py --skip-tests
```

### Environment Variables
```bash
# Force CPU mode (if GPU issues)
export CUDA_VISIBLE_DEVICES=""

# Increase pip timeout
export PIP_TIMEOUT=300
```

---

## 🚨 Troubleshooting Guide

### Installation Fails with "metadata-generation-failed"
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Use CPU-only PyTorch
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Use pre-compiled packages only
pip install --only-binary=all -r requirements.txt
```

### "No module named 'torchxrayvision'"
```bash
# Install dependencies first
pip install scikit-image tqdm albumentations

# Then install torchxrayvision
pip install torchxrayvision
```

### Port 5000 Already in Use
```bash
# Find process using port 5000
netstat -ano | findstr :5000    # Windows
lsof -i :5000                   # macOS/Linux

# Kill process or use different port
# App will automatically suggest alternatives
```

### Application Starts but No Predictions
- This is normal "Demo Mode" behavior
- Install trained model files for full functionality
- See "Challenge 4" above for solution

### Memory Errors During Analysis
```bash
# Close other applications
# Use smaller images (resize to 1024x1024 max)
# Restart the application
```

---

## 📁 File Organization

### Required Model Files for Full Functionality
```
models/
├── best_model_all_out_v1.pth      # Champion model (~100MB)
└── model.pth.tar                  # Arnoweng model (~50MB)

# OR

kaggle_outputs/
├── best_model_all_out_v1.pth
├── model.pth.tar
├── optimal_thresholds_ensemble_final_v1.json
├── final_metrics_ensemble_final_v1.json
└── classification_report_ensemble_final_v1.txt
```

### Installation Files
```
install_xray_ai.py              # Automatic installer
verify_setup.py                 # Setup verification
requirements.txt                # Full dependencies
requirements_minimal.txt        # Essential dependencies only
requirements_order.txt          # Installation order reference
scripts/windows_path_fix.py     # Windows utilities
```

---

## 🎯 Success Criteria

### ✅ Installation Successful When:
1. `python verify_setup.py` shows all green checkmarks
2. `python app_fixed.py` starts without errors
3. Web interface accessible at http://localhost:5000
4. File upload works (even in demo mode)

### ✅ Full Functionality When:
1. Model files are present and loaded
2. AI predictions work on uploaded images
3. Disease classification provides confidence scores
4. No "Demo Mode" warnings in console

---

## 📞 Getting Help

### Diagnostic Information
```bash
# Get detailed system info
python verify_setup.py

# Check Windows path issues
python scripts/windows_path_fix.py --diagnose

# View installation log
cat installation_log.txt  # Created by installer
```

### Error Reporting
When reporting issues, include:
1. Python version: `python --version`
2. Operating system
3. Verification output: `python verify_setup.py`
4. Error messages from console
5. Installation method used

---

## 🏁 Quick Start After Installation

```bash
# Activate environment (if using venv)
# Windows: xray-env\Scripts\activate
# macOS/Linux: source xray-env/bin/activate

# Start application
python app_fixed.py

# Open browser to http://localhost:5000
# Upload a chest X-ray image
# View AI analysis results (if models available)
```

---

**🎉 Installation Complete! The X-ray AI application is ready to help with medical image analysis.**