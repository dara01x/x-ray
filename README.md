# 🩺 X-Ray AI Web Application

**Professional AI-powered chest X-ray analysis with 14 disease detection**

A comprehensive medical AI application that uses advanced deep learning ensemble models to analyze chest X-ray images and detect 14 different thoracic diseases including Pneumonia, Cardiomegaly, Atelectasis, and more.

## 🎯 Two Usage Modes

### ✅ **FULL FUNCTIONALITY** (Recommended)
- **Real AI predictions** with ensemble models
- **14 disease classification** with confidence scores
- **Medical insights** and detailed analysis
- **High accuracy** validated on medical datasets

### ⚠️ **DEMO MODE** (Limited)
- Web interface works but **NO real AI analysis**
- Useful for testing interface only
- Shows placeholder results

---

## 🚀 Installation Instructions

### **METHOD 1: One-Click Installation (Recommended)**

**Step 1: Clone the Repository**
```bash
git clone https://github.com/dara01x/x-ray.git
cd x-ray
```

**Step 2: Run Automatic Installer**
```bash
python install_xray_ai.py
```
*This handles all dependencies, creates virtual environment, and sets up everything automatically.*

**Step 3: Verify Installation**
```bash
# Using the created virtual environment
C:\venv\xray-ai\Scripts\python.exe verify_setup.py
```

**Step 4: Start Application**
```bash
# Easy way (double-click)
start_ensemble_mode.bat

# Or command line
C:\venv\xray-ai\Scripts\python.exe app_fixed.py
```

**Step 5: Access Application**
- Open browser to `http://localhost:5000`
- Upload chest X-ray image (PNG, JPG, JPEG, DCM)
- Get real AI analysis with confidence scores!

---

### **METHOD 2: Manual Installation**

**For Advanced Users or Troubleshooting**

```bash
# 1. Create virtual environment
python -m venv xray-env

# 2. Activate environment
# Windows:
xray-env\Scripts\activate
# macOS/Linux:
source xray-env/bin/activate

# 3. Install PyTorch first (prevents conflicts)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Verify installation
python verify_setup.py

# 6. Start application
python app_fixed.py
```

---

## 🔍 **How to Verify Your Installation**

### **Check Installation Status:**
```bash
python verify_setup.py
```

**Expected Output for Full Functionality:**
```
✅ Dependencies: OK
✅ Model Files: full_ensemble
✅ READY TO USE!
```

### **Check Ensemble Mode Specifically:**
```bash
python verify_ensemble_mode.py
```

**Expected Output:**
```
🎉 ENSEMBLE MODE READY!
✅ All required components are available
✅ Full AI functionality will be available
```

---

## 🌐 **Using the Application**

1. **Start the Application:**
   - Double-click `start_ensemble_mode.bat` 
   - Or run: `python app_fixed.py`

2. **Open Your Browser:**
   - Go to `http://localhost:5000`

3. **Upload an Image:**
   - Drag & drop or click to select chest X-ray
   - Supported: PNG, JPG, JPEG, DCM files
   - Maximum size: 16MB

4. **View Results:**
   - AI analysis appears in 5-15 seconds
   - See confidence scores for 14 diseases
   - Interactive charts with detailed information

---

## 🧪 **Disease Detection Capabilities**

The AI ensemble can identify these 14 thoracic conditions:

| Disease | Description | Severity Level |
|---------|-------------|----------------|
| **Atelectasis** | Lung collapse or incomplete expansion | Moderate |
| **Cardiomegaly** | Enlarged heart condition | High |
| **Effusion** | Fluid accumulation around lungs | Moderate |
| **Infiltration** | Inflammatory changes in lung tissue | Moderate |
| **Mass** | Tumor or lesion detection | High |
| **Nodule** | Small round opacity in lungs | Moderate |
| **Pneumonia** | Lung infection and inflammation | High |
| **Pneumothorax** | Collapsed lung due to air leak | High |
| **Consolidation** | Lung tissue filled with liquid | Moderate |
| **Edema** | Fluid accumulation in lung tissue | High |
| **Emphysema** | Damaged air sacs in lungs | High |
| **Fibrosis** | Lung scarring and tissue damage | High |
| **Pleural Thickening** | Thickened lung lining | Moderate |
| **Hernia** | Organ displacement | Moderate |

---

## 📋 **System Requirements**

### **Minimum Requirements:**
- **Operating System**: Windows 10+, macOS 10.14+, Linux Ubuntu 18.04+
- **Python**: 3.8 or higher (3.9-3.12 recommended)
- **RAM**: 4GB minimum
- **Storage**: 2GB free space
- **Internet**: Required for initial setup only

### **Recommended Specifications:**
- **Python**: 3.10 or 3.11 (best compatibility)
- **RAM**: 8GB or more
- **Storage**: 5GB free space (for models and cache)
- **CPU**: Multi-core processor for faster analysis

---

## 🛠️ **Troubleshooting Common Issues**

### **Issue 1: Application Shows "Demo Mode"**
**Problem**: Web interface works but no AI predictions
**Solution**: 
```bash
# This means model files are missing - normal for first install
# Check status:
python verify_setup.py

# Expected: Will show "Model Files: demo" 
# This is normal - the web interface will work for testing
```

### **Issue 2: "DLL load failed" or Import Errors**
**Problem**: PyTorch or dependencies not properly installed
**Solution**:
```bash
# Run automatic installer to fix dependencies:
python install_xray_ai.py

# Or install PyTorch manually:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Issue 3: Windows Path Length Errors**
**Problem**: `OSError: [Errno 2] No such file or directory` with long paths
**Solution**:
```bash
# Use our Windows fix utility:
python scripts/windows_path_fix.py --diagnose
python scripts/windows_path_fix.py --create-venv

# This creates virtual environment in short path location
```

### **Issue 4: Package Installation Conflicts**
**Problem**: `metadata-generation-failed` or version conflicts
**Solution**:
```bash
# Use ordered installation:
python install_xray_ai.py

# Or minimal requirements:
pip install -r requirements_minimal.txt
```

### **Issue 5: Port 5000 Already in Use**
**Problem**: `Address already in use` error
**Solution**:
```bash
# Find what's using port 5000:
netstat -ano | findstr :5000    # Windows
lsof -i :5000                   # macOS/Linux

# Kill the process or the app will suggest alternative ports
```

### **Issue 6: Slow Analysis or Memory Errors**
**Problem**: Analysis takes too long or crashes
**Solution**:
- Close other applications to free RAM
- Use smaller images (resize to 1024x1024 max)
- Restart the application
- First analysis is always slower (model loading)

---

## 🔧 **Advanced Configuration**

### **Custom Installation Path:**
```bash
# Install virtual environment in custom location:
python install_xray_ai.py --venv-path "D:\custom\path\xray-ai"
```

### **Skip Tests During Installation:**
```bash
# For faster installation (advanced users):
python install_xray_ai.py --skip-tests
```

### **Environment Variables:**
```bash
# Force CPU mode (if GPU issues):
set CUDA_VISIBLE_DEVICES=""        # Windows
export CUDA_VISIBLE_DEVICES=""     # macOS/Linux

# Increase timeout for slow connections:
set PIP_TIMEOUT=300                 # Windows
export PIP_TIMEOUT=300              # macOS/Linux
```

### **Alternative Requirements Files:**
```bash
# If main requirements fail:
pip install -r requirements_minimal.txt

# See installation order:
cat requirements_order.txt
```

---

## 📁 **Project Structure**

```
x-ray/
├── 📄 README.md                    # This file
├── 🚀 install_xray_ai.py           # Automatic installer
├── ✅ verify_setup.py               # Installation verification
├── 🎯 verify_ensemble_mode.py      # Ensemble mode verification  
├── 🌐 app_fixed.py                 # Main web application
├── 🦇 start_ensemble_mode.bat      # Easy start script (Windows)
├── 📋 requirements.txt             # Full dependencies
├── 📋 requirements_minimal.txt     # Essential dependencies
├── 📋 requirements_order.txt       # Installation order guide
├── 📋 INSTALLATION_GUIDE.md        # Detailed install guide
├── 📋 ENSEMBLE_MODE_READY.md       # Ensemble setup guide
├── scripts/
│   └── 🪟 windows_path_fix.py      # Windows utilities
├── src/                            # Source code modules
│   ├── models/                     # AI model definitions
│   ├── utils/                      # Utility functions
│   └── ...
├── templates/                      # Web page templates
├── static/                         # CSS, JavaScript, images
├── configs/                        # Configuration files
├── models/                         # Model files (when available)
├── kaggle_outputs/                 # Alternative model location
└── outputs/                        # Results and logs
```

---

## 🔍 **Verification Commands Reference**

### **Check Overall Status:**
```bash
python verify_setup.py
```

### **Check Model Files Specifically:**
```bash
python verify_ensemble_mode.py
```

### **Test Basic Import:**
```bash
python -c "import torch, flask; print('Basic packages OK')"
```

### **Check Virtual Environment:**
```bash
# Windows:
C:\venv\xray-ai\Scripts\python.exe --version

# Check if packages installed:
C:\venv\xray-ai\Scripts\python.exe -c "import torch, torchxrayvision; print('AI packages OK')"
```

### **Diagnose Windows Issues:**
```bash
python scripts/windows_path_fix.py --diagnose
```

---

## 📊 **API Reference**

### **Status Endpoint:**
```bash
GET http://localhost:5000/api/status
```
**Returns**: Current application status, model availability, and capabilities

### **Upload Endpoint:**
```bash
POST http://localhost:5000/api/upload
Content-Type: multipart/form-data
Body: file=[chest_xray_image]
```
**Returns**: AI analysis results with confidence scores

### **Disease Info Endpoint:**
```bash
GET http://localhost:5000/api/disease-info
```
**Returns**: Information about all 14 detectable diseases

---

## 🎓 **For Developers**

### **Adding New Features:**
1. **Backend**: Modify `app_fixed.py` or add modules in `src/`
2. **Frontend**: Update templates in `templates/` and assets in `static/`
3. **Models**: Add model definitions in `src/models/`
4. **Tests**: Add tests in `tests/` directory

### **Running Tests:**
```bash
# Install test dependencies:
pip install pytest pytest-cov

# Run tests:
pytest tests/

# With coverage:
pytest --cov=src tests/
```

### **Development Mode:**
```bash
# Enable Flask debug mode:
export FLASK_DEBUG=1    # macOS/Linux
set FLASK_DEBUG=1       # Windows

python app_fixed.py
```

---

## 🚨 **Important Notes**

### **Medical Disclaimer:**
> ⚠️ **This software is for research and educational purposes only.**
> 
> **It is NOT intended for clinical diagnosis or treatment decisions.**
> 
> **Always consult qualified medical professionals for medical advice.**

### **Model Performance:**
- **Accuracy**: Validated on large medical datasets
- **Ensemble Approach**: Combines two specialized models
- **Optimal Thresholds**: Per-disease optimized prediction thresholds
- **CPU Optimized**: Works on any computer without GPU

### **Data Privacy:**
- **No Data Storage**: Uploaded images are processed and immediately deleted
- **Local Processing**: All AI analysis happens on your computer
- **No Internet Required**: After installation, works completely offline

---

## 🆘 **Getting Help**

### **Step 1: Run Diagnostics**
```bash
# Check overall status:
python verify_setup.py

# Check ensemble mode:
python verify_ensemble_mode.py

# Check Windows issues (if applicable):
python scripts/windows_path_fix.py --diagnose
```

### **Step 2: Check Common Solutions**
1. **Missing Dependencies**: Run `python install_xray_ai.py`
2. **Demo Mode**: Normal if no model files - web interface still works
3. **Import Errors**: Use virtual environment: `C:\venv\xray-ai\Scripts\python.exe app_fixed.py`
4. **Port Issues**: App will suggest alternatives or kill conflicting processes

### **Step 3: Review Log Files**
- `installation_log.txt` - Installation details
- `outputs/logs/` - Application logs

### **Step 4: Report Issues**
When reporting problems, include:
1. Operating system and Python version
2. Output from `python verify_setup.py`
3. Error messages from console
4. Installation method used

---

## 🏁 **Quick Start Summary**

**For New Users:**
```bash
git clone https://github.com/dara01x/x-ray.git
cd x-ray
python install_xray_ai.py
start_ensemble_mode.bat
# Open browser to http://localhost:5000
```

**For Developers:**
```bash
git clone https://github.com/dara01x/x-ray.git
cd x-ray
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python verify_setup.py
python app_fixed.py
```

---

## 📄 **License & Credits**

- **License**: MIT License - Free for educational and research use
- **Built with**: PyTorch, Flask, TorchXRayVision
- **Purpose**: Advancing medical AI research and education
- **Contribution**: Open source - contributions welcome!

---

**🩺 Ready to advance medical AI research with automated chest X-ray analysis!**

*For detailed installation help, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)*  
*For ensemble mode setup, see [ENSEMBLE_MODE_READY.md](ENSEMBLE_MODE_READY.md)*