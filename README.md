# 🩺 X-Ray AI - Simple Setup

**AI-powered chest X-ray analysis with 14 disease detection**

Analyze chest X-rays for 14 different diseases including Pneumonia, Cardiomegaly, Atelectasis, and more.

## 🚀 Quick Start

```bash
# 1. Get the code
git clone https://github.com/dara01x/x-ray.git
cd x-ray

# 2. Install dependencies
python setup.py

# 3. Start the app
python app.py
```

## 📱 How to Use

1. **Open** http://localhost:5000 in your browser
2. **Upload** a chest X-ray image (PNG, JPG, JPEG, DCM)  
3. **Get** instant AI analysis with disease predictions!

## 🎯 Ready to Use

- **🤖 Full AI Mode**: Real disease detection with trained models included!
- **✨ No Setup Required**: Model files are included in the repository

## 💡 AI Features Included

The repository includes pre-trained models:
- `best_model_all_out_v1.pth` (Champion model) ✅ Included
- `model.pth.tar` (Arnoweng model) ✅ Included
- Optimal thresholds and configurations ✅ Included

**Ready for immediate AI analysis!**

## ✅ Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection (for setup only)

## 🔧 Troubleshooting

**Problem**: App shows "Demo Mode"  
**Solution**: Make sure you cloned the full repository with model files. If models are missing, re-clone the repository.

**Problem**: Import errors or "DLL load failed"  
**Solution**: Run `python setup.py` again or use a virtual environment.