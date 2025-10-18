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

## 🎯 Two Modes

- **🤖 Full AI Mode**: Real disease detection (needs model files)
- **🎮 Demo Mode**: Interface testing (works without model files)

## 💡 Enable Full AI Features

To get real AI analysis, place these files in the `models/` folder:
- `best_model_all_out_v1.pth` (Champion model)
- `model.pth.tar` (Arnoweng model)

Without these files, the app works in demo mode for interface testing.

## ✅ Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Internet connection (for setup only)

## 🔧 Troubleshooting

**Problem**: App shows "Demo Mode"  
**Solution**: This is normal! Demo mode lets you test the interface. Place model files in `models/` folder for real AI.

**Problem**: Import errors or "DLL load failed"  
**Solution**: Run `python setup.py` again or use a virtual environment.