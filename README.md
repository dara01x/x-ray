# 🩺 X-Ray AI Web Application

**AI-powered chest X-ray analysis with 14 disease detection**

A user-friendly web application that uses advanced deep learning models to analyze chest X-ray images and detect 14 different thoracic diseases including Pneumonia, Cardiomegaly, and more.

## 🚀 Quick Start Guide

### Step 1: Download & Setup
```bash
# Clone the repository
git clone https://github.com/dara01x/x-ray.git
cd x-ray

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Web Application
```bash
# Start the web app
python app.py
```

### Step 3: Access the Application
1. Open your web browser
2. Go to `http://localhost:5000`
3. Navigate to the "Analyze" tab
4. Upload a chest X-ray image (PNG, JPG, JPEG)
5. View AI analysis results with confidence scores

## � What It Detects

The AI can identify these 14 conditions:
- **Atelectasis** - Lung collapse
- **Cardiomegaly** - Enlarged heart  
- **Effusion** - Fluid accumulation
- **Infiltration** - Inflammatory changes
- **Mass** - Tumor or lesion
- **Nodule** - Small round opacity
- **Pneumonia** - Lung infection
- **Pneumothorax** - Collapsed lung
- **Consolidation** - Lung solidification
- **Edema** - Fluid in lungs
- **Emphysema** - Lung tissue damage
- **Fibrosis** - Lung scarring
- **Pleural Thickening** - Pleural abnormalities
- **Hernia** - Diaphragmatic hernia

## � System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for initial setup only

## 🖥️ How to Use

1. **Upload Image**: Drag and drop or click to select a chest X-ray
2. **Wait for Analysis**: AI processes the image (takes 5-15 seconds)
3. **View Results**: See detection confidence for each condition
4. **Interactive Charts**: Hover over results for detailed information

## 🔧 Troubleshooting

**App won't start?**
- Make sure Python 3.8+ is installed
- Check that virtual environment is activated
- Try: `pip install --upgrade pip`

**Upload not working?**
- Supported formats: PNG, JPG, JPEG
- Maximum file size: 10MB
- Try a different browser

**Slow analysis?**
- First analysis takes longer (model loading)
- Subsequent analyses are faster
- Close other applications to free up RAM

## ⚠️ Medical Disclaimer

**This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for medical advice.**

## 📞 Support

Having issues? Try these solutions:
1. Check the troubleshooting section above
2. Make sure Python 3.8+ is installed
3. Verify virtual environment is activated
4. For technical issues, check the project repository

## 📄 License

MIT License - Free for educational and research use.

---
**🩺 Built for advancing medical AI research**