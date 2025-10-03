# 🔬 Chest X-ray AI Web Application

A professional-grade web application for AI-powered chest X-ray disease classification using deep learning ensemble models.

![Status](https://img.shields.io/badge/status-production--ready-green)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Framework](https://img.shields.io/badge/framework-Flask-lightgrey)
![AI](https://img.shields.io/badge/AI-PyTorch-orange)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Modern web browser

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd x-ray
   ```

2. **Set up the environment:**
   ```bash
   python setup_webapp.py
   ```

3. **Start the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

That's it! The application will automatically handle:
- ✅ Virtual environment setup
- ✅ Dependency installation  
- ✅ Model configuration
- ✅ Threshold optimization
- ✅ Demo data creation

## 🎯 Features

### 🔬 **AI-Powered Analysis**
- **14 Disease Detection**: Comprehensive pathology classification
- **Ensemble Models**: Multiple neural networks for improved accuracy
- **Optimized Thresholds**: Per-disease optimal classification boundaries
- **Real-time Processing**: Results in seconds

### 🎨 **Modern Interface**
- **Responsive Design**: Works on all devices
- **Drag & Drop Upload**: Intuitive file handling
- **Interactive Results**: Detailed visualizations and charts
- **Clean Navigation**: Streamlined user experience

### 🛡️ **Medical Grade**
- **Privacy First**: No permanent data storage
- **Secure Processing**: Local analysis only
- **Medical Disclaimers**: Clear safety guidelines
- **Educational Focus**: Research and learning oriented

## 📋 Detectable Conditions

The AI can analyze chest X-rays for these 14 conditions:

| Condition | Description | Risk Level |
|-----------|-------------|------------|
| **Atelectasis** | Lung collapse | Moderate |
| **Cardiomegaly** | Enlarged heart | High |
| **Effusion** | Fluid around lungs | Moderate |
| **Infiltration** | Abnormal substances | Moderate |
| **Mass** | Tissue growths | High |
| **Nodule** | Small lung nodules | Moderate |
| **Pneumonia** | Lung infection | High |
| **Pneumothorax** | Collapsed lung | High |
| **Consolidation** | Fluid-filled tissue | Moderate |
| **Edema** | Fluid accumulation | High |
| **Emphysema** | Damaged air sacs | High |
| **Fibrosis** | Lung scarring | High |
| **Pleural Thickening** | Thickened lining | Moderate |
| **Hernia** | Organ displacement | Moderate |

## 📊 Technical Specifications

### **AI Models**
- **Architecture**: DenseNet121 with TorchXRayVision backbone
- **Training Data**: Large-scale chest X-ray datasets
- **Input Format**: 224×224 grayscale images
- **Supported Files**: PNG, JPG, JPEG, DCM (up to 16MB)

### **Web Technology**
- **Backend**: Flask with PyTorch integration
- **Frontend**: Modern JavaScript with Chart.js
- **Styling**: CSS Grid and Flexbox for responsive design
- **Icons**: Font Awesome for consistent UI

## 🔧 Advanced Usage

### **Model Modes**
The application automatically detects and uses available models:

1. **Ensemble Mode** (Best Performance)
   - Requires: `kaggle_outputs/model.pth.tar` + `outputs/models/best_model.pth`
   - Uses: Average predictions from multiple models

2. **Single Model Mode** (Good Performance)  
   - Requires: `outputs/models/best_model.pth`
   - Uses: Individual model predictions

3. **Demo Mode** (Testing)
   - No models required
   - Generates: Simulated results for demonstration

### **Configuration**
Modify `configs/config.yaml` to customize:
- Image preprocessing parameters
- Model architecture settings
- Training hyperparameters
- Output directories

## 📱 Usage Guide

### **1. Upload Image**
- Drag and drop X-ray image or click to browse
- Supported formats: PNG, JPG, JPEG, DCM
- Maximum size: 16MB
- Image preview shows before analysis

### **2. Analyze**
- Click "Analyze X-ray" button
- Watch real-time progress indicator
- Analysis completes in 3-10 seconds

### **3. Review Results**
- **Summary**: Key findings overview
- **Details**: Individual disease probabilities
- **Charts**: Visual representation of results
- **Export**: Download reports and share results

### **4. Additional Features**
- Download JSON reports
- Share results via link
- Print-friendly format available
- View disease information by clicking conditions

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for **educational and research purposes only**.

- ❌ **NOT for medical diagnosis**
- ❌ **NOT a substitute for professional medical advice**
- ❌ **NOT for clinical decision-making**

✅ **Always consult qualified healthcare professionals for medical concerns**

## 🛠️ Development

### **Project Structure**
```
x-ray/
├── app.py                    # Main Flask application
├── setup_webapp.py           # Automated setup script
├── templates/index.html      # Web interface
├── static/                   # CSS, JS, assets
├── src/                     # Source code modules
├── configs/                 # Configuration files
├── outputs/                 # Generated results
└── requirements.txt         # Dependencies
```

### **Key Components**
- **`app.py`**: Flask server with API endpoints
- **`src/models/`**: AI model definitions and ensemble logic
- **`src/data/`**: Data loading and preprocessing
- **`templates/index.html`**: Single-page application interface
- **`static/js/app.js`**: Frontend application logic

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** comprehensive tests
4. **Update** documentation
5. **Submit** a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Healthcare Integration

For healthcare organizations interested in integration:

- **HIPAA Compliance**: Additional security measures available
- **Clinical Integration**: API endpoints for EMR systems
- **Custom Training**: Model fine-tuning for specific populations
- **Professional Support**: Enterprise support options

Contact the development team for healthcare-specific requirements.

## 📞 Support

- **Documentation**: Comprehensive guides in `/docs`
- **Issues**: GitHub Issues for bug reports
- **Features**: Feature requests via GitHub Discussions
- **Security**: Security issues via private disclosure

---

**🎓 Educational Use**: Perfect for medical education, research demonstrations, and AI/ML learning.

**⚡ Production Ready**: Scalable architecture suitable for real-world deployment.

**🔬 Research Grade**: Built with medical imaging best practices and safety standards.

---

*Made with ❤️ for advancing medical AI education and research*