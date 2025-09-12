# Radiology AI - Chest X-ray Disease Classification

A comprehensive deep learning system for multi-label disease detection in chest X-ray images. This project implements a state-of-the-art computer vision pipeline for automated diagnosis of 14 different thoracic diseases from chest radiographs.

## 🏥 Medical Applications

This system can assist radiologists in detecting:
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

## 🚀 Features

### Advanced Deep Learning Architecture
- **TorchXRayVision DenseNet121** backbone with medical imaging pretraining
- **Custom classification head** with batch normalization and dropout
- **Multi-label classification** supporting simultaneous disease detection
- **Mixed precision training** for efficient GPU utilization

### Sophisticated Data Pipeline
- **Patient-level data splitting** to prevent data leakage
- **Advanced preprocessing** with CLAHE and proper scaling
- **Robust augmentation** using Albumentations library
- **Custom data loaders** with error handling

### Professional Training Framework
- **Focal Loss** for handling class imbalance
- **Discriminative learning rates** for backbone and head
- **OneCycleLR scheduler** for optimal convergence
- **Early stopping** with configurable patience
- **Comprehensive checkpointing** and resumption

### Extensive Evaluation Tools
- **Per-class AUC calculation** and tracking
- **Optimal threshold finding** for each disease
- **Confusion matrix visualization** and analysis
- **Classification reports** with detailed metrics
- **Explainable AI** with Grad-CAM visualizations

## 📁 Project Structure

```
radiology-ai/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Training utilities and loss functions
│   ├── evaluation/               # Evaluation and metrics
│   └── utils/                    # Utility functions
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference script
├── tests/                        # Test suite
├── configs/                      # Configuration files
├── notebooks/                    # Jupyter notebooks
├── data/                         # Data directory (not included)
└── outputs/                      # Output directory
    ├── models/                   # Saved models
    ├── results/                  # Evaluation results
    ├── logs/                     # Training logs
    └── checkpoints/              # Model checkpoints
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (recommended: 3.9-3.11)
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ storage for project + datasets

### Quick Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dara01x/x-ray-ai.git
   cd x-ray-ai
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

5. **Verify installation**
   ```bash
   python -m pytest tests/ -v
   ```

### Alternative: Install from PyPI (if published)
```bash
pip install radiology-ai
```

## 📊 Dataset Setup

### Option 1: Quick Start with Demo Data (Recommended for Testing)

1. **Create synthetic demo data**
   ```bash
   python create_demo_data.py
   ```
   This creates 200 synthetic chest X-ray images for testing.

2. **Run a quick test**
   ```bash
   python simple_accuracy_test.py
   ```

### Option 2: Full NIH Chest X-ray Dataset (Production)

1. **Download the NIH Chest X-ray Dataset**
   - Visit: [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
   - Download `Data_Entry_2017.csv` (~45MB)
   - Download image archives: `images_001.tar.gz` through `images_012.tar.gz` (~45GB total)

2. **Organize your data directory**
   ```
   data/
   ├── Data_Entry_2017.csv
   ├── images_001/
   │   └── images/
   │       ├── 00000001_000.png
   │       ├── 00000001_001.png
   │       └── ...
   ├── images_002/
   │   └── images/
   └── ... (up to images_012)
   ```

3. **Extract the archives**
   ```bash
   # Extract all image archives to data/ directory
   tar -xzf images_001.tar.gz -C data/
   tar -xzf images_002.tar.gz -C data/
   # ... repeat for all archives
   ```

4. **Verify data setup**
   ```bash
   python scripts/verify_data.py
   ```

## 🚀 Quick Start

### 1. Basic Setup and Testing
```bash
# After installation, create demo data for quick testing
python create_demo_data.py

# Test the model (uses random weights)
python simple_accuracy_test.py

# Run the test suite
python -m pytest tests/ -v
```

### 2. Training a Model
```bash
# Train with demo data (quick, ~5-10 minutes)
python scripts/train.py --config configs/config.yaml

# Train with full NIH dataset (6-12 hours)
# First ensure you have the NIH dataset in data/
python scripts/train.py --config configs/config.yaml
```

### 3. Evaluating a Model
```bash
# Evaluate a trained model with threshold optimization
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --find-thresholds

# Basic evaluation without threshold optimization
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth
```

### 4. Running Inference
```bash
# Single image inference
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image path/to/chest_xray.png

# Batch inference on a directory
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image-dir path/to/xray_images/

# Get probability scores and disease predictions
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image path/to/chest_xray.png \
    --output-csv results.csv
```

## ⚙️ Configuration

The system is highly configurable through YAML files. Key configuration sections:

### Model Configuration
```yaml
model:
  backbone: "densenet121"
  num_classes: 14
  dropout_rate: 0.6
  pretrained_weights: "densenet121-res224-all"
```

### Training Configuration
```yaml
training:
  batch_size: 64
  num_epochs: 100
  backbone_lr: 1e-5
  head_lr: 1e-3
  loss:
    type: "focal"
    alpha: 0.25
    gamma: 2.0
```

### Data Configuration
```yaml
data:
  data_dir: "./data"
  csv_file: "Data_Entry_2017.csv"
  valid_views: ["PA", "AP"]
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_models.py -v
```

## 📈 Performance Monitoring

### Training Metrics
- Training and validation loss curves
- Per-class AUC tracking
- Learning rate scheduling visualization
- GPU memory usage monitoring

### Evaluation Metrics
- Macro and micro-averaged AUC
- Per-class precision, recall, and F1-scores
- Confusion matrices for all diseases
- Optimal threshold analysis

## 🔬 Explainable AI

The system includes Grad-CAM visualization for model interpretability:

```python
from src.evaluation import create_evaluator
from pytorch_grad_cam import GradCAM

# Generate attention maps for specific diseases
evaluator.generate_gradcam_visualization(model, image, disease_class)
```

## 🚨 Important Notes

### Medical Disclaimer
⚠️ **This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for medical advice.**

### Data Privacy
- Ensure compliance with HIPAA and other relevant regulations
- Anonymize patient data before processing
- Implement appropriate security measures for data handling

### Performance Considerations
- GPU with 8GB+ VRAM recommended for training
- Training time: ~6-12 hours on modern GPU
- Inference time: ~50ms per image on GPU

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{radiology_ai_2025,
  title={X-Ray AI: Multi-label Chest X-ray Disease Classification},
  author={Dara Mustafa},
  year={2025},
  url={https://github.com/dara01x/x-ray-ai}
}
```

## 🔗 Related Work

- [TorchXRayVision](https://github.com/mlmed/torchxrayvision)
- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXNet Paper](https://arxiv.org/abs/1711.05225)

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Join our [Discord community](https://discord.gg/radiologyai)

## 🙏 Acknowledgments

- NIH Clinical Center for providing the chest X-ray dataset
- TorchXRayVision team for the medical imaging models
- The open-source community for the amazing tools and libraries

---

**Built with ❤️ for the medical AI community**
