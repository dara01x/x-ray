# X-Ray AI - Chest X-ray Disease Classification

A professional-grade deep learning system for multi-label disease detection in chest X-ray images. This system implements state-of-the-art computer vision techniques for automated diagnosis of 14 different thoracic diseases from chest radiographs.

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

### Professional Data Pipeline
- **Patient-level data splitting** to prevent data leakage
- **Advanced preprocessing** with CLAHE and proper scaling
- **Robust augmentation** using Albumentations library
- **Custom data loaders** with error handling

### Production Training Framework
- **Focal Loss** for handling class imbalance
- **Discriminative learning rates** for backbone and head
- **OneCycleLR scheduler** for optimal convergence
- **Early stopping** with configurable patience
- **Comprehensive checkpointing** and resumption

### Comprehensive Evaluation Tools
- **Per-class AUC calculation** and tracking
- **Optimal threshold finding** for each disease
- **Confusion matrix visualization** and analysis
- **Classification reports** with detailed metrics
- **Explainable AI** with Grad-CAM visualizations

## 📁 Project Structure

```
x-ray/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Training utilities and loss functions
│   ├── evaluation/               # Evaluation and metrics
│   └── utils/                    # Utility functions
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py              # Evaluation script
│   ├── inference.py             # Inference script
│   └── benchmark.py             # Performance benchmarking
├── tests/                        # Test suite
├── configs/                      # Configuration files
├── notebooks/                    # Jupyter notebooks
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

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dara01x/x-ray.git
   cd x-ray
   ```

2. **Create virtual environment**
   ```powershell
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
   pip install -e .
   ```

4. **Verify installation**
   ```bash
   python -m pytest tests/ -v
   ```

## 📊 Dataset Setup

### Option 1: Quick Demo (Recommended for Testing)

Create synthetic demo data for immediate testing:
```bash
python -c "from src.data import create_demo_data; create_demo_data(num_samples=200)"
```

### Option 2: NIH Chest X-ray Dataset (Production)

1. **Download the NIH dataset**
   - Download `Data_Entry_2017.csv` (~45MB)
   - Download image archives: `images_001.tar.gz` through `images_012.tar.gz` (~45GB total)

2. **Organize data directory**
   ```
   data/
   ├── Data_Entry_2017.csv
   ├── images_001/
   │   └── images/
   └── ... (images_002 through images_012)
   ```

3. **Update configuration**
   Edit `configs/config.yaml` to point to your data directory.

## 🚀 Usage

### Training
```bash
# Quick training on demo data
python scripts/train.py --config configs/config.yaml

# Full training (requires NIH dataset)
python scripts/train.py --config configs/config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --find-thresholds
```

### Inference
```bash
# Single image
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image path/to/chest_xray.png

# Batch inference
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image-dir path/to/images/
```

### Performance Benchmarking
```bash
python scripts/benchmark.py
```

## ⚙️ Configuration

The system uses YAML configuration files for easy customization. Key sections:

```yaml
model:
  backbone: "densenet121"
  num_classes: 14
  dropout_rate: 0.6

training:
  batch_size: 64
  num_epochs: 100
  backbone_lr: 1e-5
  head_lr: 1e-3

data:
  data_dir: "./data"
  csv_file: "Data_Entry_2017.csv"
```

## 📈 Performance

### Current Benchmarks
- **Model Creation**: 323ms
- **Inference Speed**: 114ms per image (8.8 FPS)
- **Model Size**: 7.48M parameters (~28.5MB)

### Expected Performance (Production)
- **Mean AUC**: 83.4% (clinical-grade)
- **Best Disease Detection**: Pneumothorax (88% AUC)
- **Training Time**: 6-12 hours on modern GPU

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific tests
python -m pytest tests/test_models.py -v
```

## 🔬 Model Architecture

- **Backbone**: TorchXRayVision DenseNet121 (medical pretrained)
- **Input**: 224x224 grayscale images
- **Output**: 14-class multi-label predictions
- **Loss Function**: Focal Loss (α=0.25, γ=2.0)
- **Optimization**: Discriminative learning rates with OneCycleLR

## 🚨 Important Notes

### Medical Disclaimer
⚠️ **This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for medical advice.**

### Performance Requirements
- **GPU**: 8GB+ VRAM recommended for training
- **CPU**: Works but significantly slower
- **Memory**: 16GB+ RAM recommended for large datasets

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIH Clinical Center for providing the chest X-ray dataset
- TorchXRayVision team for the medical imaging models
- The open-source community for the amazing tools and libraries

---

**Built with ❤️ for advancing medical AI**