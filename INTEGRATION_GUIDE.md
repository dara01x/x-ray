# Integration Guide for Your New Merged Model

## 🎯 Overview
This guide will help you integrate your new merged model from Kaggle into this repository and update the entire codebase around it.

## 📋 Step-by-Step Integration Process

### Step 1: Download Your Model from Kaggle

#### Option A: Using Kaggle API (Recommended)
1. **Setup Kaggle API credentials:**
   ```powershell
   # Create .kaggle directory in your home folder
   mkdir ~/.kaggle  # or C:\Users\{username}\.kaggle on Windows
   
   # Go to Kaggle.com → Your Account → Create New API Token
   # Download kaggle.json and place it in the .kaggle directory
   ```

2. **Run the download script:**
   ```powershell
   python download_new_model.py
   ```

#### Option B: Manual Download
1. Go to your Kaggle dataset/notebook
2. Download all model files (.pth, .pt, .ckpt, etc.)
3. Download any configuration or metadata files
4. Place them in `./downloads/` folder in this repository

### Step 2: Analyze Your Model

1. **Run the analysis:**
   ```powershell
   python download_new_model.py
   ```
   
2. **The script will tell you:**
   - What files were downloaded
   - Model architecture type (DenseNet, ResNet, etc.)
   - Output layer configuration
   - Required input size

### Step 3: Update Model Integration

1. **Edit `src/models/new_merged_model.py`:**
   - Replace the placeholder architecture with your actual model
   - Update the `load_from_checkpoint` method to match your checkpoint format
   - Ensure the forward pass matches your model's structure

2. **Example updates needed:**
   ```python
   # If your model is a DenseNet variant:
   self.backbone = torchvision.models.densenet121(pretrained=False)
   
   # If your model has custom preprocessing:
   def preprocess(self, x):
       # Add your preprocessing steps here
       return processed_x
   ```

### Step 4: Update Configuration

1. **Edit `configs/new_model_config.yaml`:**
   - Update `checkpoint_path` to point to your model file
   - Set correct `input_size` (e.g., [512, 512] if different from 224x224)
   - Update `num_classes` if different from 14
   - Set correct normalization values if your model uses different ones

2. **Update data paths:**
   - If you have new training data, update `new_data_dir`
   - Set correct `input_format` (DICOM, PNG, JPG, etc.)

### Step 5: Test Model Loading

```powershell
# Test if your model loads correctly
python -c "from src.models.new_merged_model import load_new_model; model = load_new_model('path/to/your/model.pth'); print('Model loaded successfully!')"
```

### Step 6: Run Inference with New Model

```powershell
# Single image inference
python scripts/inference_new.py --config configs/new_model_config.yaml --image path/to/test_image.png --model-type new

# Batch inference
python scripts/inference_new.py --config configs/new_model_config.yaml --image-dir path/to/images/ --model-type new
```

### Step 7: Update Repository Documentation

1. **Update README.md** to reflect new model capabilities
2. **Update performance benchmarks** after testing
3. **Document any new features** your merged model provides

## 🔧 Common Issues and Solutions

### Issue: Model Architecture Mismatch
**Solution:** Check the layer names in your checkpoint and update the model class accordingly.

### Issue: Input Size Mismatch  
**Solution:** Update `image_size` in config and ensure preprocessing matches your training setup.

### Issue: Normalization Differences
**Solution:** If your model was trained with different normalization, update the preprocessing pipeline.

### Issue: Class Count Mismatch
**Solution:** Update `num_classes` in config if your model predicts different number of diseases.

## 📊 What Information I Need About Your Model

To help you better integrate your model, please provide:

1. **Model Architecture:**
   - What models were merged? (e.g., "DenseNet121 + ResNet50")
   - What merging technique was used? (ensemble, feature fusion, etc.)

2. **Training Details:**
   - Input image size
   - Normalization values used
   - Number of output classes
   - Any special preprocessing steps

3. **Performance:**
   - Validation accuracy/AUC
   - Which diseases does it perform best on?
   - Any known limitations

4. **File Structure:**
   - How is the checkpoint saved? (full model, state_dict, etc.)
   - Are there separate config files?
   - Any additional files needed?

## 🚀 After Integration

Once your model is integrated:

1. **Run comprehensive evaluation:**
   ```powershell
   python scripts/evaluate.py --config configs/new_model_config.yaml --model-type new
   ```

2. **Benchmark performance:**
   ```powershell
   python scripts/benchmark.py --model-type new
   ```

3. **Compare with legacy model:**
   ```powershell
   python scripts/model_comparison.py
   ```

4. **Update the main README.md** with new performance metrics and capabilities.

## 📞 Need Help?

If you encounter issues:
1. Run `python download_new_model.py` to analyze your files
2. Check the console output for specific error messages
3. Verify all file paths in the configuration
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

The integration process is designed to be flexible and handle different model architectures. The key is updating the model class and configuration to match your specific setup.