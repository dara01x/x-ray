#!/usr/bin/env python3
"""
Kaggle Ensemble Model Setup Script
Downloads and organizes the ensemble model files from your Kaggle notebook.
"""

import os
import shutil
import json
from pathlib import Path
import argparse
import zipfile
import requests
from typing import Dict, List

class KaggleEnsembleSetup:
    """Setup ensemble model from Kaggle outputs."""
    
    def __init__(self, kaggle_outputs_dir: str = "./kaggle_outputs"):
        self.kaggle_outputs_dir = Path(kaggle_outputs_dir)
        self.kaggle_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected files from the notebook
        self.expected_files = {
            'champion_checkpoint': 'best_model_all_out_v1.pth',
            'champion_thresholds': 'optimal_thresholds_all_out_v1.json',
            'arnoweng_checkpoint': 'model.pth.tar',
            'ensemble_thresholds': 'optimal_thresholds_ensemble_final_v1.json',
            'ensemble_metrics': 'final_metrics_ensemble_final_v1.json',
            'classification_report': 'classification_report_ensemble_final_v1.txt'
        }
        
    def check_existing_files(self) -> Dict[str, bool]:
        """Check which files already exist."""
        status = {}
        
        print(f"\n🔍 Checking for existing files in {self.kaggle_outputs_dir}")
        print("=" * 60)
        
        for key, filename in self.expected_files.items():
            file_path = self.kaggle_outputs_dir / filename
            exists = file_path.exists()
            status[key] = exists
            
            status_icon = "✅" if exists else "❌"
            size_info = ""
            if exists:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                size_info = f" ({size_mb:.1f} MB)"
            
            print(f"{status_icon} {key:<20}: {filename}{size_info}")
            
        return status
        
    def manual_download_instructions(self):
        """Print instructions for manual download."""
        print(f"\n📋 MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("1. Go to your Kaggle notebook: ab-ensemble.ipynb")
        print("2. Make sure it has been committed (saved)")
        print("3. Download the following files from the notebook output:")
        print()
        
        for key, filename in self.expected_files.items():
            description = self._get_file_description(key)
            print(f"   📁 {filename}")
            print(f"      {description}")
            print()
            
        print(f"4. Place all files in: {self.kaggle_outputs_dir.absolute()}")
        print("5. Run this script again to verify the setup")
        
    def _get_file_description(self, key: str) -> str:
        """Get description for each file type."""
        descriptions = {
            'champion_checkpoint': "Your trained champion model weights",
            'champion_thresholds': "Optimal thresholds for your champion model",
            'arnoweng_checkpoint': "Arnoweng's CheXNet model weights", 
            'ensemble_thresholds': "Optimal thresholds for the ensemble model",
            'ensemble_metrics': "Complete evaluation metrics for ensemble",
            'classification_report': "Detailed classification report"
        }
        return descriptions.get(key, "Model file")
        
    def setup_kaggle_api(self):
        """Guide user through Kaggle API setup."""
        print(f"\n🔧 KAGGLE API SETUP")
        print("=" * 60)
        print("To download files automatically, you need to setup Kaggle API:")
        print()
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'") 
        print("3. Download the kaggle.json file")
        print("4. Place it in your home directory:")
        
        if os.name == 'nt':  # Windows
            kaggle_dir = Path.home() / '.kaggle'
            print(f"   Windows: {kaggle_dir}")
        else:  # Unix/Linux/Mac
            kaggle_dir = Path.home() / '.kaggle'
            print(f"   Unix/Linux/Mac: {kaggle_dir}")
            
        print()
        print("5. Install Kaggle API: pip install kaggle")
        print("6. Test with: kaggle competitions list")
        
    def download_with_kaggle_api(self, username: str, notebook_name: str):
        """Attempt to download using Kaggle API."""
        try:
            import kaggle
            
            print(f"\n📥 Attempting to download from Kaggle...")
            print(f"User: {username}")
            print(f"Notebook: {notebook_name}")
            
            # This is a placeholder - Kaggle API doesn't directly support
            # downloading notebook outputs. User will need to manually download.
            print("❌ Kaggle API doesn't support direct notebook output download.")
            print("Please use manual download method.")
            
            return False
            
        except ImportError:
            print("❌ Kaggle API not installed. Run: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Kaggle API error: {e}")
            return False
            
    def validate_model_files(self) -> bool:
        """Validate that model files are correct format."""
        print(f"\n🔍 Validating model files...")
        
        validation_results = {}
        
        # Check champion checkpoint
        champion_path = self.kaggle_outputs_dir / self.expected_files['champion_checkpoint']
        if champion_path.exists():
            try:
                # Try to load as PyTorch checkpoint (without actually loading the model)
                import torch
                checkpoint = torch.load(champion_path, map_location='cpu', weights_only=False)
                
                has_model_state = 'model_state_dict' in checkpoint
                has_epoch = 'epoch' in checkpoint
                
                validation_results['champion_checkpoint'] = {
                    'valid': True,
                    'has_model_state': has_model_state,
                    'has_epoch': has_epoch,
                    'epoch': checkpoint.get('epoch', 'Unknown')
                }
                
                print(f"✅ Champion checkpoint valid (epoch: {checkpoint.get('epoch', 'N/A')})")
                
            except Exception as e:
                validation_results['champion_checkpoint'] = {'valid': False, 'error': str(e)}
                print(f"❌ Champion checkpoint invalid: {e}")
        
        # Check JSON files
        for key in ['champion_thresholds', 'ensemble_thresholds', 'ensemble_metrics']:
            json_path = self.kaggle_outputs_dir / self.expected_files[key]
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    validation_results[key] = {'valid': True, 'keys': len(data)}
                    print(f"✅ {key} valid ({len(data)} entries)")
                    
                except Exception as e:
                    validation_results[key] = {'valid': False, 'error': str(e)}
                    print(f"❌ {key} invalid: {e}")
        
        return all(result.get('valid', False) for result in validation_results.values())
        
    def create_integration_guide(self):
        """Create a guide for integrating the ensemble model."""
        guide_path = self.kaggle_outputs_dir / "INTEGRATION_GUIDE.md"
        
        guide_content = f"""# Ensemble Model Integration Guide

## Files Downloaded
The following files have been set up from your Kaggle notebook:

"""
        
        for key, filename in self.expected_files.items():
            file_path = self.kaggle_outputs_dir / filename
            status = "✅ Available" if file_path.exists() else "❌ Missing"
            description = self._get_file_description(key)
            
            guide_content += f"- **{filename}**: {description} - {status}\n"
            
        guide_content += f"""

## Next Steps

### 1. Test the Ensemble Model
```bash
python scripts/ensemble_inference.py \\
    --image path/to/chest_xray.png \\
    --kaggle-outputs {self.kaggle_outputs_dir} \\
    --visualize \\
    --save-results
```

### 2. Run Evaluation
```bash
python scripts/evaluate_ensemble.py \\
    --config configs/ensemble_config.yaml \\
    --kaggle-outputs {self.kaggle_outputs_dir}
```

### 3. Compare with Legacy Model
```bash
python scripts/model_comparison.py \\
    --ensemble-results outputs/results/ensemble/ \\
    --legacy-results outputs/results/legacy/
```

## Model Information
- **Architecture**: Ensemble of Champion DenseNet121 + Arnoweng CheXNet
- **Combination Method**: Simple averaging of predictions
- **Optimal Thresholds**: Per-class thresholds optimized for F1-score
- **Expected Performance**: Improved over individual models

## File Structure
```
{self.kaggle_outputs_dir}/
├── {self.expected_files['champion_checkpoint']}     # Champion model weights
├── {self.expected_files['champion_thresholds']}    # Champion thresholds
├── {self.expected_files['arnoweng_checkpoint']}    # Arnoweng model weights
├── {self.expected_files['ensemble_thresholds']}   # Ensemble thresholds
├── {self.expected_files['ensemble_metrics']}      # Performance metrics
└── {self.expected_files['classification_report']} # Detailed report
```

## Dependencies
Make sure you have installed:
```bash
pip install torch torchvision torchxrayvision albumentations scikit-learn opencv-python matplotlib pandas numpy
```

## Troubleshooting
- Ensure all files are completely downloaded
- Check file sizes match expected values
- Verify PyTorch can load the checkpoint files
- Make sure you have enough GPU memory (8GB+ recommended)
"""
        
        with open(guide_path, 'w') as f:
            f.write(guide_content)
            
        print(f"📖 Integration guide created: {guide_path}")
        
    def setup_directory_structure(self):
        """Create the expected directory structure."""
        directories = [
            "outputs/models/ensemble",
            "outputs/checkpoints/ensemble",
            "outputs/results/ensemble",
            "outputs/logs/ensemble"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        print("✅ Directory structure created")
        
    def run_setup(self, kaggle_username: str = None, notebook_name: str = None):
        """Run the complete setup process."""
        print("🚀 Ensemble Model Setup")
        print("=" * 60)
        
        # Check existing files
        status = self.check_existing_files()
        missing_files = [key for key, exists in status.items() if not exists]
        
        if not missing_files:
            print("\n✅ All files found! Proceeding with validation...")
            if self.validate_model_files():
                print("\n🎉 Setup complete! All files are valid.")
                self.create_integration_guide()
                self.setup_directory_structure()
                return True
            else:
                print("\n❌ Some files failed validation. Please re-download.")
                return False
                
        print(f"\n⚠️ Missing {len(missing_files)} files:")
        for key in missing_files:
            print(f"  - {self.expected_files[key]}")
            
        print(f"\n📋 Please download the missing files manually:")
        self.manual_download_instructions()
        
        if kaggle_username and notebook_name:
            self.download_with_kaggle_api(kaggle_username, notebook_name)
            
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Ensemble Model from Kaggle")
    parser.add_argument("--kaggle-outputs", type=str, default="./kaggle_outputs",
                       help="Directory for Kaggle output files")
    parser.add_argument("--kaggle-username", type=str,
                       help="Your Kaggle username (for API download)")
    parser.add_argument("--notebook-name", type=str,
                       help="Kaggle notebook name (for API download)")
    parser.add_argument("--setup-api", action="store_true",
                       help="Show Kaggle API setup instructions")
    
    args = parser.parse_args()
    
    setup = KaggleEnsembleSetup(args.kaggle_outputs)
    
    if args.setup_api:
        setup.setup_kaggle_api()
        return
        
    success = setup.run_setup(args.kaggle_username, args.notebook_name)
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"1. Test inference: python scripts/ensemble_inference.py --image test_image.png")
        print(f"2. Read integration guide: {args.kaggle_outputs}/INTEGRATION_GUIDE.md")
    else:
        print(f"\n🔄 Run this script again after downloading the files")

if __name__ == "__main__":
    main()