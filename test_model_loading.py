#!/usr/bin/env python3
"""
Test the ensemble model loading
"""

import sys
import os
sys.path.append('.')

try:
    from src.models.ensemble_model import load_ensemble_model
    
    print("🚀 Testing Ensemble Model Loading...")
    print("=" * 50)
    
    # Load the ensemble model
    ensemble = load_ensemble_model(
        champion_checkpoint='./kaggle_outputs/best_model_all_out_v1.pth',
        arnoweng_checkpoint='./kaggle_outputs/model.pth.tar',
        ensemble_thresholds='./kaggle_outputs/optimal_thresholds_ensemble_final_v1.json'
    )
    
    print("✅ Ensemble model loaded successfully!")
    
    # Get model info
    info = ensemble.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"  Device: {info['device']}")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Ensemble type: {info['ensemble_type']}")
    print(f"  Disease labels: {len(info['disease_labels'])}")
    print(f"  Has ensemble thresholds: {info['has_ensemble_thresholds']}")
    
    print(f"\n🎯 Your ensemble model is ready to use!")
    print(f"Run: python scripts/ensemble_inference.py --image your_chest_xray.png --kaggle-outputs ./kaggle_outputs --visualize")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nThis might be due to:")
    print(f"1. Missing PyTorch dependencies")
    print(f"2. Corrupted model files")
    print(f"3. Missing required packages")