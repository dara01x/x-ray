#!/usr/bin/env python3
"""
Quick test script - run this after downloading files
"""

import os
from pathlib import Path

def quick_check():
    print("Quick File Check")
    print("=" * 30)
    
    kaggle_dir = Path("./kaggle_outputs")
    files = [
        'best_model_all_out_v1.pth',
        'optimal_thresholds_all_out_v1.json',
        'model.pth.tar', 
        'optimal_thresholds_ensemble_final_v1.json'
    ]
    
    found = 0
    for f in files:
        path = kaggle_dir / f
        if path.exists():
            size = path.stat().st_size / (1024*1024)
            print(f"✓ {f} ({size:.1f} MB)")
            found += 1
        else:
            print(f"✗ {f} - missing")
    
    print(f"\nStatus: {found}/{len(files)} files found")
    
    if found == len(files):
        print("\n🎉 All essential files present!")
        print("Run: python scripts/ensemble_inference.py --image your_image.png --kaggle-outputs ./kaggle_outputs")
    else:
        print(f"\n📥 Download the missing files from your Kaggle notebook")

if __name__ == "__main__":
    quick_check()