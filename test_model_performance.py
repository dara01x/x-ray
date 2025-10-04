#!/usr/bin/env python3
"""
Test script to verify model performance and threshold usage
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.ensemble_model import load_ensemble_model

def test_model_performance():
    """Test the ensemble model performance and threshold usage."""
    
    print("🔬 Testing X-Ray AI Model Performance")
    print("=" * 50)
    
    # Model paths
    champion_checkpoint = 'models/best_model_all_out_v1.pth'
    arnoweng_checkpoint = 'models/model.pth.tar'
    thresholds_json = 'models/optimal_thresholds_ensemble_final.json'
    thresholds_csv = 'models/optimal_thresholds_ensemble_final.csv'
    
    # Check files exist
    print("📁 Checking model files:")
    print(f"   Champion model: {champion_checkpoint} - {'✅' if os.path.exists(champion_checkpoint) else '❌'}")
    print(f"   Arnoweng model: {arnoweng_checkpoint} - {'✅' if os.path.exists(arnoweng_checkpoint) else '❌'}")
    print(f"   Thresholds JSON: {thresholds_json} - {'✅' if os.path.exists(thresholds_json) else '❌'}")
    print(f"   Thresholds CSV: {thresholds_csv} - {'✅' if os.path.exists(thresholds_csv) else '❌'}")
    
    if not os.path.exists(champion_checkpoint) or not os.path.exists(arnoweng_checkpoint):
        print("❌ Required model files not found!")
        return
    
    # Load thresholds from CSV and convert to JSON if needed
    if not os.path.exists(thresholds_json) and os.path.exists(thresholds_csv):
        print("📄 Converting CSV thresholds to JSON...")
        import pandas as pd
        df = pd.read_csv(thresholds_csv)
        thresholds = {}
        for _, row in df.iterrows():
            thresholds[row['Disease']] = float(row['Optimal_Threshold'])
        
        with open(thresholds_json, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"✅ Created {thresholds_json}")
    
    # Load optimal thresholds
    print("\n🎯 Loading optimal thresholds:")
    with open(thresholds_json, 'r') as f:
        thresholds = json.load(f)
    
    for disease, threshold in thresholds.items():
        print(f"   {disease}: {threshold:.3f}")
    
    # Load ensemble model
    print("\n🤖 Loading ensemble model...")
    try:
        ensemble_model = load_ensemble_model(
            champion_checkpoint=champion_checkpoint,
            arnoweng_checkpoint=arnoweng_checkpoint,
            ensemble_thresholds=thresholds_json
        )
        print("✅ Ensemble model loaded successfully!")
        print(f"📊 Model info: {ensemble_model.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load ensemble model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with a sample image
    print("\n🖼️ Testing with sample images...")
    test_images = []
    
    # Check for demo images
    if os.path.exists('demo_images'):
        for img_file in os.listdir('demo_images'):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join('demo_images', img_file))
    
    # Check data directory
    if not test_images and os.path.exists('data/images_001/images'):
        img_dir = 'data/images_001/images'
        for img_file in os.listdir(img_dir)[:3]:  # Test first 3 images
            if img_file.lower().endswith('.png'):
                test_images.append(os.path.join(img_dir, img_file))
    
    if not test_images:
        print("⚠️ No test images found. Creating a dummy test...")
        # Create a dummy image for testing
        dummy_image = Image.new('L', (224, 224), color=128)
        dummy_path = 'test_dummy.png'
        dummy_image.save(dummy_path)
        test_images = [dummy_path]
    
    # Test predictions
    for i, image_path in enumerate(test_images[:2]):  # Test first 2 images
        print(f"\n📷 Testing image {i+1}: {os.path.basename(image_path)}")
        
        try:
            result = ensemble_model.predict_single_image(image_path)
            
            if result:
                print("✅ Prediction successful!")
                print("\n🔍 Results summary:")
                
                # Show positive findings
                positive_findings = ensemble_model.get_positive_predictions(result)
                if positive_findings:
                    print(f"   Positive findings: {', '.join(positive_findings)}")
                else:
                    print("   No positive findings detected")
                
                # Show top 5 highest probability predictions
                predictions = result['predictions']
                sorted_preds = sorted(predictions.items(), 
                                    key=lambda x: x[1]['ensemble_prob'], 
                                    reverse=True)
                
                print("\n📈 Top 5 highest probabilities:")
                for disease, pred_data in sorted_preds[:5]:
                    prob = pred_data['ensemble_prob']
                    threshold = pred_data['threshold_used']
                    prediction = "POSITIVE" if pred_data['prediction'] else "negative"
                    print(f"   {disease:18} | {prob:.3f} | threshold: {threshold:.3f} | {prediction}")
                
                # Check if thresholds are being used correctly
                print("\n🎯 Threshold verification:")
                using_optimal = all(pred_data['threshold_used'] != 0.5 for pred_data in predictions.values())
                if using_optimal:
                    print("   ✅ Using optimal thresholds!")
                else:
                    default_count = sum(1 for pred_data in predictions.values() if pred_data['threshold_used'] == 0.5)
                    print(f"   ⚠️ {default_count} diseases using default threshold (0.5)")
                
            else:
                print("❌ Prediction failed!")
                
        except Exception as e:
            print(f"❌ Error testing image: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up dummy image
    if os.path.exists('test_dummy.png'):
        os.remove('test_dummy.png')
    
    print("\n🏁 Model performance test completed!")

if __name__ == "__main__":
    test_model_performance()