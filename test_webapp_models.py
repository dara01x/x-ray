#!/usr/bin/env python3
"""
Quick test script to verify the webapp models are working correctly
"""

import os
import sys
import requests
import json

def test_webapp_models():
    """Test the webapp with a sample image."""
    
    # Check if demo image exists
    demo_image_path = 'demo_images/demo_chest_xray.png'
    
    if not os.path.exists(demo_image_path):
        print("❌ Demo image not found. Creating one...")
        # Create a demo image if it doesn't exist
        from PIL import Image
        import numpy as np
        
        os.makedirs('demo_images', exist_ok=True)
        img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        Image.fromarray(img_array, mode='L').save(demo_image_path)
        print(f"✅ Created demo image: {demo_image_path}")
    
    # Test the API status
    try:
        print("🔄 Testing webapp API...")
        
        # Check status endpoint
        response = requests.get('http://localhost:5000/api/status', timeout=10)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ API Status: {status_data['status']}")
            print(f"🧠 Model Status: {status_data['model_status']}")
            print(f"💻 Device: {status_data['device']}")
        
        # Test file upload and analysis
        print("\n🔄 Testing image analysis...")
        
        with open(demo_image_path, 'rb') as f:
            files = {'file': ('demo.png', f, 'image/png')}
            response = requests.post('http://localhost:5000/api/upload', files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image analysis successful!")
            print(f"📊 Model Type: {result.get('model_type', 'Unknown')}")
            print(f"🔍 Positive Findings: {result.get('positive_findings', [])}")
            
            # Show some prediction details
            predictions = result.get('predictions', {})
            print(f"\n📋 Sample Predictions:")
            for i, (disease, data) in enumerate(predictions.items()):
                if i >= 5:  # Show only first 5
                    break
                prob = data.get('probability', 0) * 100
                status = "POSITIVE" if data.get('prediction', 0) == 1 else "NEGATIVE"
                print(f"   {disease}: {prob:.1f}% ({status})")
            
            if len(predictions) > 5:
                print(f"   ... and {len(predictions) - 5} more diseases")
                
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to webapp. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Error testing webapp: {e}")

def main():
    print("🧪 Testing Chest X-ray AI Webapp Models")
    print("=" * 50)
    
    test_webapp_models()
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")
    print("\n💡 If models are working correctly, you should see:")
    print("   ✅ Model Status: 'Ensemble model active'")
    print("   ✅ Real probability predictions (not demo values)")
    print("   ✅ Disease classifications based on your trained models")

if __name__ == "__main__":
    main()