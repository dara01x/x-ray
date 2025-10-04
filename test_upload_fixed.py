#!/usr/bin/env python3
"""
Test script to verify the fixed web application upload functionality
"""

import requests
import json
import os
import time
from pathlib import Path

def test_server_status():
    """Test if server is running"""
    try:
        response = requests.get('http://127.0.0.1:5000', timeout=5)
        print(f"✅ Server is running - Status: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Server connection failed: {e}")
        return False

def test_image_upload():
    """Test image upload and analysis"""
    # Find a test image
    test_images = [
        'data/images_001/images/00000000_000.png',
        'data/images_001/images/00000001_000.png',
        'demo_images/sample_xray.png'
    ]
    
    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found")
        return False
    
    print(f"🔬 Testing with image: {test_image}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                'http://127.0.0.1:5000/analyze',
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image analysis successful!")
            print(f"📊 Predictions received for {len(result.get('predictions', {}))} diseases")
            
            # Check for ensemble results
            if 'ensemble_result' in result:
                ensemble = result['ensemble_result']
                print(f"🎯 Ensemble confidence: {ensemble.get('overall_confidence', 'N/A')}")
                
                # Show top predictions
                predictions = ensemble.get('predictions', {})
                sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                print("🏆 Top 3 predictions:")
                for disease, conf in sorted_preds[:3]:
                    print(f"   {disease}: {conf:.3f}")
            
            return True
        else:
            print(f"❌ Upload failed - Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🔧 Testing Fixed X-ray Analysis Web Application")
    print("=" * 50)
    
    # Test 1: Server status
    print("1. Testing server status...")
    if not test_server_status():
        print("❌ Server test failed. Make sure app_fixed.py is running.")
        return
    
    time.sleep(1)
    
    # Test 2: Image upload
    print("\n2. Testing image upload and analysis...")
    if test_image_upload():
        print("\n✅ All tests passed! The webapp is working correctly.")
        print("🎉 You can now upload X-ray images at: http://127.0.0.1:5000")
    else:
        print("\n❌ Upload test failed. Check the server logs for errors.")

if __name__ == "__main__":
    main()