#!/usr/bin/env python3
"""
Debug script to test image upload and analysis directly
"""

import requests
import os
import json

def test_upload_directly():
    """Test the upload API directly."""
    
    print("🔍 Testing Upload API Directly")
    print("=" * 40)
    
    # Test with a demo image
    demo_image_path = 'demo_images/demo_chest_xray.png'
    
    if not os.path.exists(demo_image_path):
        print(f"❌ Demo image not found at {demo_image_path}")
        
        # Check for other test images
        test_paths = [
            'data/images_001/images/00000000_000.png',
            'data/images_001/images/00000001_000.png',
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                demo_image_path = path
                print(f"✅ Using test image: {demo_image_path}")
                break
        else:
            print("❌ No test images found!")
            # Create a simple test image
            from PIL import Image
            import numpy as np
            
            # Create a synthetic test image
            test_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            img = Image.fromarray(test_img, mode='L')
            demo_image_path = 'test_image.png'
            img.save(demo_image_path)
            print(f"✅ Created test image: {demo_image_path}")
    
    # Test the upload endpoint
    url = 'http://localhost:5000/api/upload'
    
    try:
        print(f"📡 Testing connection to {url}")
        
        # First test if the server is responding
        status_response = requests.get('http://localhost:5000/api/status', timeout=5)
        print(f"📊 Status check: {status_response.status_code}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f"🔧 Server status: {status_data}")
        
        with open(demo_image_path, 'rb') as f:
            files = {'file': f}
            print(f"📤 Uploading: {demo_image_path}")
            
            response = requests.post(url, files=files, timeout=60)
            
            print(f"📊 Response Status: {response.status_code}")
            print(f"📝 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print("✅ JSON Response received:")
                    print(json.dumps(data, indent=2))
                    
                    if data.get('success'):
                        print("🎉 Upload and analysis successful!")
                        
                        if 'predictions' in data:
                            print(f"\n📈 Found {len(data['predictions'])} predictions")
                            
                            # Show top 3 predictions
                            predictions = data['predictions']
                            sorted_preds = sorted(predictions.items(), 
                                                key=lambda x: x[1]['probability'], 
                                                reverse=True)
                            
                            print("\n🔝 Top 3 predictions:")
                            for disease, pred_data in sorted_preds[:3]:
                                prob = pred_data['probability'] * 100
                                status = "POSITIVE" if pred_data['prediction'] else "negative"
                                confidence = pred_data.get('confidence', 'N/A')
                                threshold = pred_data.get('threshold', 0.5) * 100
                                print(f"   {disease:15} | {prob:5.1f}% | {status:8} | Conf: {confidence:10} | Threshold: {threshold:5.1f}%")
                        
                        if 'positive_findings' in data:
                            findings = data['positive_findings']
                            print(f"\n🎯 Positive findings: {', '.join(findings)}")
                        
                        if 'summary' in data:
                            summary = data['summary']
                            print(f"\n📋 Summary:")
                            print(f"   Model: {data.get('model_type', 'Unknown')}")
                            print(f"   Total diseases checked: {summary.get('total_diseases_checked', 'N/A')}")
                            print(f"   Positive findings: {summary.get('positive_findings_count', 'N/A')}")
                            print(f"   Model confidence: {summary.get('model_confidence', 'N/A')}")
                            print(f"   Optimal thresholds: {summary.get('optimal_thresholds_used', 'N/A')}")
                            
                    else:
                        print("❌ Analysis failed!")
                        if 'error' in data:
                            print(f"   Error: {data['error']}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"📄 Raw response: {response.text[:500]}...")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"📄 Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed! Is the server running on localhost:5000?")
    except requests.exceptions.Timeout:
        print("❌ Request timed out! The analysis might be taking too long.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    except FileNotFoundError:
        print(f"❌ Image file not found: {demo_image_path}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test image if we created it
        if demo_image_path == 'test_image.png' and os.path.exists(demo_image_path):
            os.remove(demo_image_path)
            print("🧹 Cleaned up test image")

if __name__ == "__main__":
    test_upload_directly()