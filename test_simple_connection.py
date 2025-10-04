#!/usr/bin/env python3
"""
Simple diagnostic script to test the webapp
"""

import requests
import time
import json

def test_simple_connection():
    """Test basic connection to the webapp."""
    
    print("🔍 Simple Connection Test")
    print("=" * 30)
    
    urls_to_test = [
        'http://localhost:5000',
        'http://127.0.0.1:5000',
        'http://localhost:5000/api/status'
    ]
    
    for url in urls_to_test:
        try:
            print(f"📡 Testing: {url}")
            response = requests.get(url, timeout=10)
            print(f"✅ Status: {response.status_code}")
            
            if 'api/status' in url and response.status_code == 200:
                data = response.json()
                print(f"📊 Server Status: {data}")
            
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed to {url}")
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout connecting to {url}")
        except Exception as e:
            print(f"❌ Error with {url}: {e}")
        
        print()

if __name__ == "__main__":
    test_simple_connection()