#!/usr/bin/env python3
"""
Create synthetic demo data for testing the X-ray AI system.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data import create_demo_data

def main():
    """Create demo data for testing."""
    print("🔬 Creating synthetic chest X-ray demo data...")
    
    # Create demo data
    create_demo_data(data_dir='./data', num_samples=200)
    
    print("✅ Demo data created successfully!")
    print("📁 Location: ./data/")
    print("📊 Samples: 200 synthetic chest X-rays")
    print("🏥 Diseases: 14 classes with realistic distribution")
    print("\n🚀 Ready to train! Run:")
    print("   python scripts/train.py --config configs/config.yaml")

if __name__ == "__main__":
    main()