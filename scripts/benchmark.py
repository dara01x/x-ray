#!/usr/bin/env python3
"""
Simple benchmark script for CI/CD pipeline
"""

import argparse
import time
import sys
from pathlib import Path
import torch

try:
    # Try importing as installed package first
    from radiology_ai.models import ChestXrayModel
    from radiology_ai.utils import load_config
except ImportError:
    # Fallback to local import
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from src.models import ChestXrayModel
    from src.utils import load_config

def benchmark_model_creation():
    """Benchmark model creation time"""
    config = load_config("configs/config.yaml")
    
    start_time = time.time()
    model = ChestXrayModel(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained_weights=config['model']['pretrained_weights']
    )
    creation_time = time.time() - start_time
    
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Model Creation Benchmark:")
    print(f"   Time: {creation_time:.3f} seconds")
    print(f"   Parameters: {param_count:,}")
    print(f"   Memory: ~{param_count * 4 / 1024 / 1024:.1f} MB")
    
    return creation_time

def benchmark_inference():
    """Benchmark inference time"""
    config = load_config("configs/config.yaml")
    
    model = ChestXrayModel(
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained_weights=config['model']['pretrained_weights']
    )
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 224, 224)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            predictions = model(dummy_input)
    inference_time = (time.time() - start_time) / 100
    
    print(f"✅ Inference Benchmark:")
    print(f"   Average time per image: {inference_time*1000:.2f} ms")
    print(f"   Throughput: {1/inference_time:.1f} images/second")
    print(f"   Output shape: {predictions.shape}")
    
    return inference_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark Radiology AI model")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    args = parser.parse_args()
    
    print("🏃 Running Radiology AI Benchmarks")
    print("=" * 40)
    
    try:
        creation_time = benchmark_model_creation()
        inference_time = benchmark_inference()
        
        print("\n📊 Summary:")
        print(f"   Model creation: {creation_time:.3f}s")
        print(f"   Inference: {inference_time*1000:.2f}ms/image")
        
        # Performance thresholds
        if creation_time > 30:  # 30 seconds
            print("⚠️  Model creation is slower than expected")
        else:
            print("✅ Model creation performance is good")
            
        if inference_time > 1.0:  # 1 second per image
            print("⚠️  Inference is slower than expected")
        else:
            print("✅ Inference performance is good")
        
        print("\n🎯 Benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
