#!/usr/bin/env python3
"""
Updated inference script for the new merged model.
Supports both the legacy model and the new merged model from Kaggle.
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Union, Tuple
import json
import time

# Try importing the new model
try:
    from src.models.new_merged_model import load_new_model
    NEW_MODEL_AVAILABLE = True
except ImportError:
    NEW_MODEL_AVAILABLE = False
    print("⚠️ New merged model not available yet. Use legacy model for now.")

# Legacy imports
from src.models import ChestXrayModel
from src.utils import load_config, setup_device

class ModelInference:
    """Unified inference class supporting both legacy and new models."""
    
    def __init__(self, config_path: str, model_type: str = "auto"):
        """
        Initialize inference system.
        
        Args:
            config_path: Path to configuration file
            model_type: "legacy", "new", or "auto" (try new first, fallback to legacy)
        """
        self.config = load_config(config_path)
        self.device = setup_device()
        self.model_type = model_type
        self.model = None
        self.class_names = self.config.get('diseases', {}).get('labels', [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ])
        
        self._load_model()
        
    def _load_model(self):
        """Load the appropriate model based on availability and preference."""
        
        if self.model_type == "new" or (self.model_type == "auto" and NEW_MODEL_AVAILABLE):
            if self._try_load_new_model():
                print(f"✅ Loaded new merged model on {self.device}")
                return
            elif self.model_type == "new":
                raise ValueError("New model requested but not available")
                
        # Fallback to legacy model
        self._load_legacy_model()
        print(f"✅ Loaded legacy model on {self.device}")
        
    def _try_load_new_model(self) -> bool:
        """Try to load the new merged model."""
        if not NEW_MODEL_AVAILABLE:
            return False
            
        try:
            # Check if new model config exists
            new_config_path = Path("configs/new_model_config.yaml")
            if new_config_path.exists():
                with open(new_config_path, 'r') as f:
                    new_config = yaml.safe_load(f)
                
                checkpoint_path = new_config['new_model']['checkpoint_path']
                if Path(checkpoint_path).exists():
                    self.model = load_new_model(checkpoint_path, device=self.device)
                    self.model_type = "new"
                    return True
                    
        except Exception as e:
            print(f"⚠️ Could not load new model: {e}")
            
        return False
        
    def _load_legacy_model(self):
        """Load the legacy model."""
        # Try different checkpoint locations
        possible_paths = [
            "outputs/checkpoints/best_model.pth",
            "outputs/models/best_model.pth",
            self.config.get('model_checkpoint', 'best_model.pth')
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if Path(path).exists():
                checkpoint_path = path
                break
                
        if not checkpoint_path:
            # Create a new model for demo purposes
            print("⚠️ No checkpoint found, creating new model for demo")
            self.model = ChestXrayModel(
                backbone=self.config['model']['backbone'],
                num_classes=len(self.class_names),
                dropout_rate=self.config['model']['dropout_rate']
            ).to(self.device)
        else:
            # Load from checkpoint
            self.model = ChestXrayModel(
                backbone=self.config['model']['backbone'],
                num_classes=len(self.class_names),
                dropout_rate=self.config['model']['dropout_rate']
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            
        self.model.eval()
        self.model_type = "legacy"
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference."""
        
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('L')  # Convert to grayscale
        else:
            image = image_path
            
        # Convert to numpy
        image_np = np.array(image)
        
        # Apply CLAHE if configured
        if self.config.get('preprocessing', {}).get('clahe'):
            clahe = cv2.createCLAHE(
                clipLimit=self.config['preprocessing']['clahe']['clip_limit'],
                tileGridSize=tuple(self.config['preprocessing']['clahe']['tile_grid_size'])
            )
            image_np = clahe.apply(image_np)
            
        # Resize
        target_size = self.config.get('preprocessing', {}).get('image_size', 224)
        image_resized = cv2.resize(image_np, (target_size, target_size))
        
        # Normalize
        if self.config.get('preprocessing', {}).get('normalize', True):
            image_normalized = image_resized.astype(np.float32) / 255.0
        else:
            image_normalized = image_resized.astype(np.float32)
            
        # Convert to tensor and add batch/channel dimensions
        tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return tensor.to(self.device)
        
    def predict_single(self, image_path: str, return_probabilities: bool = True) -> Dict:
        """Make prediction on a single image."""
        
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
        inference_time = time.time() - start_time
        
        # Create results
        results = {
            'image_path': str(image_path),
            'model_type': self.model_type,
            'inference_time_ms': inference_time * 1000,
            'predictions': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            prob = float(probabilities[i])
            results['predictions'][class_name] = {
                'probability': prob,
                'predicted': prob > 0.5  # Default threshold
            }
            
        if return_probabilities:
            results['raw_probabilities'] = probabilities.tolist()
            
        return results
        
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Make predictions on multiple images."""
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(image_paths)}")
                
            result = self.predict_single(image_path)
            results.append(result)
            
        return results
        
    def get_top_predictions(self, results: Dict, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top K predictions from results."""
        predictions = [(name, data['probability']) 
                      for name, data in results['predictions'].items()]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

def main():
    parser = argparse.ArgumentParser(description="X-Ray Disease Classification Inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--image", type=str, 
                       help="Path to single image for inference")
    parser.add_argument("--image-dir", type=str,
                       help="Directory containing images for batch inference")
    parser.add_argument("--output", type=str, default="inference_results.json",
                       help="Output file for results")
    parser.add_argument("--model-type", choices=["auto", "legacy", "new"], default="auto",
                       help="Which model to use")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Show top K predictions")
    
    args = parser.parse_args()
    
    # Initialize inference system
    print("🚀 Initializing X-Ray AI Inference System...")
    inference_system = ModelInference(args.config, model_type=args.model_type)
    
    results = []
    
    if args.image:
        # Single image inference
        print(f"\n📸 Processing single image: {args.image}")
        result = inference_system.predict_single(args.image)
        results.append(result)
        
        # Display results
        print(f"\n✅ Results for {args.image}:")
        print(f"Model: {result['model_type']}")
        print(f"Inference time: {result['inference_time_ms']:.1f}ms")
        
        top_predictions = inference_system.get_top_predictions(result, args.top_k)
        print(f"\nTop {args.top_k} predictions:")
        for disease, prob in top_predictions:
            print(f"  {disease}: {prob:.3f} ({prob*100:.1f}%)")
            
    elif args.image_dir:
        # Batch inference
        image_dir = Path(args.image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
        image_paths = [p for p in image_dir.rglob('*') 
                      if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            print(f"❌ No images found in {args.image_dir}")
            return
            
        print(f"\n📁 Processing {len(image_paths)} images from {args.image_dir}")
        results = inference_system.predict_batch(image_paths)
        
        # Display summary
        print(f"\n✅ Batch processing complete!")
        print(f"Processed: {len(results)} images")
        avg_time = np.mean([r['inference_time_ms'] for r in results])
        print(f"Average inference time: {avg_time:.1f}ms")
        
    else:
        print("❌ Please provide either --image or --image-dir")
        return
        
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {args.output}")
    
    # Model info
    print(f"\n🤖 Model Information:")
    print(f"Type: {inference_system.model_type}")
    print(f"Device: {inference_system.device}")
    if hasattr(inference_system.model, 'parameters'):
        total_params = sum(p.numel() for p in inference_system.model.parameters())
        print(f"Parameters: {total_params:,}")

if __name__ == "__main__":
    main()