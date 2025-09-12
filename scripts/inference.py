#!/usr/bin/env python3
"""
Inference script for chest X-ray disease classification.
"""

import argparse
import sys
import os
import torch
import cv2
import numpy as np
from typing import Dict, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, load_json, get_device
from models import create_model, load_model_weights
from data import create_transforms


class ChestXrayPredictor:
    """Predictor class for chest X-ray disease classification."""
    
    def __init__(self, config_path: str, checkpoint_path: str, thresholds_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            thresholds_path: Optional path to optimal thresholds
        """
        # Load configuration
        self.config = load_config(config_path)
        self.device = get_device()
        
        # Load model
        self.model = create_model(self.config)
        self.model = load_model_weights(self.model, checkpoint_path, self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Create transforms (using validation transform for inference)
        _, self.transform = create_transforms(self.config)
        
        # Load optimal thresholds if provided
        if thresholds_path and os.path.exists(thresholds_path):
            self.thresholds = load_json(thresholds_path)
        else:
            # Use default threshold of 0.5 for all classes
            self.thresholds = {label: 0.5 for label in self.config['diseases']['labels']}
        
        self.disease_labels = self.config['diseases']['labels']
    
    def predict_single_image(self, image_path: str) -> Dict[str, Dict[str, float]]:
        """
        Predict diseases for a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with predictions for each disease
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise IOError(f"Could not load image: {image_path}")
            
            # Apply transforms
            augmented = self.transform(image=image)
            image_tensor = augmented['image']
            
            # Ensure proper preprocessing
            image_tensor = image_tensor.float() / 255.0
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(image_tensor)
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Create results
            results = {}
            for i, disease in enumerate(self.disease_labels):
                prob = float(probabilities[i])
                threshold = self.thresholds.get(disease, 0.5)
                prediction = 1 if prob >= threshold else 0
                
                results[disease] = {
                    'probability': prob,
                    'prediction': prediction,
                    'threshold': threshold
                }
            
            return results
            
        except Exception as e:
            print(f"Error predicting for image {image_path}: {e}")
            return {}
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Dict[str, float]]]:
        """
        Predict diseases for a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of predictions for each image
        """
        results = []
        for image_path in image_paths:
            result = self.predict_single_image(image_path)
            results.append(result)
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference on chest X-ray images')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Path to optimal thresholds JSON file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory containing images for batch inference')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for predictions')
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")
    
    try:
        # Create predictor
        predictor = ChestXrayPredictor(args.config, args.checkpoint, args.thresholds)
        
        if args.image:
            # Single image inference
            print(f"Running inference on: {args.image}")
            results = predictor.predict_single_image(args.image)
            
            if results:
                print("\nPrediction Results:")
                print("-" * 50)
                
                # Find positive predictions
                positive_predictions = [disease for disease, data in results.items() 
                                      if data['prediction'] == 1]
                
                if positive_predictions:
                    print(f"Detected diseases: {', '.join(positive_predictions)}")
                else:
                    print("No diseases detected")
                
                print("\nDetailed Results:")
                for disease, data in results.items():
                    status = "POSITIVE" if data['prediction'] == 1 else "NEGATIVE"
                    print(f"  {disease:<20}: {data['probability']:.4f} ({status})")
                
                # Save results
                from utils import save_json
                save_json({args.image: results}, args.output)
                print(f"\nResults saved to: {args.output}")
            
        elif args.image_dir:
            # Batch inference
            print(f"Running batch inference on directory: {args.image_dir}")
            
            # Get all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.dcm']
            image_paths = []
            
            for file in os.listdir(args.image_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(args.image_dir, file))
            
            if not image_paths:
                print("No images found in the specified directory")
                return
            
            print(f"Found {len(image_paths)} images")
            
            # Run batch inference
            batch_results = predictor.predict_batch(image_paths)
            
            # Combine results
            combined_results = {}
            for image_path, result in zip(image_paths, batch_results):
                combined_results[os.path.basename(image_path)] = result
            
            # Save results
            from utils import save_json
            save_json(combined_results, args.output)
            print(f"Batch inference completed. Results saved to: {args.output}")
            
            # Print summary
            total_images = len(image_paths)
            images_with_findings = sum(1 for result in batch_results 
                                     if any(data['prediction'] == 1 for data in result.values()))
            
            print(f"\nSummary:")
            print(f"  Total images processed: {total_images}")
            print(f"  Images with positive findings: {images_with_findings}")
            print(f"  Images with no findings: {total_images - images_with_findings}")
        
    except Exception as e:
        print(f"Inference failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
