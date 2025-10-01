#!/usr/bin/env python3
"""
Ensemble Model Inference Script
Based on the ab-ensemble.ipynb notebook - provides inference using the ensemble of two models.
"""

import argparse
import torch
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

try:
    from src.models.ensemble_model import load_ensemble_model
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("⚠️ Ensemble model not available. Install required dependencies.")

def setup_ensemble_from_kaggle_outputs(kaggle_working_dir: str = "./kaggle_outputs") -> Dict[str, str]:
    """
    Setup paths for ensemble model files downloaded from Kaggle.
    
    Args:
        kaggle_working_dir: Directory containing Kaggle outputs
        
    Returns:
        Dictionary with file paths
    """
    kaggle_path = Path(kaggle_working_dir)
    
    # Expected files from the Kaggle notebook
    files = {
        'champion_checkpoint': 'best_model_all_out_v1.pth',
        'arnoweng_checkpoint': 'model.pth.tar', 
        'champion_thresholds': 'optimal_thresholds_all_out_v1.json',
        'ensemble_thresholds': 'optimal_thresholds_ensemble_final_v1.json',
        'ensemble_metrics': 'final_metrics_ensemble_final_v1.json',
        'classification_report': 'classification_report_ensemble_final_v1.txt'
    }
    
    paths = {}
    for key, filename in files.items():
        file_path = kaggle_path / filename
        if file_path.exists():
            paths[key] = str(file_path)
            print(f"✅ Found {key}: {filename}")
        else:
            paths[key] = None
            print(f"⚠️ Missing {key}: {filename}")
    
    return paths

def visualize_ensemble_predictions(results: Dict, save_path: str = None):
    """Visualize ensemble predictions with comparison between models."""
    
    predictions = results['predictions']
    diseases = list(predictions.keys())
    
    champion_probs = [predictions[disease]['champion_prob'] for disease in diseases]
    arnoweng_probs = [predictions[disease]['arnoweng_prob'] for disease in diseases]
    ensemble_probs = [predictions[disease]['ensemble_prob'] for disease in diseases]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Model comparison
    x = np.arange(len(diseases))
    width = 0.25
    
    ax1.bar(x - width, champion_probs, width, label='Champion Model', alpha=0.8, color='blue')
    ax1.bar(x, arnoweng_probs, width, label='Arnoweng Model', alpha=0.8, color='red')
    ax1.bar(x + width, ensemble_probs, width, label='Ensemble (Average)', alpha=0.8, color='green')
    
    ax1.set_xlabel('Diseases')
    ax1.set_ylabel('Probability')
    ax1.set_title('Model Predictions Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(diseases, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Final predictions with thresholds
    predicted_diseases = [disease for disease, data in predictions.items() if data['prediction'] == 1]
    colors = ['green' if predictions[disease]['prediction'] == 1 else 'gray' for disease in diseases]
    
    bars = ax2.bar(diseases, ensemble_probs, color=colors, alpha=0.7)
    
    # Add threshold lines
    for i, disease in enumerate(diseases):
        threshold = predictions[disease]['threshold_used']
        ax2.axhline(y=threshold, xmin=i/len(diseases), xmax=(i+1)/len(diseases), 
                   color='red', linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Diseases')
    ax2.set_ylabel('Ensemble Probability')
    ax2.set_title(f'Final Ensemble Predictions (Green = Positive)\nPredicted: {", ".join(predicted_diseases) if predicted_diseases else "No Finding"}')
    ax2.set_xticklabels(diseases, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_detailed_results(results: Dict):
    """Print detailed ensemble results."""
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE MODEL PREDICTION RESULTS")
    print(f"{'='*80}")
    print(f"Image: {Path(results['image_path']).name}")
    
    # Get positive predictions
    positive_diseases = [disease for disease, data in results['predictions'].items() 
                        if data['prediction'] == 1]
    
    print(f"\n🎯 FINAL PREDICTION: {', '.join(positive_diseases) if positive_diseases else 'No Finding'}")
    
    print(f"\n📊 DETAILED PREDICTIONS:")
    print(f"{'Disease':<20} {'Champion':<10} {'Arnoweng':<10} {'Ensemble':<10} {'Threshold':<10} {'Prediction'}")
    print(f"{'-'*80}")
    
    for disease, pred_data in results['predictions'].items():
        prediction_marker = "✅ POSITIVE" if pred_data['prediction'] == 1 else "❌ Negative"
        
        print(f"{disease:<20} "
              f"{pred_data['champion_prob']:<10.4f} "
              f"{pred_data['arnoweng_prob']:<10.4f} "
              f"{pred_data['ensemble_prob']:<10.4f} "
              f"{pred_data['threshold_used']:<10.3f} "
              f"{prediction_marker}")

def main():
    parser = argparse.ArgumentParser(description="Ensemble Model Inference")
    parser.add_argument("--image", type=str, required=True,
                       help="Path to chest X-ray image")
    parser.add_argument("--kaggle-outputs", type=str, default="./kaggle_outputs",
                       help="Directory containing Kaggle notebook outputs")
    parser.add_argument("--champion-checkpoint", type=str,
                       help="Path to champion model checkpoint")
    parser.add_argument("--arnoweng-checkpoint", type=str,
                       help="Path to Arnoweng model checkpoint") 
    parser.add_argument("--ensemble-thresholds", type=str,
                       help="Path to ensemble optimal thresholds JSON")
    parser.add_argument("--output-dir", type=str, default="./outputs/ensemble_results",
                       help="Output directory for results")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization of predictions")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if not ENSEMBLE_AVAILABLE:
        print("❌ Ensemble model dependencies not available.")
        print("Install required packages: pip install torch torchvision torchxrayvision albumentations")
        return
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Setup file paths
    if args.champion_checkpoint and args.arnoweng_checkpoint:
        # Use provided paths
        champion_checkpoint = args.champion_checkpoint
        arnoweng_checkpoint = args.arnoweng_checkpoint
        ensemble_thresholds = args.ensemble_thresholds
    else:
        # Try to find files from Kaggle outputs
        print("🔍 Looking for Kaggle output files...")
        paths = setup_ensemble_from_kaggle_outputs(args.kaggle_outputs)
        
        champion_checkpoint = paths['champion_checkpoint']
        arnoweng_checkpoint = paths['arnoweng_checkpoint']
        ensemble_thresholds = paths['ensemble_thresholds']
        
        if not champion_checkpoint or not arnoweng_checkpoint:
            print("❌ Required model files not found.")
            print("Either provide --champion-checkpoint and --arnoweng-checkpoint")
            print("Or place Kaggle output files in the --kaggle-outputs directory")
            return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🚀 Loading Ensemble Model...")
    start_time = time.time()
    
    try:
        # Load ensemble model
        ensemble = load_ensemble_model(
            champion_checkpoint=champion_checkpoint,
            arnoweng_checkpoint=arnoweng_checkpoint,
            ensemble_thresholds=ensemble_thresholds
        )
        
        load_time = time.time() - start_time
        print(f"✅ Ensemble model loaded in {load_time:.2f} seconds")
        
        # Print model info
        model_info = ensemble.get_model_info()
        print(f"\n🤖 Model Information:")
        print(f"  Ensemble Type: {model_info['ensemble_type']}")
        print(f"  Device: {model_info['device']}")
        print(f"  Total Parameters: {model_info['total_params']:,}")
        print(f"  Optimal Thresholds: {'✅ Available' if model_info['has_ensemble_thresholds'] else '❌ Using default (0.5)'}")
        
        # Make prediction
        print(f"\n📸 Making prediction on: {Path(args.image).name}")
        inference_start = time.time()
        
        results = ensemble.predict_single_image(args.image)
        
        if results:
            inference_time = time.time() - inference_start
            print(f"✅ Inference completed in {inference_time*1000:.1f}ms")
            
            # Print detailed results
            print_detailed_results(results)
            
            # Save results if requested
            if args.save_results:
                results_path = output_dir / f"ensemble_results_{Path(args.image).stem}.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {results_path}")
            
            # Create visualization if requested
            if args.visualize:
                viz_path = output_dir / f"ensemble_visualization_{Path(args.image).stem}.png"
                visualize_ensemble_predictions(results, str(viz_path))
            
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()