#!/usr/bin/env python3
"""
Evaluation script for chest X-ray disease classification.
"""

import argparse
import sys
import os
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, setup_logging, get_device, save_json
from data import load_and_prepare_data, create_data_splits, create_data_loaders
from models import create_model, load_model_weights
from evaluation import create_evaluator


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate chest X-ray disease classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--find-thresholds', action='store_true',
                       help='Find optimal thresholds for each class')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    logger.info("Starting evaluation pipeline")
    
    # Get device
    device = get_device()
    
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df, image_path_map = load_and_prepare_data(config)
        
        # Create train/val splits (we'll use validation set for evaluation)
        logger.info("Creating train/validation splits...")
        train_df, val_df = create_data_splits(df, config)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        _, val_loader = create_data_loaders(train_df, val_df, config)
        
        # Create model and load weights
        logger.info("Loading model...")
        model = create_model(config)
        model = load_model_weights(model, args.checkpoint, device)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        
        # Create evaluator
        evaluator = create_evaluator(config['diseases']['labels'])
        
        # Evaluate model
        logger.info("Evaluating model...")
        criterion = torch.nn.BCEWithLogitsLoss()
        val_loss, metrics = evaluator.evaluate(model, criterion, val_loader, device)
        
        logger.info(f"Validation Loss: {val_loss:.4f}")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # Save basic metrics
        results = {
            'validation_loss': val_loss,
            'metrics': metrics,
            'checkpoint_path': args.checkpoint
        }
        
        metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
        save_json(results, metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Find optimal thresholds if requested
        if args.find_thresholds:
            logger.info("Finding optimal thresholds...")
            optimal_thresholds = evaluator.find_optimal_thresholds(model, val_loader, device)
            
            # Save thresholds
            thresholds_path = os.path.join(args.output_dir, 'optimal_thresholds.json')
            save_json(optimal_thresholds, thresholds_path)
            logger.info(f"Optimal thresholds saved to {thresholds_path}")
            
            # Generate classification report with optimal thresholds
            logger.info("Generating classification report...")
            report = evaluator.generate_classification_report(model, val_loader, device, optimal_thresholds)
            
            # Save classification report
            report_path = os.path.join(args.output_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {report_path}")
            
            # Plot confusion matrices
            logger.info("Generating confusion matrices...")
            cm_path = os.path.join(args.output_dir, 'confusion_matrices.png')
            evaluator.plot_confusion_matrices(model, val_loader, device, optimal_thresholds, cm_path)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
