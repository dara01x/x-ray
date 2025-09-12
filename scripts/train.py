#!/usr/bin/env python3
"""
Main training script for chest X-ray disease classification.
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, setup_logging, create_output_directories, get_device, seed_everything
from data import load_and_prepare_data, create_data_splits, create_data_loaders
from models import create_model
from training import create_trainer
from evaluation import create_evaluator


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train chest X-ray disease classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_output_directories(config)
    
    # Setup logging
    logger = setup_logging(config['output']['logs_dir'])
    logger.info("Starting training pipeline")
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Get device
    device = get_device()
    
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df, image_path_map = load_and_prepare_data(config)
        
        # Create train/val splits
        logger.info("Creating train/validation splits...")
        train_df, val_df = create_data_splits(df, config)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(train_df, val_df, config)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Create evaluator
        evaluator = create_evaluator(config['diseases']['labels'])
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(model, config, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.best_metric = checkpoint.get('best_metric_value', -float('inf'))
            logger.info(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader, evaluator)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
