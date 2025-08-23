#!/usr/bin/env python3
"""
Optimized training script for neural mesh simplification.

This script uses the optimized components to achieve significantly better performance:
- Optimized data preprocessing and loading
- Mixed precision training
- Better batch sizes and data loading configuration  
- Gradient accumulation
- Enhanced GPU utilization

Usage:
    python scripts/train_optimized.py --config configs/optimized.yaml
    python scripts/train_optimized.py --config configs/optimized.yaml --preprocess-first
    python scripts/train_optimized.py --config configs/optimized.yaml --debug
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.neural_mesh_simplification.trainer.optimized_trainer import OptimizedTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Neural Mesh Simplification model with optimizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimized.yaml",
        help="Path to the training configuration file"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Override data path from config"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Override checkpoint directory from config"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with subset of data"
    )
    
    parser.add_argument(
        "--preprocess-first",
        action="store_true",
        help="Run data preprocessing before training"
    )
    
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    import yaml
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override configuration with command line arguments."""
    if args.data_path:
        config["data"]["data_dir"] = args.data_path
        logger.info(f"Override data_dir: {args.data_path}")
    
    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
        logger.info(f"Override checkpoint_dir: {args.checkpoint_dir}")
    
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        logger.info(f"Override batch_size: {args.batch_size}")
    
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
        logger.info(f"Override learning_rate: {args.learning_rate}")
    
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
        logger.info(f"Override num_epochs: {args.epochs}")
    
    if args.no_mixed_precision:
        config["mixed_precision"] = False
        logger.info("Disabled mixed precision training")
    
    if args.debug:
        config["debug_subset_size"] = 20  # Use very small subset for debugging
        config["training"]["num_epochs"] = 3
        config["training"]["early_stopping_patience"] = 2
        logger.info("Debug mode enabled - using subset of data")
    
    return config


def run_preprocessing(config: dict):
    """Run data preprocessing if needed."""
    try:
        from scripts.preprocess_data_optimized import OptimizedMeshPreprocessor
        
        input_dir = config["data"]["data_dir"]
        output_dir = config["data"].get("optimized_data_dir", "data/processed_optimized")
        
        logger.info("Running optimized data preprocessing...")
        preprocessor = OptimizedMeshPreprocessor(input_dir, output_dir)
        preprocessor.preprocess_dataset()
        logger.info("Data preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        logger.warning("Training will proceed with fallback dataset (reduced performance)")


def check_system_requirements():
    """Check system requirements and log system info."""
    import torch
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        try:
            logger.info(f"CUDA version: {torch.version.cuda}")
        except:
            logger.info("CUDA version: Unknown")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        logger.warning("CUDA not available - training will be slow on CPU")
    
    # Check for PyTorch Geometric
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        logger.error("PyTorch Geometric not installed!")
        logger.error("Please install with: pip install torch-geometric")
        sys.exit(1)
    
    logger.info("=== System Check Complete ===")


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting optimized neural mesh simplification training")
    logger.info(f"Command: {' '.join(sys.argv)}")
    
    # Check system requirements
    check_system_requirements()
    
    # Load and override configuration
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Create checkpoint directory
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Run preprocessing if requested
    if args.preprocess_first:
        run_preprocessing(config)
    
    # Check if optimized data exists
    optimized_data_dir = config["data"].get("optimized_data_dir", "data/processed_optimized")
    if not os.path.exists(os.path.join(optimized_data_dir, "metadata")):
        logger.warning(f"Optimized data not found at {optimized_data_dir}")
        logger.warning("Consider running with --preprocess-first for optimal performance")
        logger.warning("Training will use fallback dataset (slower performance)")
    
    try:
        # Initialize optimized trainer
        logger.info("Initializing optimized trainer...")
        trainer = OptimizedTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Log training configuration
        logger.info("=== Training Configuration ===")
        logger.info(f"Batch size: {config['training']['batch_size']}")
        logger.info(f"Learning rate: {config['training']['learning_rate']}")
        logger.info(f"Epochs: {config['training']['num_epochs']}")
        logger.info(f"Mixed precision: {config.get('mixed_precision', True)}")
        logger.info(f"Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
        effective_batch = config['training']['batch_size'] * config.get('gradient_accumulation_steps', 1)
        logger.info(f"Effective batch size: {effective_batch}")
        
        # Start training
        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time
        
        # Training summary
        summary = trainer.get_training_summary()
        logger.info("=== Training Summary ===")
        logger.info(f"Total training time: {total_time:.1f}s")
        logger.info(f"Total epochs: {summary.get('total_epochs', 0)}")
        logger.info(f"Average epoch time: {summary.get('avg_epoch_time', 0):.1f}s")
        logger.info(f"Best validation loss: {summary.get('best_val_loss', 0):.6f}")
        logger.info(f"Final learning rate: {summary.get('final_lr', 0):.2e}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
