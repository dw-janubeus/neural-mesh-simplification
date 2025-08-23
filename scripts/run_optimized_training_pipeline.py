#!/usr/bin/env python3
"""
Complete Neural Mesh Simplification Optimized Training Pipeline

This script orchestrates the entire optimized workflow:
1. Download dataset from Hugging Face (if needed)
2. Preprocess the raw mesh data using the optimized preprocessor (if needed)
3. Train the neural mesh simplification model using the optimized trainer

The script includes smart skip logic to avoid redundant work and leverages
the performance improvements from the optimized components.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

import yaml # pyright: ignore[reportMissingModuleSource]
import torch

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from scripts.preprocess_data_optimized import OptimizedMeshPreprocessor
from src.neural_mesh_simplification.trainer.optimized_trainer import OptimizedTrainer


def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def check_data_exists(data_dir, file_extension, check_metadata=False):
    """Check if data directory exists and contains files with given extension or metadata."""
    if not os.path.exists(data_dir):
        return False
    
    if check_metadata:
        metadata_path = Path(data_dir) / "metadata" / "dataset_metadata.pkl"
        if metadata_path.exists():
            return True
        
    files = [f for f in os.listdir(data_dir) if f.endswith(file_extension)]
    return len(files) > 0


def download_dataset(raw_data_dir, force_download=False):
    """Download mesh dataset from Hugging Face."""
    logger = logging.getLogger(__name__)
    
    # Check if data already exists
    if not force_download and check_data_exists(raw_data_dir, '.ply'):
        logger.info(f"Raw data already exists in {raw_data_dir}, skipping download")
        return True
    
    logger.info("Downloading dataset from Hugging Face...")
    
    try:
        from huggingface_hub import snapshot_download # pyright: ignore[reportMissingImports]
        
        # Create directories
        os.makedirs(raw_data_dir, exist_ok=True)
        wip_folder = os.path.join(raw_data_dir, "wip")
        os.makedirs(wip_folder, exist_ok=True)
        
        # Download patterns - using abc_extra_noisy as it's smaller than abc_train
        folder_patterns = ["abc_extra_noisy/03_meshes/*.ply"]
        
        logger.info(f"Downloading to temporary folder: {wip_folder}")
        snapshot_download(
            repo_id="perler/ppsurf",
            repo_type="dataset",
            cache_dir=wip_folder,
            allow_patterns=folder_patterns[0],
        )
        
        # Move files from wip folder to target folder
        files_moved = 0
        for root, _, files in os.walk(wip_folder):
            for file in files:
                if file.endswith(".ply"):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(raw_data_dir, file)
                    shutil.copy2(src_file, dest_file)
                    files_moved += 1
        
        # Clean up wip folder
        shutil.rmtree(wip_folder)
        
        logger.info(f"Successfully downloaded {files_moved} mesh files to {raw_data_dir}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install it with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def run_optimized_preprocessing(raw_data_dir, optimized_data_dir, force_preprocess=False):
    """Run the optimized data preprocessing pipeline."""
    logger = logging.getLogger(__name__)
    
    # Check if processed data already exists
    if not force_preprocess and check_data_exists(optimized_data_dir, '.pt', check_metadata=True):
        logger.info(f"Optimized processed data already exists in {optimized_data_dir}, skipping preprocessing")
        return True
    
    # Check if raw data exists
    if not check_data_exists(raw_data_dir, '.ply'):
        logger.error(f"No raw data found in {raw_data_dir}. Download dataset first.")
        return False
    
    logger.info("Running optimized mesh data preprocessing...")
    
    try:
        preprocessor = OptimizedMeshPreprocessor(raw_data_dir, optimized_data_dir)
        preprocessor.preprocess_dataset()
        logger.info("Optimized data preprocessing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Optimized data preprocessing failed: {e}")
        return False


def load_config(config_path):
    """Load training configuration from YAML file."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def run_optimized_training(config, resume_checkpoint=None):
    """Run the optimized training process."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize optimized trainer
        logger.info("Initializing optimized trainer...")
        trainer = OptimizedTrainer(config)
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            if os.path.exists(resume_checkpoint):
                logger.info(f"Resuming training from {resume_checkpoint}")
                trainer.load_checkpoint(resume_checkpoint)
            else:
                logger.warning(f"Resume checkpoint not found: {resume_checkpoint}")
        
        # Start training
        logger.info("Starting optimized training...")
        trainer.train()
        
        logger.info("Optimized training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Optimized training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete Neural Mesh Simplification Optimized Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimized_training_pipeline.py                    # Run full pipeline with optimized defaults
  python run_optimized_training_pipeline.py --skip-download    # Skip download, start from preprocessing
  python run_optimized_training_pipeline.py --force-download   # Force re-download data
  python run_optimized_training_pipeline.py --resume data/checkpoints_optimized/checkpoint.pth  # Resume training
  python run_optimized_training_pipeline.py --debug            # Run in debug mode with small dataset
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument(
        "--raw-data-dir", 
        type=str, 
        default="data/raw",
        help="Directory for raw mesh data (default: data/raw)"
    )
    data_group.add_argument(
        "--optimized-data-dir", 
        type=str, 
        default="data/processed_optimized",
        help="Directory for optimized processed mesh data (default: data/processed_optimized)"
    )
    data_group.add_argument(
        "--force-download", 
        action="store_true",
        help="Force re-download dataset even if it exists"
    )
    data_group.add_argument(
        "--force-preprocess", 
        action="store_true",
        help="Force re-preprocessing even if optimized processed data exists"
    )
    data_group.add_argument(
        "--skip-download", 
        action="store_true",
        help="Skip data download step"
    )
    data_group.add_argument(
        "--skip-preprocess", 
        action="store_true",
        help="Skip data preprocessing step"
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument(
        "--config", 
        type=str, 
        default="configs/optimized.yaml",
        help="Path to optimized training configuration file (default: configs/optimized.yaml)"
    )
    train_group.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="data/checkpoints_optimized",
        help="Directory to save model checkpoints (default: data/checkpoints_optimized)"
    )
    train_group.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # General arguments
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging and use a small data subset for quick testing"
    )
    general_group.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - showing what would be executed")
    
    logger.info("=" * 60)
    logger.info("Neural Mesh Simplification Optimized Training Pipeline")
    logger.info("=" * 60)
    
    # Verify PyTorch availability
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Step 1: Download dataset
    if not args.skip_download:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Downloading Raw Dataset")
        logger.info("=" * 40)
        
        if args.dry_run:
            logger.info(f"Would download dataset to: {args.raw_data_dir}")
        else:
            success = download_dataset(args.raw_data_dir, args.force_download)
            if not success:
                logger.error("Dataset download failed")
                return 1
    else:
        logger.info("Skipping raw dataset download as requested")
    
    # Step 2: Preprocess data
    if not args.skip_preprocess:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Optimized Data Preprocessing")
        logger.info("=" * 40)
        
        if args.dry_run:
            logger.info(f"Would preprocess data from {args.raw_data_dir} to {args.optimized_data_dir}")
        else:
            success = run_optimized_preprocessing(
                args.raw_data_dir, 
                args.optimized_data_dir, 
                args.force_preprocess
            )
            if not success:
                logger.error("Optimized data preprocessing failed")
                return 1
    else:
        logger.info("Skipping optimized data preprocessing as requested")
    
    # Step 3: Load configuration and run training
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Training Model with Optimized Trainer")
    logger.info("=" * 40)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Override config with CLI args
    if args.debug:
        config["debug_subset_size"] = 20
        config["training"]["num_epochs"] = 3
        config["training"]["early_stopping_patience"] = 2
        logger.info("Debug mode enabled: using small data subset and fewer epochs")
    
    # Ensure optimized data directory is set in config for the trainer
    config["data"]["optimized_data_dir"] = args.optimized_data_dir
    config["training"]["checkpoint_dir"] = args.checkpoint_dir
    
    if args.dry_run:
        logger.info(f"Would train model with:")
        logger.info(f"  Optimized Data directory: {args.optimized_data_dir}")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Checkpoints: {args.checkpoint_dir}")
        if args.resume:
            logger.info(f"  Resume from: {args.resume}")
        logger.info(f"  Effective Batch Size: {config['training']['batch_size'] * config.get('gradient_accumulation_steps', 1)}")
        logger.info(f"  Mixed Precision: {config.get('mixed_precision', True)}")
    else:
        # Verify optimized processed data exists
        if not check_data_exists(args.optimized_data_dir, '.pt', check_metadata=True):
            logger.error(f"No optimized processed data found in {args.optimized_data_dir}")
            logger.error("Run optimized preprocessing first or check data directories")
            return 1
        
        success = run_optimized_training(
            config,
            args.resume,
        )
        
        if not success:
            logger.error("Optimized training failed")
            return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("Optimized Training Pipeline completed successfully!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
