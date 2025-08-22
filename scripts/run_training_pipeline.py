#!/usr/bin/env python3
"""
Complete Neural Mesh Simplification Training Pipeline

This script handles the entire workflow:
1. Download dataset from Hugging Face (if needed)
2. Preprocess the raw mesh data (if needed)  
3. Train the neural mesh simplification model

The script includes smart skip logic to avoid redundant work.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

import yaml
import torch


def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def check_data_exists(data_dir, file_extension):
    """Check if data directory exists and contains files with given extension."""
    if not os.path.exists(data_dir):
        return False
    
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
        from huggingface_hub import snapshot_download
        
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


def preprocess_dataset(raw_data_dir, processed_data_dir, force_preprocess=False):
    """Preprocess the downloaded mesh data."""
    logger = logging.getLogger(__name__)
    
    # Check if processed data already exists
    if not force_preprocess and check_data_exists(processed_data_dir, '.stl'):
        logger.info(f"Processed data already exists in {processed_data_dir}, skipping preprocessing")
        return True
    
    # Check if raw data exists
    if not check_data_exists(raw_data_dir, '.ply'):
        logger.error(f"No raw data found in {raw_data_dir}. Download dataset first.")
        return False
    
    logger.info("Preprocessing mesh data...")
    
    try:
        import networkx as nx
        import trimesh
        from neural_mesh_simplification.data import MeshSimplificationDataset
        from neural_mesh_simplification.data.dataset import load_mesh, preprocess_mesh
        
        # Create output directory
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Initialize dataset
        dataset = MeshSimplificationDataset(data_dir=raw_data_dir)
        
        processed_count = 0
        skipped_count = 0
        
        for idx in tqdm(range(len(dataset)), desc="Processing meshes"):
            file_path = os.path.join(dataset.data_dir, dataset.file_list[idx])
            
            try:
                mesh = load_mesh(file_path)
                if mesh is None:
                    logger.warning(f"Failed to load mesh: {dataset.file_list[idx]}")
                    skipped_count += 1
                    continue
                
                # Preprocess mesh
                mesh = preprocess_mesh(mesh)
                if mesh is None:
                    logger.warning(f"Failed to preprocess mesh: {dataset.file_list[idx]}")
                    skipped_count += 1
                    continue
                
                # Check connectivity
                face_adjacency = trimesh.graph.face_adjacency(mesh.faces)
                G = nx.Graph()
                G.add_edges_from(face_adjacency)
                components = list(nx.connected_components(G))
                
                # Filter meshes with single connected component
                if len(components) != 1:
                    logger.debug(f"Skipping mesh with {len(components)} components: {dataset.file_list[idx]}")
                    skipped_count += 1
                    continue
                
                # Save processed mesh
                output_file = os.path.join(processed_data_dir, dataset.file_list[idx])
                output_file = output_file.replace(".ply", ".stl")
                mesh.export(output_file)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing {dataset.file_list[idx]}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Preprocessing complete: {processed_count} processed, {skipped_count} skipped")
        return processed_count > 0
        
    except ImportError as e:
        logger.error(f"Missing required libraries: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
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


def run_training(config, checkpoint_dir, resume_checkpoint=None, monitor=False):
    """Run the training process."""
    logger = logging.getLogger(__name__)
    
    try:
        from neural_mesh_simplification.trainer import Trainer
        
        # Update config with checkpoint directory
        config["training"]["checkpoint_dir"] = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Add monitoring if requested
        if monitor:
            config["monitor_resources"] = True
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(config)
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            if os.path.exists(resume_checkpoint):
                logger.info(f"Resuming training from {resume_checkpoint}")
                trainer.load_checkpoint(resume_checkpoint)
            else:
                logger.warning(f"Resume checkpoint not found: {resume_checkpoint}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        try:
            # Try to save training state for recovery
            state_file = os.path.join(checkpoint_dir, "training_state.pth")
            trainer.save_training_state(state_file)
            logger.info(f"Training state saved to {state_file}")
        except:
            pass
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete Neural Mesh Simplification Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_pipeline.py                    # Run full pipeline with defaults
  python run_training_pipeline.py --skip-download    # Skip download, start from preprocessing  
  python run_training_pipeline.py --force-download   # Force re-download data
  python run_training_pipeline.py --resume checkpoint.pth  # Resume training
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
        "--processed-data-dir", 
        type=str, 
        default="data/processed",
        help="Directory for processed mesh data (default: data/processed)"
    )
    data_group.add_argument(
        "--force-download", 
        action="store_true",
        help="Force re-download dataset even if it exists"
    )
    data_group.add_argument(
        "--force-preprocess", 
        action="store_true",
        help="Force re-preprocessing even if processed data exists"
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
        default="configs/default.yaml",
        help="Path to training configuration file (default: configs/default.yaml)"
    )
    train_group.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)"
    )
    train_group.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    train_group.add_argument(
        "--monitor", 
        action="store_true", 
        default=True,
        help="Monitor CPU and memory usage during training (default: True)"
    )
    
    # General arguments
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
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
    logger.info("Neural Mesh Simplification Training Pipeline")
    logger.info("=" * 60)
    
    # Verify PyTorch availability
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Step 1: Download dataset
    if not args.skip_download:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Downloading Dataset")
        logger.info("=" * 40)
        
        if args.dry_run:
            logger.info(f"Would download dataset to: {args.raw_data_dir}")
        else:
            success = download_dataset(args.raw_data_dir, args.force_download)
            if not success:
                logger.error("Dataset download failed")
                return 1
    else:
        logger.info("Skipping dataset download as requested")
    
    # Step 2: Preprocess data
    if not args.skip_preprocess:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Preprocessing Data")
        logger.info("=" * 40)
        
        if args.dry_run:
            logger.info(f"Would preprocess data from {args.raw_data_dir} to {args.processed_data_dir}")
        else:
            success = preprocess_dataset(
                args.raw_data_dir, 
                args.processed_data_dir, 
                args.force_preprocess
            )
            if not success:
                logger.error("Data preprocessing failed")
                return 1
    else:
        logger.info("Skipping data preprocessing as requested")
    
    # Step 3: Load configuration and run training
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Training Model")
    logger.info("=" * 40)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return 1
    
    # Update config with processed data directory
    config["data"]["data_dir"] = args.processed_data_dir
    
    if args.dry_run:
        logger.info(f"Would train model with:")
        logger.info(f"  Data directory: {args.processed_data_dir}")
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Checkpoints: {args.checkpoint_dir}")
        if args.resume:
            logger.info(f"  Resume from: {args.resume}")
        logger.info(f"  Monitor: {args.monitor}")
    else:
        # Verify processed data exists
        if not check_data_exists(args.processed_data_dir, '.stl'):
            logger.error(f"No processed data found in {args.processed_data_dir}")
            logger.error("Run preprocessing first or check data directories")
            return 1
        
        success = run_training(
            config,
            args.checkpoint_dir,
            args.resume,
            args.monitor
        )
        
        if not success:
            logger.error("Training failed")
            return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
