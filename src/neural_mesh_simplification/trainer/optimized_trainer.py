"""
Optimized trainer for neural mesh simplification.

This trainer addresses the major performance bottlenecks:
- Uses optimized dataset for fast data loading
- Implements mixed precision training
- Optimizes batch size and data loading configuration
- Adds gradient accumulation for memory efficiency
- Improves GPU utilization
"""

import logging
import os
import time
from multiprocessing import Event, Process
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from .improved_resource_monitor import ImprovedResourceMonitor
from ..data.optimized_dataset import OptimizedMeshSimplificationDataset, collate_mesh_data, DataAugmentation
from ..losses import CombinedMeshSimplificationLoss
from ..metrics import (
    chamfer_distance,
    normal_consistency,
    edge_preservation,
    hausdorff_distance,
)
from ..models import NeuralMeshSimplification

logger = logging.getLogger(__name__)


class OptimizedTrainer:
    """
    Optimized trainer with significantly improved performance.
    
    Key optimizations:
    - Fast tensor-based data loading (eliminates CPU bottleneck)
    - Mixed precision training (faster GPU computation)
    - Optimized batch size and data loading
    - Gradient accumulation (larger effective batch sizes)
    - Better memory management
    - Improved resource monitoring with GPUtil
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = time.time()
        
        logger.info("Initializing optimized trainer...")
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, falling back to CPU")
        
        # Mixed precision setup
        self.use_mixed_precision = config.get("mixed_precision", True) and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Using mixed precision training (AMP)")
        
        # Gradient accumulation setup
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.effective_batch_size = config["training"]["batch_size"] * self.gradient_accumulation_steps
        logger.info(f"Effective batch size: {self.effective_batch_size} (batch_size={config['training']['batch_size']}, accumulation_steps={self.gradient_accumulation_steps})")
        
        # Initialize model
        logger.info("Initializing model...")
        self._setup_model()
        
        # Initialize optimizer and scheduler
        logger.info("Setting up optimizer and scheduler...")
        self._setup_optimizer_and_scheduler()
        
        # Initialize loss function
        logger.info("Setting up loss function...")
        self._setup_loss_function()
        
        # Setup early stopping
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        
        # Setup checkpointing
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup data loaders
        logger.info("Preparing optimized data loaders...")
        self.train_loader, self.val_loader = self._prepare_optimized_data_loaders()
        
        # Resource monitoring setup - improved version
        self.monitor_resources = config.get("monitor_resources", False)
        self.resource_monitor = None
        
        # Training metrics
        self.training_stats = {
            'epoch_times': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
        }
        
        logger.info("Optimized trainer initialization complete")
    
    def _setup_model(self):
        """Initialize and setup the model."""
        self.model = NeuralMeshSimplification(
            input_dim=self.config["model"]["input_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
            edge_hidden_dim=self.config["model"]["edge_hidden_dim"],
            num_layers=self.config["model"]["num_layers"],
            k=self.config["model"]["k"],
            edge_k=self.config["model"]["edge_k"],
            target_ratio=self.config["model"]["target_ratio"],
            device=self.device,
        ).to(self.device)
        
        # Model compilation for potential speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.get("compile_model", False):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        optimizer_type = self.config["training"].get("optimizer", "adamw").lower()
        
        if optimizer_type == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config["training"]["learning_rate"],
                weight_decay=self.config["training"]["weight_decay"],
            )
        
        # Learning rate scheduler
        scheduler_type = self.config["training"].get("scheduler", "plateau").lower()
        
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config["training"]["num_epochs"],
                eta_min=1e-7
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5,
                min_lr=1e-7
            )
        
        logger.info(f"Using {optimizer_type.upper()} optimizer with {scheduler_type} scheduler")
    
    def _setup_loss_function(self):
        """Setup the loss function."""
        self.criterion = CombinedMeshSimplificationLoss(
            lambda_c=self.config["loss"]["lambda_c"],
            lambda_e=self.config["loss"]["lambda_e"],
            lambda_o=self.config["loss"]["lambda_o"],
            device=self.device,
        )
    
    def _prepare_optimized_data_loaders(self):
        """Setup optimized data loaders with improved performance."""
        data_dir = self.config["data"].get("optimized_data_dir", self.config["data"]["data_dir"])
        
        # Check if optimized data exists
        if not os.path.exists(os.path.join(data_dir, "metadata")):
            logger.warning(f"Optimized data not found at {data_dir}")
            logger.warning("Falling back to original dataset - performance will be suboptimal")
            return self._prepare_fallback_data_loaders()
        
        logger.info(f"Loading optimized dataset from {data_dir}")
        
        # Setup data augmentation
        augmentation = None
        if self.config["training"].get("use_augmentation", True):
            augmentation = DataAugmentation(
                rotation_prob=0.5,
                noise_prob=0.3,
                noise_std=0.01
            )
        
        # Load optimized dataset
        dataset = OptimizedMeshSimplificationDataset(
            data_dir=data_dir,
            transform=augmentation,
            subset_size=self.config.get("debug_subset_size")  # For debugging with smaller dataset
        )
        
        # Log dataset statistics
        stats = dataset.get_statistics()
        logger.info(f"Dataset statistics:")
        logger.info(f"  Samples: {stats['total_samples']}")
        logger.info(f"  Avg vertices: {stats['avg_vertices']:.1f}")
        logger.info(f"  Avg faces: {stats['avg_faces']:.1f}")
        
        # Split dataset
        val_size = int(len(dataset) * self.config["data"]["val_split"])
        train_size = len(dataset) - val_size
        logger.info(f"Dataset split: {train_size} train, {val_size} validation")
        
        if val_size == 0:
            raise ValueError("Validation set is empty. Increase dataset size or reduce val_split.")
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        # Optimized data loader settings
        num_workers = min(4, (os.cpu_count() or 1) // 2)  # Avoid CPU contention
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0
        
        logger.info(f"Data loader settings: {num_workers} workers, pin_memory={pin_memory}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_mesh_data,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_mesh_data,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        
        logger.info("Optimized data loaders prepared successfully")
        return train_loader, val_loader
    
    def _prepare_fallback_data_loaders(self):
        """Fallback to original data loaders if optimized data not available."""
        from ..data import MeshSimplificationDataset
        
        logger.warning("Using fallback data loaders - expect reduced performance")
        
        dataset = MeshSimplificationDataset(
            data_dir=self.config["data"]["data_dir"],
            preprocess=False,
        )
        
        val_size = int(len(dataset) * self.config["data"]["val_split"])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Reduce num_workers to avoid CPU bottleneck
        num_workers = min(2, (os.cpu_count() or 1) // 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        return train_loader, val_loader
    
    def train(self):
        """Main training loop with optimizations and improved monitoring."""
        # Initialize improved resource monitoring
        if self.monitor_resources:
            main_pid = os.getpid()
            self.resource_monitor = ImprovedResourceMonitor(
                main_pid=main_pid,
                update_interval=1.0,
                show_detailed=True
            )
            self.resource_monitor.start_monitoring()
            logger.info("Improved resource monitoring started")
        
        try:
            logger.info("Starting optimized training loop...")
            
            for epoch in range(self.config["training"]["num_epochs"]):
                epoch_start_time = time.time()
                
                # Signal epoch start to monitor
                if self.resource_monitor:
                    self.resource_monitor.start_epoch()
                
                # Training phase
                train_loss = self._train_one_epoch(epoch)
                
                # Validation phase
                val_loss = self._validate()
                
                # Update monitoring with current metrics
                if self.resource_monitor:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.resource_monitor.update_training_progress(
                        epoch=epoch + 1,
                        batch=len(self.train_loader),  # End of epoch
                        total_batches=len(self.train_loader),
                        train_loss=train_loss,
                        val_loss=val_loss,
                        learning_rate=current_lr
                    )
                
                # Update learning rate scheduler
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                # Track metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.training_stats['epoch_times'].append(epoch_time)
                self.training_stats['train_losses'].append(train_loss)
                self.training_stats['val_losses'].append(val_loss)
                self.training_stats['learning_rates'].append(current_lr)
                
                # Logging
                logger.info(
                    f"Epoch [{epoch + 1}/{self.config['training']['num_epochs']}] "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s"
                )
                
                # Save checkpoint
                self._save_checkpoint(epoch, val_loss)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Early stopping check
                if self._early_stopping(val_loss):
                    logger.info("Early stopping triggered")
                    break
                    
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise e
            
        finally:
            # Stop improved resource monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
                print()  # New line after monitoring output
            
            # Final statistics
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {total_time:.1f}s")
            if self.training_stats['epoch_times']:
                avg_epoch_time = sum(self.training_stats['epoch_times']) / len(self.training_stats['epoch_times'])
                logger.info(f"Average epoch time: {avg_epoch_time:.1f}s")
    
    def _train_one_epoch(self, epoch: int) -> float:
        """Optimized training epoch with gradient accumulation and mixed precision."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        total_batches = len(self.train_loader)
        
        # Reset gradients for accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Signal batch start to monitor
            if self.resource_monitor:
                self.resource_monitor.start_batch()
            
            batch = batch.to(self.device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if self.use_mixed_precision:
                with autocast():
                    output = self.model(batch)
                    loss = self.criterion(batch, output)
                    # Scale loss by accumulation steps
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard precision training
                output = self.model(batch)
                loss = self.criterion(batch, output)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Signal batch end to monitor
            if self.resource_monitor:
                self.resource_monitor.end_batch(batch.num_graphs if hasattr(batch, 'num_graphs') else self.config["training"]["batch_size"])
                
                # Update training progress periodically
                if batch_idx % 10 == 0:  # Update every 10 batches
                    current_lr = self.optimizer.param_groups[0]['lr']
                    current_loss = running_loss / num_batches if num_batches > 0 else 0.0
                    self.resource_monitor.update_training_progress(
                        epoch=epoch + 1,
                        batch=batch_idx,
                        total_batches=total_batches,
                        train_loss=current_loss,
                        learning_rate=current_lr
                    )
            
            # Clean up batch to save memory
            del batch, output, loss
        
        return running_loss / num_batches
    
    def _validate(self) -> float:
        """Validation with mixed precision."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        output = self.model(batch)
                        loss = self.criterion(batch, output)
                else:
                    output = self.model(batch)
                    loss = self.criterion(batch, output)
                
                val_loss += loss.item()
                num_batches += 1
                
                del batch, output, loss
        
        return val_loss / num_batches
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
        )
        
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "training_stats": self.training_stats,
        }
        
        if self.use_mixed_precision:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            logger.debug(f"Saved best model (val_loss: {val_loss:.6f})")
    
    def _early_stopping(self, val_loss: float) -> bool:
        """Check for early stopping."""
        if val_loss < self.best_val_loss:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.early_stopping_patience
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["val_loss"]
        
        if self.use_mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if "training_stats" in checkpoint:
            self.training_stats = checkpoint["training_stats"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    
    def get_training_summary(self) -> Dict:
        """Get summary of training performance."""
        if not self.training_stats['epoch_times']:
            return {}
        
        summary = {
            'total_epochs': len(self.training_stats['epoch_times']),
            'total_time': sum(self.training_stats['epoch_times']),
            'avg_epoch_time': sum(self.training_stats['epoch_times']) / len(self.training_stats['epoch_times']),
            'best_val_loss': self.best_val_loss,
            'final_lr': self.training_stats['learning_rates'][-1] if self.training_stats['learning_rates'] else 0,
        }
        
        # Add resource monitoring summary if available
        if self.resource_monitor:
            current_stats = self.resource_monitor.get_current_stats()
            summary['resource_monitoring'] = {
                'gpu_available': current_stats['gpu_available'],
                'gpu_count': current_stats['gpu_count'],
            }
        
        return summary
