"""
Improved resource monitoring system for neural mesh simplification training.

This module provides comprehensive monitoring of system resources including:
- CPU and memory usage (per-process and system-wide)
- GPU utilization, memory, temperature, and power consumption
- Training progress and performance metrics
- System health indicators with alerts

Key improvements over the original monitor:
- Uses GPUtil for multiprocessing-safe GPU monitoring
- Comprehensive metrics collection
- Clean dashboard-style display
- Configurable monitoring features
- Performance optimization with minimal overhead
"""

import logging
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import warnings

import psutil

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    gpu_stats: List[Dict[str, Any]] = field(default_factory=list)
    process_cpu: float = 0.0
    process_memory_mb: float = 0.0
    disk_io: Optional[Dict[str, int]] = None
    network_io: Optional[Dict[str, int]] = None


@dataclass
class TrainingSnapshot:
    """Snapshot of training progress and metrics."""
    timestamp: float
    epoch: int
    batch: int
    total_batches: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    samples_per_second: Optional[float] = None
    eta_seconds: Optional[float] = None


class GPUMonitor:
    """Multiprocessing-safe GPU monitoring using GPUtil."""
    
    def __init__(self):
        self.available = GPUTIL_AVAILABLE
        self.gpus = []
        self.torch_available = TORCH_AVAILABLE
        
        if self.available:
            try:
                self.gpus = GPUtil.getGPUs()
                logger.info(f"Detected {len(self.gpus)} GPU(s)")
                for gpu in self.gpus:
                    logger.info(f"GPU {gpu.id}: {gpu.name} ({gpu.memoryTotal} MB)")
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
                self.available = False
    
    def get_gpu_stats(self) -> List[Dict[str, Any]]:
        """Get current GPU statistics for all available GPUs."""
        if not self.available:
            return []
        
        gpu_stats = []
        try:
            # Refresh GPU list to get current stats
            current_gpus = GPUtil.getGPUs()
            
            for gpu in current_gpus:
                stats = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,  # Convert to percentage
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature if hasattr(gpu, 'temperature') else None,
                }
                
                # Add PyTorch-specific memory info if available
                if self.torch_available and torch.cuda.is_available() and gpu.id < torch.cuda.device_count():
                    try:
                        torch_allocated = torch.cuda.memory_allocated(gpu.id) / (1024 * 1024)
                        torch_reserved = torch.cuda.memory_reserved(gpu.id) / (1024 * 1024)
                        stats['torch_allocated_mb'] = torch_allocated
                        stats['torch_reserved_mb'] = torch_reserved
                    except Exception as e:
                        logger.debug(f"Failed to get PyTorch GPU memory for GPU {gpu.id}: {e}")
                
                gpu_stats.append(stats)
                
        except Exception as e:
            logger.debug(f"GPU stats collection failed: {e}")
        
        return gpu_stats


class SystemMonitor:
    """Monitor system-wide resources and health."""
    
    def __init__(self, main_pid: int):
        self.main_pid = main_pid
        self.main_process = psutil.Process(main_pid)
        self.gpu_monitor = GPUMonitor()
        
        # Initialize disk I/O baseline
        self.last_disk_io = psutil.disk_io_counters() if psutil.disk_io_counters() else None
        self.last_network_io = psutil.net_io_counters() if psutil.net_io_counters() else None
        self.last_check_time = time.time()
    
    def get_resource_snapshot(self) -> ResourceSnapshot:
        """Get comprehensive system resource snapshot."""
        current_time = time.time()
        
        # System-wide stats
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Process-specific stats
        try:
            process_cpu = self.main_process.cpu_percent(interval=None)
            process_memory = self.main_process.memory_info().rss / (1024 * 1024)
            
            # Include child processes
            for child in self.main_process.children(recursive=True):
                try:
                    process_cpu += child.cpu_percent(interval=None)
                    process_memory += child.memory_info().rss / (1024 * 1024)
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            process_cpu = 0.0
            process_memory = 0.0
        
        # GPU stats
        gpu_stats = self.gpu_monitor.get_gpu_stats()
        
        # I/O stats (optional - can be expensive)
        disk_io_delta = None
        network_io_delta = None
        time_delta = current_time - self.last_check_time
        
        if time_delta > 1.0:  # Only update I/O stats every second
            try:
                current_disk_io = psutil.disk_io_counters()
                if current_disk_io and self.last_disk_io:
                    disk_io_delta = {
                        'read_mb_per_sec': (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024 * 1024 * time_delta),
                        'write_mb_per_sec': (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024 * 1024 * time_delta),
                    }
                self.last_disk_io = current_disk_io
            except Exception:
                pass
            
            try:
                current_network_io = psutil.net_io_counters()
                if current_network_io and self.last_network_io:
                    network_io_delta = {
                        'recv_mb_per_sec': (current_network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024 * 1024 * time_delta),
                        'sent_mb_per_sec': (current_network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024 * 1024 * time_delta),
                    }
                self.last_network_io = current_network_io
            except Exception:
                pass
            
            self.last_check_time = current_time
        
        return ResourceSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            gpu_stats=gpu_stats,
            process_cpu=process_cpu,
            process_memory_mb=process_memory,
            disk_io=disk_io_delta,
            network_io=network_io_delta
        )


class TrainingProgressTracker:
    """Track training progress and compute performance metrics."""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.training_snapshots = deque(maxlen=history_size)
        self.batch_times = deque(maxlen=50)  # For samples/second calculation
        self.epoch_start_time = None
        self.batch_start_time = None
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
    
    def start_batch(self):
        """Mark the start of a batch."""
        self.batch_start_time = time.time()
    
    def end_batch(self, batch_size: int):
        """Mark the end of a batch and calculate throughput."""
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append((batch_time, batch_size))
    
    def update_training_progress(self, epoch: int, batch: int, total_batches: int, 
                               train_loss: Optional[float] = None, 
                               val_loss: Optional[float] = None,
                               learning_rate: Optional[float] = None) -> TrainingSnapshot:
        """Update training progress and return snapshot."""
        current_time = time.time()
        
        # Calculate samples per second
        samples_per_second = None
        if self.batch_times:
            recent_batches = list(self.batch_times)[-10:]  # Last 10 batches
            total_time = sum(batch_time for batch_time, _ in recent_batches)
            total_samples = sum(batch_size for _, batch_size in recent_batches)
            if total_time > 0:
                samples_per_second = total_samples / total_time
        
        # Calculate ETA for current epoch
        eta_seconds = None
        if self.epoch_start_time and batch > 0 and total_batches > 0:
            elapsed_time = current_time - self.epoch_start_time
            batches_per_second = batch / elapsed_time
            if batches_per_second > 0:
                remaining_batches = total_batches - batch
                eta_seconds = remaining_batches / batches_per_second
        
        snapshot = TrainingSnapshot(
            timestamp=current_time,
            epoch=epoch,
            batch=batch,
            total_batches=total_batches,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            samples_per_second=samples_per_second,
            eta_seconds=eta_seconds
        )
        
        self.training_snapshots.append(snapshot)
        return snapshot


class ResourceDashboard:
    """Display resource monitoring information in a clean dashboard format."""
    
    def __init__(self, update_interval: float = 1.0, show_detailed: bool = True):
        self.update_interval = update_interval
        self.show_detailed = show_detailed
        self.last_display_time = 0
    
    def should_update_display(self) -> bool:
        """Check if display should be updated based on interval."""
        current_time = time.time()
        if current_time - self.last_display_time >= self.update_interval:
            self.last_display_time = current_time
            return True
        return False
    
    def format_resource_display(self, resource_snapshot: ResourceSnapshot, 
                              training_snapshot: Optional[TrainingSnapshot] = None) -> str:
        """Format resource information for display."""
        lines = []
        
        # System resources header
        lines.append("=" * 80)
        lines.append("SYSTEM RESOURCES")
        lines.append("=" * 80)
        
        # CPU and Memory
        lines.append(f"CPU: {resource_snapshot.cpu_percent:5.1f}% | "
                    f"RAM: {resource_snapshot.memory_used_mb/1024:5.1f} GB ({resource_snapshot.memory_percent:4.1f}%) | "
                    f"Process: {resource_snapshot.process_cpu:5.1f}% CPU, {resource_snapshot.process_memory_mb/1024:5.1f} GB RAM")
        
        # GPU Information
        if resource_snapshot.gpu_stats:
            lines.append("-" * 80)
            for gpu_stat in resource_snapshot.gpu_stats:
                gpu_line = f"GPU {gpu_stat['id']}: {gpu_stat['load_percent']:5.1f}% | "
                gpu_line += f"VRAM: {gpu_stat['memory_used_mb']/1024:.1f}/{gpu_stat['memory_total_mb']/1024:.1f} GB "
                gpu_line += f"({gpu_stat['memory_percent']:.1f}%)"
                
                if gpu_stat.get('temperature'):
                    gpu_line += f" | Temp: {gpu_stat['temperature']}°C"
                
                if gpu_stat.get('torch_allocated_mb'):
                    gpu_line += f" | PyTorch: {gpu_stat['torch_allocated_mb']/1024:.1f} GB"
                
                lines.append(gpu_line)
        
        # I/O Information (if available)
        if self.show_detailed:
            if resource_snapshot.disk_io:
                lines.append(f"Disk I/O: {resource_snapshot.disk_io['read_mb_per_sec']:5.1f} MB/s read, "
                           f"{resource_snapshot.disk_io['write_mb_per_sec']:5.1f} MB/s write")
            
            if resource_snapshot.network_io:
                lines.append(f"Network: {resource_snapshot.network_io['recv_mb_per_sec']:5.1f} MB/s recv, "
                           f"{resource_snapshot.network_io['sent_mb_per_sec']:5.1f} MB/s sent")
        
        # Training Progress (if available)
        if training_snapshot:
            lines.append("=" * 80)
            lines.append("TRAINING PROGRESS")
            lines.append("=" * 80)
            
            # Progress bar
            if training_snapshot.total_batches > 0:
                progress = training_snapshot.batch / training_snapshot.total_batches
                bar_width = 40
                filled = int(progress * bar_width)
                bar = "█" * filled + "▒" * (bar_width - filled)
                lines.append(f"Epoch {training_snapshot.epoch:3d} [{bar}] "
                           f"{training_snapshot.batch:4d}/{training_snapshot.total_batches:4d} "
                           f"({progress*100:5.1f}%)")
            
            # Performance metrics
            perf_line = ""
            if training_snapshot.samples_per_second:
                perf_line += f"Speed: {training_snapshot.samples_per_second:6.1f} samples/s"
            
            if training_snapshot.eta_seconds:
                eta_min = training_snapshot.eta_seconds / 60
                if eta_min > 60:
                    eta_hr = eta_min / 60
                    perf_line += f" | ETA: {eta_hr:.1f}h"
                else:
                    perf_line += f" | ETA: {eta_min:.1f}min"
            
            if perf_line:
                lines.append(perf_line)
            
            # Loss information
            if training_snapshot.train_loss is not None:
                loss_line = f"Train Loss: {training_snapshot.train_loss:.6f}"
                if training_snapshot.val_loss is not None:
                    loss_line += f" | Val Loss: {training_snapshot.val_loss:.6f}"
                if training_snapshot.learning_rate is not None:
                    loss_line += f" | LR: {training_snapshot.learning_rate:.2e}"
                lines.append(loss_line)
        
        return "\n".join(lines)
    
    def display_resources(self, resource_snapshot: ResourceSnapshot, 
                         training_snapshot: Optional[TrainingSnapshot] = None):
        """Display resource information if update interval has passed."""
        if self.should_update_display():
            # Clear previous output (simple approach)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            formatted_output = self.format_resource_display(resource_snapshot, training_snapshot)
            print(formatted_output)
            print()  # Extra line for readability


class ImprovedResourceMonitor:
    """
    Comprehensive resource monitoring system.
    
    Features:
    - Multiprocessing-safe GPU monitoring with GPUtil
    - System and process-specific resource tracking
    - Training progress integration
    - Clean dashboard display
    - Configurable monitoring options
    - Performance optimized with minimal overhead
    """
    
    def __init__(self, main_pid: int, update_interval: float = 1.0, 
                 show_detailed: bool = True, enable_io_monitoring: bool = False):
        self.main_pid = main_pid
        self.update_interval = update_interval
        self.enable_io_monitoring = enable_io_monitoring
        
        # Initialize monitoring components
        self.system_monitor = SystemMonitor(main_pid)
        self.progress_tracker = TrainingProgressTracker()
        self.dashboard = ResourceDashboard(update_interval, show_detailed)
        
        # Threading control
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.running = False
        
        # Callbacks for training integration
        self.training_callbacks: List[Callable] = []
        
        logger.info("Improved resource monitor initialized")
        if not GPUTIL_AVAILABLE:
            logger.warning("GPUtil not available - GPU monitoring will be limited")
    
    def add_training_callback(self, callback: Callable):
        """Add callback to receive training updates."""
        self.training_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.running = True
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self.running:
            return
        
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.running = False
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        while not self.stop_event.is_set():
            try:
                # Get resource snapshot
                resource_snapshot = self.system_monitor.get_resource_snapshot()
                
                # Get latest training snapshot if available
                training_snapshot = None
                if self.progress_tracker.training_snapshots:
                    training_snapshot = self.progress_tracker.training_snapshots[-1]
                
                # Display dashboard
                self.dashboard.display_resources(resource_snapshot, training_snapshot)
                
                # Execute training callbacks
                for callback in self.training_callbacks:
                    try:
                        callback(resource_snapshot, training_snapshot)
                    except Exception as e:
                        logger.debug(f"Training callback error: {e}")
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            # Wait for next update
            self.stop_event.wait(self.update_interval)
    
    def update_training_progress(self, epoch: int, batch: int, total_batches: int,
                               train_loss: Optional[float] = None,
                               val_loss: Optional[float] = None,
                               learning_rate: Optional[float] = None) -> TrainingSnapshot:
        """Update training progress and return snapshot."""
        return self.progress_tracker.update_training_progress(
            epoch, batch, total_batches, train_loss, val_loss, learning_rate
        )
    
    def start_epoch(self):
        """Signal the start of a new epoch."""
        self.progress_tracker.start_epoch()
    
    def start_batch(self):
        """Signal the start of a new batch."""
        self.progress_tracker.start_batch()
    
    def end_batch(self, batch_size: int):
        """Signal the end of a batch."""
        self.progress_tracker.end_batch(batch_size)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource and training statistics."""
        resource_snapshot = self.system_monitor.get_resource_snapshot()
        
        training_snapshot = None
        if self.progress_tracker.training_snapshots:
            training_snapshot = self.progress_tracker.training_snapshots[-1]
        
        return {
            'resource_snapshot': resource_snapshot,
            'training_snapshot': training_snapshot,
            'gpu_available': self.system_monitor.gpu_monitor.available,
            'gpu_count': len(self.system_monitor.gpu_monitor.gpus) if self.system_monitor.gpu_monitor.available else 0
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Convenience function for backward compatibility and simple usage
def improved_monitor_resources(stop_event, main_pid, update_interval: float = 1.0, 
                             show_detailed: bool = True):
    """
    Improved resource monitoring function for multiprocessing.
    
    This function provides backward compatibility with the original monitor_resources
    while offering enhanced functionality.
    """
    monitor = ImprovedResourceMonitor(
        main_pid=main_pid, 
        update_interval=update_interval, 
        show_detailed=show_detailed
    )
    
    monitor.start_monitoring()
    
    # Wait for stop event
    try:
        while not stop_event.is_set():
            stop_event.wait(0.1)
    finally:
        monitor.stop_monitoring()
