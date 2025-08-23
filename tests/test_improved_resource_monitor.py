"""
Test suite for the improved resource monitoring system.

These tests verify that the enhanced monitoring system works correctly
with multiprocessing-safe GPU monitoring and comprehensive metrics.
"""

import os
import time
import unittest
from multiprocessing import Process
import tempfile
import shutil

import pytest

from src.neural_mesh_simplification.trainer.improved_resource_monitor import (
    ImprovedResourceMonitor,
    GPUMonitor,
    SystemMonitor,
    ResourceDashboard,
    TrainingProgressTracker,
    ResourceSnapshot,
    TrainingSnapshot
)


class TestGPUMonitor(unittest.TestCase):
    """Test GPU monitoring functionality."""
    
    def test_gpu_monitor_initialization(self):
        """Test GPU monitor initializes correctly."""
        monitor = GPUMonitor()
        self.assertIsInstance(monitor, GPUMonitor)
        self.assertIsInstance(monitor.available, bool)
        
    def test_get_gpu_stats(self):
        """Test GPU statistics collection."""
        monitor = GPUMonitor()
        stats = monitor.get_gpu_stats()
        self.assertIsInstance(stats, list)
        
        # If GPUs are available, check structure
        if monitor.available and stats:
            for gpu_stat in stats:
                self.assertIn('id', gpu_stat)
                self.assertIn('name', gpu_stat)
                self.assertIn('load_percent', gpu_stat)
                self.assertIn('memory_used_mb', gpu_stat)
                self.assertIn('memory_total_mb', gpu_stat)
                self.assertIn('memory_percent', gpu_stat)


class TestSystemMonitor(unittest.TestCase):
    """Test system-wide resource monitoring."""
    
    def test_system_monitor_initialization(self):
        """Test system monitor initializes with current process."""
        current_pid = os.getpid()
        monitor = SystemMonitor(current_pid)
        self.assertEqual(monitor.main_pid, current_pid)
        self.assertIsNotNone(monitor.gpu_monitor)
        
    def test_resource_snapshot(self):
        """Test resource snapshot collection."""
        current_pid = os.getpid()
        monitor = SystemMonitor(current_pid)
        snapshot = monitor.get_resource_snapshot()
        
        self.assertIsInstance(snapshot, ResourceSnapshot)
        self.assertGreater(snapshot.timestamp, 0)
        self.assertGreaterEqual(snapshot.cpu_percent, 0)
        self.assertGreater(snapshot.memory_used_mb, 0)
        self.assertGreater(snapshot.memory_percent, 0)
        self.assertGreaterEqual(snapshot.process_cpu, 0)
        self.assertGreater(snapshot.process_memory_mb, 0)
        self.assertIsInstance(snapshot.gpu_stats, list)


class TestTrainingProgressTracker(unittest.TestCase):
    """Test training progress tracking functionality."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initializes correctly."""
        tracker = TrainingProgressTracker(history_size=50)
        self.assertEqual(tracker.history_size, 50)
        self.assertEqual(len(tracker.training_snapshots), 0)
        self.assertEqual(len(tracker.batch_times), 0)
        
    def test_epoch_tracking(self):
        """Test epoch start/end tracking."""
        tracker = TrainingProgressTracker()
        
        # Start epoch
        tracker.start_epoch()
        self.assertIsNotNone(tracker.epoch_start_time)
        
    def test_batch_tracking(self):
        """Test batch timing tracking."""
        tracker = TrainingProgressTracker()
        
        # Start and end batch
        tracker.start_batch()
        time.sleep(0.01)  # Small delay
        tracker.end_batch(batch_size=4)
        
        self.assertEqual(len(tracker.batch_times), 1)
        batch_time, batch_size = tracker.batch_times[0]
        self.assertGreater(batch_time, 0)
        self.assertEqual(batch_size, 4)
        
    def test_training_progress_update(self):
        """Test training progress updates."""
        tracker = TrainingProgressTracker()
        
        snapshot = tracker.update_training_progress(
            epoch=1,
            batch=10,
            total_batches=100,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=1e-4
        )
        
        self.assertIsInstance(snapshot, TrainingSnapshot)
        self.assertEqual(snapshot.epoch, 1)
        self.assertEqual(snapshot.batch, 10)
        self.assertEqual(snapshot.total_batches, 100)
        self.assertEqual(snapshot.train_loss, 0.5)
        self.assertEqual(snapshot.val_loss, 0.6)
        self.assertEqual(snapshot.learning_rate, 1e-4)
        self.assertEqual(len(tracker.training_snapshots), 1)


class TestResourceDashboard(unittest.TestCase):
    """Test resource dashboard display functionality."""
    
    def test_dashboard_initialization(self):
        """Test dashboard initializes correctly."""
        dashboard = ResourceDashboard(update_interval=2.0, show_detailed=False)
        self.assertEqual(dashboard.update_interval, 2.0)
        self.assertFalse(dashboard.show_detailed)
        
    def test_should_update_display(self):
        """Test display update timing."""
        dashboard = ResourceDashboard(update_interval=0.1)
        
        # Should update initially
        self.assertTrue(dashboard.should_update_display())
        
        # Should not update immediately after
        self.assertFalse(dashboard.should_update_display())
        
        # Should update after interval
        time.sleep(0.15)
        self.assertTrue(dashboard.should_update_display())
        
    def test_format_resource_display(self):
        """Test resource information formatting."""
        dashboard = ResourceDashboard()
        
        # Create sample resource snapshot
        resource_snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=25.5,
            memory_used_mb=4096.0,
            memory_percent=50.0,
            gpu_stats=[{
                'id': 0,
                'name': 'Test GPU',
                'load_percent': 75.0,
                'memory_used_mb': 2048,
                'memory_total_mb': 8192,
                'memory_percent': 25.0,
                'temperature': 65
            }]
        )
        
        formatted = dashboard.format_resource_display(resource_snapshot)
        self.assertIsInstance(formatted, str)
        self.assertIn('SYSTEM RESOURCES', formatted)
        self.assertIn('CPU: 25.5%', formatted)
        self.assertIn('Test GPU', formatted)


class TestImprovedResourceMonitor(unittest.TestCase):
    """Test the main improved resource monitor."""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        current_pid = os.getpid()
        monitor = ImprovedResourceMonitor(
            main_pid=current_pid,
            update_interval=0.5,
            show_detailed=False
        )
        
        self.assertEqual(monitor.main_pid, current_pid)
        self.assertEqual(monitor.update_interval, 0.5)
        self.assertIsNotNone(monitor.system_monitor)
        self.assertIsNotNone(monitor.progress_tracker)
        self.assertIsNotNone(monitor.dashboard)
        
    def test_context_manager(self):
        """Test monitor works as context manager."""
        current_pid = os.getpid()
        
        with ImprovedResourceMonitor(main_pid=current_pid, update_interval=0.1) as monitor:
            self.assertTrue(monitor.running)
            time.sleep(0.2)  # Let it run briefly
            
        self.assertFalse(monitor.running)
        
    def test_training_integration(self):
        """Test training progress integration."""
        current_pid = os.getpid()
        monitor = ImprovedResourceMonitor(main_pid=current_pid)
        
        # Test epoch/batch tracking
        monitor.start_epoch()
        monitor.start_batch()
        time.sleep(0.01)
        monitor.end_batch(batch_size=4)
        
        # Test progress update
        snapshot = monitor.update_training_progress(
            epoch=1, batch=5, total_batches=20,
            train_loss=0.3, learning_rate=1e-3
        )
        
        self.assertIsInstance(snapshot, TrainingSnapshot)
        self.assertEqual(snapshot.epoch, 1)
        self.assertEqual(snapshot.train_loss, 0.3)
        
    def test_get_current_stats(self):
        """Test current statistics retrieval."""
        current_pid = os.getpid()
        monitor = ImprovedResourceMonitor(main_pid=current_pid)
        
        stats = monitor.get_current_stats()
        self.assertIn('resource_snapshot', stats)
        self.assertIn('gpu_available', stats)
        self.assertIn('gpu_count', stats)
        
        self.assertIsInstance(stats['resource_snapshot'], ResourceSnapshot)
        self.assertIsInstance(stats['gpu_available'], bool)
        self.assertIsInstance(stats['gpu_count'], int)


class TestMonitoringIntegration(unittest.TestCase):
    """Integration tests for the monitoring system."""
    
    def test_monitor_with_simulated_training(self):
        """Test monitor during simulated training scenario."""
        current_pid = os.getpid()
        
        with ImprovedResourceMonitor(main_pid=current_pid, update_interval=0.1) as monitor:
            # Simulate training loop
            for epoch in range(2):
                monitor.start_epoch()
                
                for batch in range(5):
                    monitor.start_batch()
                    
                    # Simulate some work
                    time.sleep(0.01)
                    
                    monitor.end_batch(batch_size=4)
                    
                    # Update progress
                    monitor.update_training_progress(
                        epoch=epoch + 1,
                        batch=batch + 1,
                        total_batches=5,
                        train_loss=1.0 / (epoch + 1) / (batch + 1),
                        learning_rate=1e-4
                    )
                
                # Brief pause between epochs
                time.sleep(0.05)
        
        # Verify we captured training data
        self.assertGreater(len(monitor.progress_tracker.training_snapshots), 0)
        self.assertGreater(len(monitor.progress_tracker.batch_times), 0)


if __name__ == '__main__':
    unittest.main()
