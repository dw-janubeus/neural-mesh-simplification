#!/usr/bin/env python3
"""
End-to-End Data Pipeline Test Script

This script tests the complete neural mesh simplification data pipeline:
1. Download dataset from Hugging Face (small subset)
2. Preprocess raw meshes to optimized tensors
3. Load data using OptimizedMeshSimplificationDataset
4. Test DataLoader iteration and batching
5. Validate data integrity throughout the pipeline

Designed for testing on real hardware with comprehensive error handling.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from scripts.download_test_meshes import download_meshes
from scripts.preprocess_data_optimized import OptimizedMeshPreprocessor
from src.neural_mesh_simplification.data.optimized_dataset import (
    OptimizedMeshSimplificationDataset,
    collate_mesh_data,
    DataAugmentation
)


class PipelineTestResults:
    """Class to track and report test results."""
    
    def __init__(self):
        self.results = {}
        self.timings = {}
        self.errors = {}
        self.warnings = []
        
    def add_result(self, test_name: str, success: bool, timing: float = 0.0, details: str = ""):
        self.results[test_name] = success
        self.timings[test_name] = timing
        if details:
            if success:
                self.warnings.append(f"{test_name}: {details}")
            else:
                self.errors[test_name] = details
    
    def print_summary(self):
        """Print comprehensive test results summary."""
        print("\n" + "=" * 80)
        print("END-TO-END PIPELINE TEST RESULTS")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for success in self.results.values() if success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if self.timings:
            total_time = sum(self.timings.values())
            print(f"Total Time: {total_time:.2f}s")
        
        print("\nDetailed Results:")
        print("-" * 50)
        for test_name, success in self.results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            timing = f"({self.timings[test_name]:.2f}s)" if test_name in self.timings else ""
            print(f"{status} {test_name} {timing}")
            
            if test_name in self.errors:
                print(f"    Error: {self.errors[test_name]}")
        
        if self.warnings:
            print("\nWarnings:")
            print("-" * 50)
            for warning in self.warnings:
                print(f"⚠ {warning}")
        
        print("=" * 80)


class EndToEndPipelineTest:
    """Main test class for the complete data pipeline."""
    
    def __init__(self, test_dir: Optional[str] = None, small_subset: bool = True, debug: bool = False):
        self.small_subset = small_subset
        self.debug = debug
        self.results = PipelineTestResults()
        
        # Setup test directory
        if test_dir:
            self.test_dir = Path(test_dir)
        else:
            self.test_dir = Path(tempfile.mkdtemp(prefix="nms_pipeline_test_"))
        
        self.raw_data_dir = self.test_dir / "raw"
        self.processed_data_dir = self.test_dir / "processed"
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized pipeline test in: {self.test_dir}")
        
    def setup_logging(self):
        """Setup detailed logging for the test."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # File handler
        log_file = self.test_dir / "pipeline_test.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    def cleanup(self):
        """Clean up test directory."""
        try:
            if self.test_dir.exists() and self.test_dir.name.startswith("nms_pipeline_test_"):
                shutil.rmtree(self.test_dir)
                self.logger.info(f"Cleaned up test directory: {self.test_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup test directory: {e}")
    
    def test_download_phase(self) -> bool:
        """Test the data download phase."""
        self.logger.info("=" * 50)
        self.logger.info("PHASE 1: Testing Data Download")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Create raw data directory
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Use small subset for testing
            test_pattern = "abc_extra_noisy/03_meshes/*.ply"  # Smaller dataset
            
            self.logger.info(f"Downloading test data to: {self.raw_data_dir}")
            self.logger.info(f"Using pattern: {test_pattern}")
            
            # Download data
            download_meshes(str(self.raw_data_dir), test_pattern)
            
            # Verify download
            ply_files = list(self.raw_data_dir.glob("*.ply"))
            
            if not ply_files:
                raise Exception("No .ply files found after download")
            
            self.logger.info(f"Downloaded {len(ply_files)} mesh files")
            
            # Test loading a few files
            test_count = min(3, len(ply_files))
            valid_meshes = 0
            
            for i, ply_file in enumerate(ply_files[:test_count]):
                try:
                    mesh = trimesh.load(str(ply_file))
                    if isinstance(mesh, trimesh.Trimesh):
                        self.logger.debug(f"✓ {ply_file.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                        valid_meshes += 1
                    else:
                        self.logger.warning(f"⚠ {ply_file.name}: Not a valid Trimesh object")
                except Exception as e:
                    self.logger.warning(f"⚠ {ply_file.name}: Failed to load - {e}")
            
            success = valid_meshes > 0
            details = f"Downloaded {len(ply_files)} files, {valid_meshes}/{test_count} validated"
            
            timing = time.time() - start_time
            self.results.add_result("download_phase", success, timing, details)
            
            if success:
                self.logger.info(f"✓ Download phase completed successfully in {timing:.2f}s")
            else:
                self.logger.error("✗ Download phase failed - no valid mesh files found")
            
            return success
            
        except Exception as e:
            timing = time.time() - start_time
            error_msg = f"Download failed: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.debug(traceback.format_exc())
            self.results.add_result("download_phase", False, timing, error_msg)
            return False
    
    def test_preprocessing_phase(self) -> bool:
        """Test the data preprocessing phase."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("PHASE 2: Testing Data Preprocessing")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Check if raw data exists
            ply_files = list(self.raw_data_dir.glob("*.ply"))
            if not ply_files:
                raise Exception("No raw data found for preprocessing")
            
            # Initialize preprocessor
            max_vertices = 5000 if self.small_subset else 10000
            preprocessor = OptimizedMeshPreprocessor(
                str(self.raw_data_dir),
                str(self.processed_data_dir),
                max_vertices=max_vertices
            )
            
            self.logger.info(f"Preprocessing {len(ply_files)} files with max_vertices={max_vertices}")
            
            # Run preprocessing
            preprocessor.preprocess_dataset()
            
            # Verify preprocessing results
            tensor_dir = self.processed_data_dir / "tensors"
            metadata_dir = self.processed_data_dir / "metadata"
            
            if not tensor_dir.exists():
                raise Exception("Tensor directory not created")
            
            if not metadata_dir.exists():
                raise Exception("Metadata directory not created")
            
            # Check tensor files
            tensor_files = list(tensor_dir.glob("*.pt"))
            if not tensor_files:
                raise Exception("No tensor files created")
            
            # Check metadata files
            metadata_file = metadata_dir / "dataset_metadata.pkl"
            index_file = metadata_dir / "file_index.pkl"
            
            if not metadata_file.exists():
                raise Exception("Dataset metadata file not created")
            
            if not index_file.exists():
                raise Exception("File index not created")
            
            # Test loading a tensor file
            test_tensor = torch.load(tensor_files[0])
            required_attrs = ['x', 'pos', 'edge_index', 'face', 'num_nodes', 'file_id']
            
            for attr in required_attrs:
                if not hasattr(test_tensor, attr):
                    raise Exception(f"Tensor missing required attribute: {attr}")
            
            self.logger.info(f"✓ Sample tensor validation:")
            self.logger.info(f"  - Vertices: {test_tensor.num_nodes}")
            self.logger.info(f"  - Edges: {test_tensor.edge_index.shape[1]}")
            self.logger.info(f"  - Faces: {test_tensor.face.shape[1]}")
            
            success = True
            details = f"Processed {len(tensor_files)} files successfully"
            
            timing = time.time() - start_time
            self.results.add_result("preprocessing_phase", success, timing, details)
            
            self.logger.info(f"✓ Preprocessing phase completed successfully in {timing:.2f}s")
            return True
            
        except Exception as e:
            timing = time.time() - start_time
            error_msg = f"Preprocessing failed: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.debug(traceback.format_exc())
            self.results.add_result("preprocessing_phase", False, timing, error_msg)
            return False
    
    def test_dataset_loading_phase(self) -> bool:
        """Test the dataset loading phase."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("PHASE 3: Testing Dataset Loading")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Check if processed data exists
            if not self.processed_data_dir.exists():
                raise Exception("Processed data directory not found")
            
            # Initialize dataset
            subset_size = 10 if self.small_subset else None
            dataset = OptimizedMeshSimplificationDataset(
                self.processed_data_dir,
                subset_size=subset_size
            )
            
            self.logger.info(f"Dataset initialized with {len(dataset)} samples")
            
            # Test dataset statistics
            stats = dataset.get_statistics()
            if stats:
                self.logger.info(f"Dataset statistics:")
                self.logger.info(f"  - Avg vertices: {stats['avg_vertices']:.1f}")
                self.logger.info(f"  - Avg faces: {stats['avg_faces']:.1f}")
                self.logger.info(f"  - Avg edges: {stats['avg_edges']:.1f}")
                self.logger.info(f"  - Vertex range: {stats['min_vertices']} - {stats['max_vertices']}")
            
            # Test sample loading
            if len(dataset) == 0:
                raise Exception("Dataset is empty")
            
            # Test first few samples
            test_count = min(3, len(dataset))
            valid_samples = 0
            
            for i in range(test_count):
                try:
                    sample = dataset[i]
                    
                    # Validate sample structure
                    required_attrs = ['x', 'pos', 'edge_index', 'face', 'num_nodes']
                    for attr in required_attrs:
                        if not hasattr(sample, attr):
                            raise Exception(f"Sample {i} missing attribute: {attr}")
                    
                    # Validate tensor shapes
                    if sample.x.shape[0] != sample.num_nodes:
                        raise Exception(f"Sample {i}: x shape mismatch")
                    
                    if sample.pos.shape[0] != sample.num_nodes:
                        raise Exception(f"Sample {i}: pos shape mismatch")
                    
                    if sample.edge_index.shape[0] != 2:
                        raise Exception(f"Sample {i}: edge_index shape invalid")
                    
                    if sample.face.shape[0] != 3:
                        raise Exception(f"Sample {i}: face shape invalid")
                    
                    self.logger.debug(f"✓ Sample {i}: {sample.num_nodes} nodes, {sample.edge_index.shape[1]} edges")
                    valid_samples += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠ Sample {i} validation failed: {e}")
            
            if valid_samples == 0:
                raise Exception("No valid samples found")
            
            # Test sample by ID
            if hasattr(dataset, 'file_list') and dataset.file_list:
                test_id = dataset.file_list[0]['file_id']
                sample_by_id = dataset.get_sample_by_id(test_id)
                if sample_by_id is None:
                    self.logger.warning(f"⚠ Failed to retrieve sample by ID: {test_id}")
                else:
                    self.logger.debug(f"✓ Successfully retrieved sample by ID: {test_id}")
            
            success = True
            details = f"Loaded dataset with {len(dataset)} samples, {valid_samples}/{test_count} validated"
            
            timing = time.time() - start_time
            self.results.add_result("dataset_loading_phase", success, timing, details)
            
            self.logger.info(f"✓ Dataset loading phase completed successfully in {timing:.2f}s")
            return True
            
        except Exception as e:
            timing = time.time() - start_time
            error_msg = f"Dataset loading failed: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.debug(traceback.format_exc())
            self.results.add_result("dataset_loading_phase", False, timing, error_msg)
            return False
    
    def test_dataloader_phase(self) -> bool:
        """Test the DataLoader iteration phase."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("PHASE 4: Testing DataLoader Iteration")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Initialize dataset
            subset_size = 10 if self.small_subset else None
            dataset = OptimizedMeshSimplificationDataset(
                self.processed_data_dir,
                subset_size=subset_size
            )
            
            if len(dataset) == 0:
                raise Exception("Dataset is empty")
            
            # Test different batch sizes
            batch_sizes = [1, 2, 4] if len(dataset) >= 4 else [1, min(2, len(dataset))]
            
            for batch_size in batch_sizes:
                self.logger.info(f"Testing DataLoader with batch_size={batch_size}")
                
                # Create DataLoader
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_mesh_data,
                    num_workers=0  # Use 0 for testing to avoid multiprocessing issues
                )
                
                # Test iteration
                batch_count = 0
                total_samples = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    batch_count += 1
                    
                    # Validate batch structure
                    if not hasattr(batch, 'batch'):
                        raise Exception(f"Batch {batch_idx} missing 'batch' attribute")
                    
                    # Count samples in batch
                    unique_batch_ids = torch.unique(batch.batch)
                    batch_samples = len(unique_batch_ids)
                    total_samples += batch_samples
                    
                    self.logger.debug(f"Batch {batch_idx}: {batch_samples} samples, {batch.num_nodes} total nodes")
                    
                    # Test only first few batches to save time
                    if batch_count >= 3:
                        break
                
                if batch_count == 0:
                    raise Exception(f"No batches produced for batch_size={batch_size}")
                
                self.logger.info(f"✓ DataLoader batch_size={batch_size}: {batch_count} batches, {total_samples} samples")
            
            # Test with data augmentation
            self.logger.info("Testing DataLoader with data augmentation")
            
            augmentation = DataAugmentation(rotation_prob=0.5, noise_prob=0.3)
            augmented_dataset = OptimizedMeshSimplificationDataset(
                self.processed_data_dir,
                transform=augmentation,
                subset_size=5 if self.small_subset else None
            )
            
            aug_dataloader = DataLoader(
                augmented_dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_mesh_data,
                num_workers=0
            )
            
            # Test one batch with augmentation
            try:
                batch = next(iter(aug_dataloader))
                self.logger.info(f"✓ Data augmentation test: batch with {len(torch.unique(batch.batch))} samples")
            except Exception as e:
                self.logger.warning(f"⚠ Data augmentation test failed: {e}")
            
            success = True
            details = f"Tested DataLoader with batch sizes {batch_sizes}, all successful"
            
            timing = time.time() - start_time
            self.results.add_result("dataloader_phase", success, timing, details)
            
            self.logger.info(f"✓ DataLoader phase completed successfully in {timing:.2f}s")
            return True
            
        except Exception as e:
            timing = time.time() - start_time
            error_msg = f"DataLoader testing failed: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.debug(traceback.format_exc())
            self.results.add_result("dataloader_phase", False, timing, error_msg)
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks and memory usage."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("PHASE 5: Performance Benchmarks")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # Initialize dataset
            dataset = OptimizedMeshSimplificationDataset(
                self.processed_data_dir,
                subset_size=20 if self.small_subset else None
            )
            
            if len(dataset) == 0:
                raise Exception("Dataset is empty for benchmarking")
            
            # Benchmark single sample loading speed
            self.logger.info("Benchmarking sample loading speed...")
            
            sample_times = []
            test_indices = list(range(min(10, len(dataset))))
            
            for idx in test_indices:
                sample_start = time.time()
                sample = dataset[idx]
                sample_time = time.time() - sample_start
                sample_times.append(sample_time)
            
            avg_sample_time = np.mean(sample_times) * 1000  # Convert to ms
            self.logger.info(f"Average sample loading time: {avg_sample_time:.2f}ms")
            
            # Benchmark batch loading
            self.logger.info("Benchmarking batch loading speed...")
            
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=collate_mesh_data,
                num_workers=0
            )
            
            batch_times = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # Test only first 5 batches
                    break
                batch_start = time.time()
                # Simulate some processing
                _ = batch.pos.mean()
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
            
            avg_batch_time = np.mean(batch_times) * 1000  # Convert to ms
            self.logger.info(f"Average batch processing time: {avg_batch_time:.2f}ms")
            
            # Memory usage estimation
            if torch.cuda.is_available():
                try:
                    # Test GPU memory usage
                    device = torch.device('cuda')
                    sample = dataset[0].to(device)
                    
                    torch.cuda.empty_cache()
                    start_memory = torch.cuda.memory_allocated()
                    
                    # Load a batch to GPU
                    batch = next(iter(dataloader)).to(device)
                    peak_memory = torch.cuda.memory_allocated()
                    
                    memory_mb = (peak_memory - start_memory) / 1024 / 1024
                    self.logger.info(f"GPU memory usage per batch: {memory_mb:.1f}MB")
                    
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    self.logger.warning(f"⚠ GPU memory test failed: {e}")
            
            success = True
            details = f"Sample: {avg_sample_time:.2f}ms, Batch: {avg_batch_time:.2f}ms"
            
            timing = time.time() - start_time
            self.results.add_result("performance_benchmarks", success, timing, details)
            
            self.logger.info(f"✓ Performance benchmarks completed in {timing:.2f}s")
            return True
            
        except Exception as e:
            timing = time.time() - start_time
            error_msg = f"Performance benchmarking failed: {str(e)}"
            self.logger.error(f"✗ {error_msg}")
            self.logger.debug(traceback.format_exc())
            self.results.add_result("performance_benchmarks", False, timing, error_msg)
            return False
    
    def run_complete_test(self, cleanup_on_success: bool = True) -> bool:
        """Run the complete end-to-end pipeline test."""
        self.logger.info("🚀 Starting End-to-End Pipeline Test")
        self.logger.info(f"Test directory: {self.test_dir}")
        self.logger.info(f"Small subset mode: {self.small_subset}")
        self.logger.info(f"Debug mode: {self.debug}")
        
        overall_start = time.time()
        
        try:
            # Phase 1: Download
            phase1_success = self.test_download_phase()
            
            # Phase 2: Preprocessing (only if download succeeded)
            phase2_success = False
            if phase1_success:
                phase2_success = self.test_preprocessing_phase()
            else:
                self.results.add_result("preprocessing_phase", False, 0, "Skipped due to download failure")
            
            # Phase 3: Dataset Loading (only if preprocessing succeeded)
            phase3_success = False
            if phase2_success:
                phase3_success = self.test_dataset_loading_phase()
            else:
                self.results.add_result("dataset_loading_phase", False, 0, "Skipped due to preprocessing failure")
            
            # Phase 4: DataLoader (only if dataset loading succeeded)
            phase4_success = False
            if phase3_success:
                phase4_success = self.test_dataloader_phase()
            else:
                self.results.add_result("dataloader_phase", False, 0, "Skipped due to dataset loading failure")
            
            # Phase 5: Performance benchmarks (only if DataLoader succeeded)
            phase5_success = False
            if phase4_success:
                phase5_success = self.test_performance_benchmarks()
            else:
                self.results.add_result("performance_benchmarks", False, 0, "Skipped due to DataLoader failure")
            
            # Overall success
            overall_success = phase1_success and phase2_success and phase3_success and phase4_success
            
            overall_time = time.time() - overall_start
            self.results.add_result("overall_pipeline", overall_success, overall_time, "Complete end-to-end test")
            
            # Print results
            self.results.print_summary()
            
            # Cleanup if successful and requested
            if overall_success and cleanup_on_success:
                self.logger.info("Test completed successfully - cleaning up test directory")
                self.cleanup()
            elif not overall_success:
                self.logger.warning(f"Test failed - preserving test directory for debugging: {self.test_dir}")
            
            return overall_success
            
        except Exception as e:
            error_msg = f"Critical test failure: {str(e)}"
            self.logger.error(f"💥 {error_msg}")
            self.logger.debug(traceback.format_exc())
            
            overall_time = time.time() - overall_start
            self.results.add_result("overall_pipeline", False, overall_time, error_msg)
            self.results.print_summary()
            
            return False


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="End-to-End Data Pipeline Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_end_to_end_pipeline.py                    # Run full test with small subset
  python test_end_to_end_pipeline.py --debug            # Run in debug mode with verbose logging
  python test_end_to_end_pipeline.py --large-dataset    # Test with larger dataset
  python test_end_to_end_pipeline.py --test-dir /tmp/test  # Use specific test directory
  python test_end_to_end_pipeline.py --no-cleanup       # Keep test files after success
        """
    )
    
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Test directory (default: auto-generated temp dir)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--large-dataset",
        action="store_true",
        help="Test with larger dataset subset (default: small subset)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep test directory after successful completion"
    )
    
    args = parser.parse_args()
    
    # Initialize and run test
    test = EndToEndPipelineTest(
        test_dir=args.test_dir,
        small_subset=not args.large_dataset,
        debug=args.debug
    )
    
    try:
        success = test.run_complete_test(cleanup_on_success=not args.no_cleanup)
        
        if success:
            print("\n🎉 All tests passed! Pipeline is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        test.cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical test error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
