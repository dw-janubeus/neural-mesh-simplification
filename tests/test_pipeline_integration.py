"""
Pytest-compatible integration test for the data pipeline.

This provides a simpler interface for running pipeline tests through pytest,
while the full end-to-end test script provides more detailed reporting.
"""

import pytest
import tempfile
from pathlib import Path

from test_end_to_end_pipeline import EndToEndPipelineTest


class TestPipelineIntegration:
    """Pytest class for pipeline integration testing."""
    
    @pytest.fixture
    def test_instance(self):
        """Create a test instance with temporary directory."""
        test_dir = tempfile.mkdtemp(prefix="pytest_pipeline_")
        instance = EndToEndPipelineTest(
            test_dir=test_dir, 
            small_subset=True, 
            debug=False
        )
        yield instance
        # Cleanup after test
        instance.cleanup()
    
    def test_download_phase(self, test_instance):
        """Test data download phase."""
        success = test_instance.test_download_phase()
        assert success, "Download phase failed"
    
    def test_preprocessing_phase(self, test_instance):
        """Test preprocessing phase (requires download first)."""
        # Run download first
        download_success = test_instance.test_download_phase()
        if not download_success:
            pytest.skip("Download phase failed, skipping preprocessing test")
        
        # Test preprocessing
        success = test_instance.test_preprocessing_phase()
        assert success, "Preprocessing phase failed"
    
    def test_dataset_loading_phase(self, test_instance):
        """Test dataset loading phase (requires preprocessing first)."""
        # Run prerequisite phases
        if not test_instance.test_download_phase():
            pytest.skip("Download phase failed, skipping dataset loading test")
        
        if not test_instance.test_preprocessing_phase():
            pytest.skip("Preprocessing phase failed, skipping dataset loading test")
        
        # Test dataset loading
        success = test_instance.test_dataset_loading_phase()
        assert success, "Dataset loading phase failed"
    
    def test_dataloader_phase(self, test_instance):
        """Test DataLoader iteration phase (requires all previous phases)."""
        # Run all prerequisite phases
        if not test_instance.test_download_phase():
            pytest.skip("Download phase failed, skipping DataLoader test")
        
        if not test_instance.test_preprocessing_phase():
            pytest.skip("Preprocessing phase failed, skipping DataLoader test")
        
        if not test_instance.test_dataset_loading_phase():
            pytest.skip("Dataset loading phase failed, skipping DataLoader test")
        
        # Test DataLoader
        success = test_instance.test_dataloader_phase()
        assert success, "DataLoader phase failed"
    
    @pytest.mark.slow
    def test_complete_pipeline(self, test_instance):
        """Test the complete end-to-end pipeline."""
        success = test_instance.run_complete_test(cleanup_on_success=True)
        assert success, "Complete pipeline test failed"
    
    @pytest.mark.benchmark
    def test_performance_benchmarks(self, test_instance):
        """Test performance benchmarks (requires complete pipeline)."""
        # Run all prerequisite phases
        phases = [
            test_instance.test_download_phase,
            test_instance.test_preprocessing_phase,
            test_instance.test_dataset_loading_phase,
            test_instance.test_dataloader_phase
        ]
        
        for phase in phases:
            if not phase():
                pytest.skip("Previous phase failed, skipping performance benchmarks")
        
        # Test performance
        success = test_instance.test_performance_benchmarks()
        assert success, "Performance benchmarks failed"
