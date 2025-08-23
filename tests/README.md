# Neural Mesh Simplification Tests

This directory contains comprehensive tests for the neural mesh simplification project, including unit tests, integration tests, and end-to-end pipeline validation.

## Test Structure

### Unit Tests
- `test_*.py` - Individual component tests (dataset, models, losses, metrics)
- Focus on isolated functionality testing
- Fast execution, suitable for CI/CD

### Integration Tests
- `test_pipeline_integration.py` - Pytest-compatible pipeline tests
- `test_end_to_end_pipeline.py` - Comprehensive pipeline validation script

### End-to-End Pipeline Test

The main pipeline test (`test_end_to_end_pipeline.py`) provides comprehensive validation of the complete data flow:

1. **Download Phase**: Downloads mesh data from HuggingFace Hub
2. **Preprocessing Phase**: Converts raw meshes to optimized PyTorch tensors
3. **Dataset Loading Phase**: Tests the OptimizedMeshSimplificationDataset
4. **DataLoader Phase**: Validates batch processing and iteration
5. **Performance Benchmarks**: Measures loading speeds and memory usage

## Running Tests

### Quick Start - Run Pipeline Test

```bash
# Run the complete pipeline test with small dataset (recommended)
python scripts/run_pipeline_test.py

# Or run directly
python tests/test_end_to_end_pipeline.py
```

### Advanced Options

```bash
# Run with debug logging
python tests/test_end_to_end_pipeline.py --debug

# Test with larger dataset subset
python tests/test_end_to_end_pipeline.py --large-dataset

# Keep test files for inspection
python tests/test_end_to_end_pipeline.py --no-cleanup

# Use specific test directory
python tests/test_end_to_end_pipeline.py --test-dir /tmp/my_test
```

### Using Pytest

```bash
# Run all unit tests
pytest

# Run only integration tests
pytest tests/test_pipeline_integration.py

# Skip slow tests
pytest -m "not slow"

# Run only benchmark tests
pytest -m benchmark

# Run with verbose output
pytest -v
```

## Test Markers

The following pytest markers are available:

- `slow` - Tests that take longer to run (>30 seconds)
- `benchmark` - Performance benchmark tests
- `integration` - Integration tests requiring multiple components
- `trimesh` - Tests specifically for trimesh functionality

## Expected Results

### Successful Test Output

When all tests pass, you should see:
- ✓ All 5 phases completed successfully
- Download of 50-200 mesh files (small subset)
- Processing to optimized tensor format
- Dataset loading with proper statistics
- DataLoader iteration with various batch sizes
- Performance benchmarks showing loading times

### Sample Performance Metrics

On a typical system, expect:
- Sample loading: 1-5ms per sample
- Batch processing: 5-20ms per batch
- GPU memory usage: 10-100MB per batch (depending on batch size)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Failures**: Check internet connection and HuggingFace Hub access
   - May require `huggingface_hub` login for some datasets

3. **Memory Issues**: Use smaller batch sizes or enable `--small-subset` mode

4. **CUDA Issues**: Tests will automatically fall back to CPU if CUDA unavailable

### Debug Mode

Enable debug mode for detailed logging:
```bash
python tests/test_end_to_end_pipeline.py --debug
```

This provides:
- Detailed logging of each operation
- Individual sample validation results
- Memory usage tracking
- Full stack traces for errors

### Test Directory Structure

The test creates the following structure:
```
<test_dir>/
├── raw/                    # Downloaded mesh files (.ply)
├── processed/              # Optimized tensor data
│   ├── tensors/           # Individual .pt files
│   └── metadata/          # Dataset metadata and indices
└── pipeline_test.log      # Detailed test log
```

## Hardware Requirements

### Minimum Requirements
- 4GB RAM
- 2GB free disk space
- Python 3.8+

### Recommended Requirements
- 8GB+ RAM
- CUDA-compatible GPU (optional but recommended)
- 5GB+ free disk space
- Fast internet connection for downloads

## CI/CD Integration

For automated testing in CI/CD pipelines:

```bash
# Fast unit tests only
pytest -m "not slow and not integration"

# Full test suite (allow extra time)
pytest --maxfail=1

# Pipeline validation (in dedicated job)
python tests/test_end_to_end_pipeline.py --debug --no-cleanup
```

The test preserves failed test directories for debugging, and provides detailed logs for CI analysis.
