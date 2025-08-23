#!/usr/bin/env python3
"""
Convenience script to run the end-to-end pipeline test.

This script provides an easy way to run the comprehensive pipeline tests
with common configurations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import and run the test
from tests.test_end_to_end_pipeline import main

if __name__ == "__main__":
    print("🚀 Starting Neural Mesh Simplification Pipeline Test")
    print("=" * 60)
    print("This will test the complete data pipeline:")
    print("1. Download dataset from HuggingFace (small subset)")
    print("2. Preprocess meshes to optimized tensors")
    print("3. Load data using OptimizedMeshSimplificationDataset")
    print("4. Test DataLoader iteration and batching") 
    print("5. Run performance benchmarks")
    print("=" * 60)
    
    # Check dependencies
    try:
        import torch
        import trimesh
        import numpy as np
        from huggingface_hub import snapshot_download
        print("✓ All required dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install torch trimesh numpy huggingface_hub")
        sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available - using CPU")
    
    print("\nStarting test...\n")
    
    # Run the main test function
    main()
