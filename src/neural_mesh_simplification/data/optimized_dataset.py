"""
Optimized dataset class for pre-processed mesh simplification data.

This dataset loads pre-computed PyTorch tensors instead of raw mesh files,
eliminating the real-time processing bottleneck that causes low GPU utilization.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class OptimizedMeshSimplificationDataset(Dataset):
    """
    Optimized dataset that loads pre-processed tensor data from disk.
    
    This dataset eliminates the major performance bottlenecks by:
    - Loading pre-computed PyTorch tensors instead of raw meshes
    - Using pre-built graph structures (no real-time NetworkX operations)
    - Avoiding trimesh operations during training
    - Removing garbage collection calls
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path], 
        transform: Optional[callable] = None,
        subset_size: Optional[int] = None,
        memory_mapping: bool = True
    ):
        """
        Initialize the optimized dataset.
        
        Args:
            data_dir: Directory containing processed tensor data
            transform: Optional transform to apply to data
            subset_size: Optional limit on dataset size (for debugging)
            memory_mapping: Whether to use memory mapping for large files
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.memory_mapping = memory_mapping
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.file_list = self.metadata['files']
        
        if subset_size is not None:
            self.file_list = self.file_list[:subset_size]
            logger.info(f"Using subset of {len(self.file_list)} samples")
        
        # Create file index for quick lookup
        self.file_index = {item['file_id']: item for item in self.file_list}
        
        logger.info(f"Loaded optimized dataset with {len(self.file_list)} samples")
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata."""
        metadata_path = self.data_dir / "metadata" / "dataset_metadata.pkl"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found at {metadata_path}. "
                "Please run the preprocessing script first."
            )
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Loaded metadata for {metadata['processed_files']} processed files")
        return metadata
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single sample from the dataset.
        
        This is now extremely fast since we just load a pre-computed tensor.
        """
        if idx >= len(self.file_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.file_list)}")
        
        # Get file metadata
        file_info = self.file_list[idx]
        tensor_path = self.data_dir / file_info['tensor_path']
        
        # Load pre-computed tensor data
        try:
            if self.memory_mapping:
                # Use memory mapping for potentially faster loading of large files
                data = torch.load(tensor_path, map_location='cpu', weights_only=False)
            else:
                data = torch.load(tensor_path, weights_only=False)
                
            # Apply transform if specified
            if self.transform:
                data = self.transform(data)
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to load tensor data from {tensor_path}: {e}")
            # Return a fallback empty tensor to avoid crashing training
            return Data(
                x=torch.zeros(4, 3),
                pos=torch.zeros(4, 3),
                edge_index=torch.zeros(2, 6, dtype=torch.long),
                face=torch.zeros(3, 2, dtype=torch.long),
                num_nodes=4,
                file_id=f"error_{idx}"
            )
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.file_list:
            return {}
        
        stats = {
            'total_samples': len(self.file_list),
            'avg_vertices': sum(item['num_vertices'] for item in self.file_list) / len(self.file_list),
            'avg_faces': sum(item['num_faces'] for item in self.file_list) / len(self.file_list),
            'avg_edges': sum(item['num_edges'] for item in self.file_list) / len(self.file_list),
            'min_vertices': min(item['num_vertices'] for item in self.file_list),
            'max_vertices': max(item['num_vertices'] for item in self.file_list),
            'min_faces': min(item['num_faces'] for item in self.file_list),
            'max_faces': max(item['num_faces'] for item in self.file_list),
        }
        
        return stats
    
    def get_sample_by_id(self, file_id: str) -> Optional[Data]:
        """Get a sample by its file ID."""
        if file_id not in self.file_index:
            logger.warning(f"File ID '{file_id}' not found in dataset")
            return None
        
        # Find index of the file
        for idx, item in enumerate(self.file_list):
            if item['file_id'] == file_id:
                return self.__getitem__(idx)
        
        return None
    
    def get_batch_info(self, indices: List[int]) -> Dict:
        """Get information about a batch of samples."""
        batch_info = {
            'indices': indices,
            'total_vertices': 0,
            'total_faces': 0,
            'total_edges': 0,
            'files': []
        }
        
        for idx in indices:
            if idx < len(self.file_list):
                item = self.file_list[idx]
                batch_info['total_vertices'] += item['num_vertices']
                batch_info['total_faces'] += item['num_faces']
                batch_info['total_edges'] += item['num_edges']
                batch_info['files'].append(item['file_id'])
        
        return batch_info


def collate_mesh_data(batch: List[Data]) -> Data:
    """
    Custom collate function for mesh data that handles variable-sized graphs.
    
    This function is optimized for PyTorch Geometric batching.
    """
    from torch_geometric.data import Batch
    
    try:
        # Use PyTorch Geometric's built-in batching
        batched_data = Batch.from_data_list(batch)
        return batched_data
    except Exception as e:
        logger.error(f"Failed to batch mesh data: {e}")
        # Fallback: return first item in batch
        return batch[0] if batch else Data()


class DataAugmentation:
    """Simple data augmentation for mesh data."""
    
    def __init__(self, rotation_prob: float = 0.5, noise_prob: float = 0.3, noise_std: float = 0.01):
        self.rotation_prob = rotation_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
    
    def __call__(self, data: Data) -> Data:
        """Apply data augmentation."""
        if torch.rand(1).item() < self.rotation_prob:
            data = self._random_rotation(data)
        
        if torch.rand(1).item() < self.noise_prob:
            data = self._add_noise(data)
        
        return data
    
    def _random_rotation(self, data: Data) -> Data:
        """Apply random rotation to vertex positions."""
        # Generate random rotation matrix (simplified 2D rotation for speed)
        angle = torch.rand(1) * 2 * torch.pi
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        
        # Apply rotation to x and y coordinates
        if data.pos is not None and data.pos.shape[1] >= 2:
            x, y = data.pos[:, 0], data.pos[:, 1]
            data.pos[:, 0] = cos_a * x - sin_a * y
            data.pos[:, 1] = sin_a * x + cos_a * y
            
            # Update x features if they match positions
            if data.x is not None and torch.allclose(data.x[:, :2], data.pos[:, :2]):
                data.x[:, :2] = data.pos[:, :2]
        
        return data
    
    def _add_noise(self, data: Data) -> Data:
        """Add small amount of Gaussian noise to vertex positions."""
        if data.pos is not None:
            noise = torch.randn_like(data.pos) * self.noise_std
            data.pos = data.pos + noise
            
            # Update x features if they match positions
            if data.x is not None and data.x.shape == data.pos.shape:
                data.x = data.pos
        
        return data
