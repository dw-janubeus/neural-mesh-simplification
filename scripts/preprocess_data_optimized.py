#!/usr/bin/env python3
"""
Optimized data preprocessing pipeline for neural mesh simplification.

This script pre-processes all meshes and saves them as PyTorch tensors with
pre-computed graph structures to eliminate real-time processing bottlenecks.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data
from tqdm import tqdm

from neural_mesh_simplification.utils.mesh_operations import build_graph_from_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedMeshPreprocessor:
    """Optimized mesh preprocessor that caches expensive operations."""
    
    def __init__(self, input_dir: str, output_dir: str, max_vertices: int = 10000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_vertices = max_vertices
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tensors").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
    def _get_mesh_files(self) -> List[Path]:
        """Get all mesh files from input directory."""
        supported_extensions = {'.ply', '.obj', '.stl'}
        mesh_files = []
        
        for ext in supported_extensions:
            mesh_files.extend(self.input_dir.glob(f"*{ext}"))
            
        return sorted(mesh_files)
    
    def _load_and_validate_mesh(self, file_path: Path) -> Optional[trimesh.Trimesh]:
        """Load and validate a single mesh."""
        try:
            mesh = trimesh.load(str(file_path))
            
            # Handle mesh sequences (take first if multiple)
            if isinstance(mesh, trimesh.Scene):
                geometries = [geom for geom in mesh.geometry.values() 
                            if isinstance(geom, trimesh.Trimesh)]
                if not geometries:
                    return None
                mesh = geometries[0]
            elif not isinstance(mesh, trimesh.Trimesh):
                return None
            
            # Basic validation
            if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
                logger.debug(f"Skipping {file_path.name}: too few vertices/faces")
                return None
                
            if len(mesh.vertices) > self.max_vertices:
                logger.debug(f"Skipping {file_path.name}: too many vertices ({len(mesh.vertices)})")
                return None
                
            # Check if mesh is valid
            if not mesh.is_valid:
                mesh.fix_normals()
                if not mesh.is_valid:
                    logger.debug(f"Skipping {file_path.name}: invalid mesh")
                    return None
                    
            return mesh
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            return None
    
    def _preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply standard preprocessing to mesh."""
        # Remove duplicated vertices and faces
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Center the mesh
        mesh.vertices -= mesh.vertices.mean(axis=0)
        
        # Scale to unit cube
        max_dim = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        if max_dim > 0:
            mesh.vertices /= max_dim
            
        return mesh
    
    def _mesh_to_tensor_data(self, mesh: trimesh.Trimesh, file_id: str) -> Data:
        """Convert mesh to PyTorch Geometric Data object with pre-computed graph."""
        # Convert to tensors
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        
        # Build graph structure (expensive operation done once)
        try:
            G = build_graph_from_mesh(mesh)
            edge_list = list(G.edges())
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                # Fallback: create edges from faces
                edge_set = set()
                for face in mesh.faces:
                    for i in range(3):
                        edge = tuple(sorted([face[i], face[(i+1)%3]]))
                        edge_set.add(edge)
                edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
                
        except Exception as e:
            logger.warning(f"Graph construction failed for {file_id}: {e}")
            # Fallback: create edges from faces
            edge_set = set()
            for face in mesh.faces:
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    edge_set.add(edge)
            edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
        
        # Create PyG Data object
        data = Data(
            x=vertices,  # Node features (coordinates)
            pos=vertices,  # Node positions
            edge_index=edge_index,  # Graph connectivity
            face=faces.t().contiguous(),  # Face connectivity
            num_nodes=len(vertices),
            file_id=file_id,
        )
        
        return data
    
    def _save_tensor_data(self, data: Data, file_id: str) -> Dict:
        """Save tensor data and return metadata."""
        # Save tensor data
        tensor_path = self.output_dir / "tensors" / f"{file_id}.pt"
        torch.save(data, tensor_path)
        
        # Create metadata
        metadata = {
            'file_id': file_id,
            'num_vertices': int(data.num_nodes),
            'num_faces': int(data.face.shape[1]),
            'num_edges': int(data.edge_index.shape[1]),
            'tensor_path': str(tensor_path.relative_to(self.output_dir)),
            'vertex_bounds': {
                'min': data.pos.min(dim=0)[0].tolist(),
                'max': data.pos.max(dim=0)[0].tolist(),
            }
        }
        
        return metadata
    
    def preprocess_dataset(self) -> None:
        """Main preprocessing pipeline."""
        mesh_files = self._get_mesh_files()
        logger.info(f"Found {len(mesh_files)} mesh files to process")
        
        if not mesh_files:
            logger.error(f"No mesh files found in {self.input_dir}")
            return
        
        metadata_list = []
        processed_count = 0
        skipped_count = 0
        
        for file_path in tqdm(mesh_files, desc="Processing meshes"):
            file_id = file_path.stem  # filename without extension
            logger.debug(f"Processing file: {file_path.name}")
            
            # Load and validate mesh
            mesh = self._load_and_validate_mesh(file_path)
            if mesh is None:
                logger.debug(f"Skipped {file_path.name}: failed validation")
                skipped_count += 1
                continue
            
            logger.debug(f"Loaded mesh {file_path.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            try:
                # Preprocess mesh
                logger.debug(f"Preprocessing mesh {file_path.name}")
                mesh = self._preprocess_mesh(mesh)
                
                # Convert to tensor data
                logger.debug(f"Converting mesh {file_path.name} to tensor")
                data = self._mesh_to_tensor_data(mesh, file_id)
                
                # Save tensor data and collect metadata
                logger.debug(f"Saving tensor data for {file_path.name}")
                metadata = self._save_tensor_data(data, file_id)
                metadata['original_file'] = str(file_path.name)
                metadata_list.append(metadata)
                
                logger.debug(f"Successfully processed {file_path.name}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                skipped_count += 1
                continue
        
        # Save dataset metadata
        dataset_metadata = {
            'total_files': len(mesh_files),
            'processed_files': processed_count,
            'skipped_files': skipped_count,
            'max_vertices': self.max_vertices,
            'files': metadata_list
        }
        
        metadata_path = self.output_dir / "metadata" / "dataset_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(dataset_metadata, f)
        
        # Save file index for quick lookup
        file_index = {item['file_id']: item for item in metadata_list}
        index_path = self.output_dir / "metadata" / "file_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(file_index, f)
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Processed: {processed_count} files")
        logger.info(f"  Skipped: {skipped_count} files")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimized mesh preprocessing pipeline")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="data/raw",
        help="Input directory containing mesh files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/processed_optimized",
        help="Output directory for processed tensors"
    )
    parser.add_argument(
        "--max-vertices", 
        type=int, 
        default=10000,
        help="Maximum number of vertices per mesh"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    preprocessor = OptimizedMeshPreprocessor(
        args.input_dir, 
        args.output_dir, 
        args.max_vertices
    )
    
    preprocessor.preprocess_dataset()


if __name__ == "__main__":
    main()
