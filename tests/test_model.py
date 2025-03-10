import pytest
import torch
import torch_geometric.utils
from torch_geometric.data import Data

from neural_mesh_simplification.models import NeuralMeshSimplification


@pytest.fixture
def sample_data() -> Data:
    num_nodes = 10
    x = torch.randn(num_nodes, 3)
    # Create a more densely connected edge index
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    pos = torch.randn(num_nodes, 3)
    return Data(x=x, edge_index=edge_index, pos=pos)


def test_neural_mesh_simplification_forward(sample_data: Data):
    # Set a fixed random seed for reproducibility
    torch.manual_seed(42)

    model = NeuralMeshSimplification(
        input_dim=3,
        hidden_dim=64,
        edge_hidden_dim=64,
        num_layers=3,
        k=3,  # Reduce k to avoid too many edges in the test
        edge_k=15,
        target_ratio=0.5,  # Ensure we sample roughly half the vertices
    )

    # First test point sampling
    sampled_indices, sampled_probs = model.sample_points(sample_data)
    assert sampled_indices.numel() > 0, "No points were sampled"
    assert sampled_indices.numel() <= sample_data.num_nodes, "Too many points sampled"

    # Get the subgraph for sampled points
    sampled_edge_index, _ = torch_geometric.utils.subgraph(
        sampled_indices,
        sample_data.edge_index,
        relabel_nodes=True,
        num_nodes=sample_data.num_nodes,
    )
    assert sampled_edge_index.numel() > 0, "No edges in sampled subgraph"

    # Now test the full forward pass
    output = model(sample_data)

    # Add assertions to check the output structure and shapes
    assert isinstance(output, dict)
    assert "sampled_indices" in output
    assert "sampled_probs" in output
    assert "sampled_vertices" in output
    assert "edge_index" in output
    assert "edge_probs" in output
    assert "candidate_triangles" in output
    assert "triangle_probs" in output
    assert "face_probs" in output
    assert "simplified_faces" in output

    # Check shapes
    assert output["sampled_indices"].dim() == 1
    # sampled_probs should match the number of sampled vertices
    assert output["sampled_probs"].shape == output["sampled_indices"].shape
    assert output["sampled_vertices"].shape[1] == 3  # 3D coordinates

    if output["edge_index"].numel() > 0:  # Only check if we have edges
        assert output["edge_index"].shape[0] == 2  # Source and target nodes
        assert (
            len(output["edge_probs"]) == output["edge_index"].shape[1]
        )  # One prob per edge

        # Check that edge indices are valid
        num_sampled_vertices = output["sampled_vertices"].shape[0]
        assert torch.all(output["edge_index"] >= 0)
        assert torch.all(output["edge_index"] < num_sampled_vertices)

    if output["candidate_triangles"].numel() > 0:  # Only check if we have triangles
        assert output["candidate_triangles"].shape[1] == 3  # Triangle indices
        assert len(output["triangle_probs"]) == len(output["candidate_triangles"])
        assert len(output["face_probs"]) == len(output["candidate_triangles"])

    # Additional checks
    assert output["sampled_indices"].shape[0] <= sample_data.num_nodes
    assert output["sampled_vertices"].shape[0] == output["sampled_indices"].shape[0]

    # Check that sampled_vertices correspond to a subset of original vertices
    original_vertices = sample_data.pos
    sampled_vertices = output["sampled_vertices"]

    # For each sampled vertex, check if it exists in original vertices
    for sv in sampled_vertices:
        # Check if this vertex exists in original vertices (within numerical precision)
        exists = torch.any(torch.all(torch.abs(original_vertices - sv) < 1e-6, dim=1))
        assert exists, "Sampled vertex not found in original vertices"

    # Check that simplified_faces only contain valid indices if not empty
    if output["simplified_faces"].numel() > 0:
        max_index = output["sampled_vertices"].shape[0] - 1
        assert torch.all(output["simplified_faces"] >= 0)
        assert torch.all(output["simplified_faces"] <= max_index)

    # Check the relationship between face_probs and simplified_faces
    if output["face_probs"].numel() > 0:
        assert output["simplified_faces"].shape[0] <= output["face_probs"].shape[0]


def test_generate_candidate_triangles():
    model = NeuralMeshSimplification(
        input_dim=3,
        hidden_dim=64,
        edge_hidden_dim=64,
        num_layers=3,
        k=5,
        edge_k=15,
        target_ratio=0.5,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    edge_probs = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.7, 0.7])

    triangles, triangle_probs = model.generate_candidate_triangles(
        edge_index, edge_probs
    )

    assert triangles.shape[1] == 3
    assert triangle_probs.shape[0] == triangles.shape[0]
    assert torch.all(triangles >= 0)
    assert torch.all(triangles < edge_index.max() + 1)
    assert torch.all(triangle_probs >= 0) and torch.all(triangle_probs <= 1)

    max_possible_triangles = edge_index.max().item() + 1  # num_nodes
    max_possible_triangles = (
        max_possible_triangles
        * (max_possible_triangles - 1)
        * (max_possible_triangles - 2)
        // 6
    )
    assert triangles.shape[0] <= max_possible_triangles
