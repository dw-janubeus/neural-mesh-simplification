import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data

from ..models import PointSampler, EdgePredictor, FaceClassifier


class NeuralMeshSimplification(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        edge_hidden_dim,  # Separate hidden dim for edge predictor
        num_layers,
        k,
        edge_k,
        target_ratio,
        device=torch.device("cpu"),
    ):
        super(NeuralMeshSimplification, self).__init__()
        self.device = device
        self.point_sampler = PointSampler(input_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.edge_predictor = EdgePredictor(
            input_dim,
            hidden_channels=edge_hidden_dim,
            k=edge_k,
        ).to(self.device)
        self.face_classifier = FaceClassifier(input_dim, hidden_dim, num_layers, k).to(
            self.device
        )
        self.k = k
        self.target_ratio = target_ratio

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        sampled_indices, sampled_probs = self.sample_points(data)

        sampled_x = x[sampled_indices].to(self.device)
        sampled_pos = (
            data.pos[sampled_indices]
            if hasattr(data, "pos") and data.pos is not None
            else sampled_x
        ).to(self.device)

        sampled_vertices = sampled_pos  # Use sampled_pos directly as vertices

        # Update edge_index to reflect the new indices
        sampled_edge_index, _ = torch_geometric.utils.subgraph(
            sampled_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )

        # Predict edges
        sampled_edge_index = sampled_edge_index.to(self.device)
        edge_index_pred, edge_probs = self.edge_predictor(sampled_x, sampled_edge_index)

        # Generate candidate triangles
        candidate_triangles, triangle_probs = self.generate_candidate_triangles(
            edge_index_pred, edge_probs
        )

        # Classify faces
        if candidate_triangles.shape[0] > 0:
            # Create triangle features by averaging vertex features
            triangle_features = torch.zeros(
                (candidate_triangles.shape[0], sampled_x.shape[1]),
                device=self.device,
            )
            for i in range(3):
                triangle_features += sampled_x[candidate_triangles[:, i]]
            triangle_features /= 3

            # Calculate triangle centers
            triangle_centers = torch.zeros(
                (candidate_triangles.shape[0], sampled_pos.shape[1]),
                device=self.device,
            )
            for i in range(3):
                triangle_centers += sampled_pos[candidate_triangles[:, i]]
            triangle_centers /= 3

            face_probs = self.face_classifier(
                triangle_features, triangle_centers, batch=None
            )
        else:
            face_probs = torch.empty(0, device=self.device)

        if candidate_triangles.shape[0] == 0:
            simplified_faces = torch.empty((0, 3), dtype=torch.long, device=self.device)
        else:
            threshold = torch.quantile(
                face_probs, 1 - self.target_ratio
            )  # Use a dynamic threshold
            simplified_faces = candidate_triangles[face_probs > threshold]

        return {
            "sampled_indices": sampled_indices,
            "sampled_probs": sampled_probs,
            "sampled_vertices": sampled_vertices,
            "edge_index": edge_index_pred,
            "edge_probs": edge_probs,
            "candidate_triangles": candidate_triangles,
            "triangle_probs": triangle_probs,
            "face_probs": face_probs,
            "simplified_faces": simplified_faces,
        }

    def sample_points(self, data: Data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        target_nodes = min(
            max(int(self.target_ratio * num_nodes), 1),
            num_nodes,
        )

        # Sample points
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        sampled_probs = self.point_sampler(x, edge_index)
        sampled_indices = self.point_sampler.sample(
            sampled_probs, num_samples=target_nodes
        )

        return sampled_indices, sampled_probs[sampled_indices]

    def generate_candidate_triangles(self, edge_index, edge_probs):

        # Handle the case when edge_index is empty
        if edge_index.numel() == 0:
            return (
                torch.empty((0, 3), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
            )

        num_nodes = edge_index.max().item() + 1

        # Create an adjacency matrix from the edge index
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)

        # Check if edge_probs is a tuple or a tensor
        if isinstance(edge_probs, tuple):
            edge_indices, edge_values = edge_probs
            adj_matrix[edge_indices[0], edge_indices[1]] = edge_values
        else:
            adj_matrix[edge_index[0], edge_index[1]] = edge_probs

        # Adjust k based on the number of nodes
        k = min(self.k, num_nodes - 1)

        # Find k-nearest neighbors for each node
        _, knn_indices = torch.topk(adj_matrix, k=k, dim=1)

        # Generate candidate triangles
        triangles = []
        triangle_probs = []

        for i in range(num_nodes):
            neighbors = knn_indices[i]
            for j in range(k):
                for l in range(j + 1, k):
                    n1, n2 = neighbors[j], neighbors[l]
                    if adj_matrix[n1, n2] > 0:  # Check if the third edge exists
                        triangle = torch.tensor([i, n1, n2], device=self.device)
                        triangles.append(triangle)

                        # Calculate triangle probability
                        prob = (
                            adj_matrix[i, n1] * adj_matrix[i, n2] * adj_matrix[n1, n2]
                        ) ** (1 / 3)
                        triangle_probs.append(prob)

        if triangles:
            triangles = torch.stack(triangles)
            triangle_probs = torch.tensor(triangle_probs, device=self.device)
        else:
            triangles = torch.empty((0, 3), dtype=torch.long, device=self.device)
            triangle_probs = torch.empty(0, device=self.device)

        return triangles, triangle_probs
