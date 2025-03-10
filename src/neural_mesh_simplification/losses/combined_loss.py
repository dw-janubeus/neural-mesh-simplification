import torch.nn as nn
from torch import device

from . import (
    ProbabilisticChamferDistanceLoss,
    ProbabilisticSurfaceDistanceLoss,
    TriangleCollisionLoss,
    EdgeCrossingLoss,
    OverlappingTrianglesLoss,
)


class CombinedMeshSimplificationLoss(nn.Module):
    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_e: float = 1.0,
        lambda_o: float = 1.0,
        device=device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.prob_chamfer_loss = ProbabilisticChamferDistanceLoss().to(self.device)
        self.prob_surface_loss = ProbabilisticSurfaceDistanceLoss().to(self.device)
        self.collision_loss = TriangleCollisionLoss().to(self.device)
        self.edge_crossing_loss = EdgeCrossingLoss().to(self.device)
        self.overlapping_triangles_loss = OverlappingTrianglesLoss().to(self.device)
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_o = lambda_o

    def forward(self, original_data, simplified_data):
        original_x = (
            original_data["pos"] if "pos" in original_data else original_data["x"]
        ).to(self.device)
        original_face = original_data["face"].to(self.device)

        sampled_vertices = simplified_data["sampled_vertices"].to(self.device)
        sampled_probs = simplified_data["sampled_probs"].to(self.device)
        sampled_faces = simplified_data["simplified_faces"].to(self.device)
        face_probs = simplified_data["face_probs"].to(self.device)

        chamfer_loss = self.prob_chamfer_loss(
            original_x, sampled_vertices, sampled_probs
        )

        del sampled_probs

        surface_loss = self.prob_surface_loss(
            original_x,
            original_face,
            sampled_vertices,
            sampled_faces,
            face_probs,
        )

        del original_x
        del original_face

        collision_loss = self.collision_loss(
            sampled_vertices,
            sampled_faces,
            face_probs,
        )
        edge_crossing_loss = self.edge_crossing_loss(
            sampled_vertices, sampled_faces, face_probs
        )

        del face_probs

        overlapping_triangles_loss = self.overlapping_triangles_loss(
            sampled_vertices, sampled_faces
        )

        del sampled_vertices
        del sampled_faces

        total_loss = (
            chamfer_loss
            + surface_loss
            + self.lambda_c * collision_loss
            + self.lambda_e * edge_crossing_loss
            + self.lambda_o * overlapping_triangles_loss
        )

        return total_loss
