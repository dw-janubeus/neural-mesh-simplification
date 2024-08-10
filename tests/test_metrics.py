import numpy as np
import trimesh
from metrics.chamfer_distance import chamfer_distance


def create_cube_mesh(scale=1.0):
    vertices = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        * scale
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ]
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def test_chamfer_distance_identical_meshes():
    mesh1 = create_cube_mesh()
    mesh2 = create_cube_mesh()

    dist = chamfer_distance(mesh1, mesh2)

    assert np.isclose(
        dist, 0.0
    ), f"Chamfer distance for identical meshes should be 0, got {dist}"


def test_chamfer_distance_different_meshes():
    mesh1 = create_cube_mesh()
    mesh2 = create_cube_mesh(scale=2.0)  # Scale the second cube to be twice as large

    dist = chamfer_distance(mesh1, mesh2)

    assert (
        dist > 0
    ), f"Chamfer distance for different meshes should be greater than 0, got {dist}"
