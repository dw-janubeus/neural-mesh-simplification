import argparse

from trimesh import Trimesh

from neural_mesh_simplification import (
    NeuralMeshSimplifier,
)  # Assuming the model class is named MeshSimplifier
from neural_mesh_simplification.data.dataset import load_mesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplify a 3D mesh using a trained model."
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to the input mesh file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Where to save the simplified mesh.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        required=False,
        default=128,
        help="Feature dimension for point sampler and face classifier",
    )
    parser.add_argument(
        "--edge-hidden-dim",
        type=int,
        required=False,
        default=64,
        help="Feature dimension for edge predictor",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (`cpu` or `cuda`).",
    )

    return parser.parse_args()


def simplify_mesh(
    input_file: str,
    output_file: str,
    model_checkpoint: str,
    hidden_dim: int,
    edge_hidden_dim: int,
    device="cpu",
):
    """
    Simplifies a 3D mesh using a trained model.

    Args:
        input_file (str): Path to the high-resolution input `.obj` file.
        model_checkpoint (str): Path to the trained model checkpoint.
        hidden_dim (int): Feature dimension for point sampler and face classifier
        edge_hidden_dim (int): Feature dimension for edge predictor
        device (str): Device to use for inference (`cpu` or `cuda`).
    """
    # Load the trained model
    print(f"Loading model from {model_checkpoint}...")
    simplifier = NeuralMeshSimplifier.using_model(
        model_checkpoint,
        hidden_dim=hidden_dim,
        edge_hidden_dim=edge_hidden_dim,
        map_location=device,
    )
    simplifier.model.to(device)
    simplifier.model.eval()

    # Load the input mesh
    print(f"Loading input mesh from {input_file}...")
    original_mesh = load_mesh(input_file)

    if not isinstance(original_mesh, Trimesh):
        raise ValueError("Invalid format for input mesh.")

    simplified_mesh = simplifier.simplify(original_mesh)

    # Save the simplified mesh
    print(f"Saving simplified mesh to {output_file}...")
    simplified_mesh.export(output_file)
    print("Simplification complete.")


def load_config(config_path):
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    args = parse_args()

    simplify_mesh(
        input_file=args.input_file,
        output_file=args.output_file,
        model_checkpoint=args.model_checkpoint,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        device=args.device,
    )


if __name__ == "__main__":
    main()
