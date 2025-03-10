import argparse
import logging
import os

from neural_mesh_simplification.trainer import Trainer

script_dir = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(script_dir, "../configs/default.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Neural Mesh Simplification model."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=False,
        help="Path to the training data directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug logs")
    parser.add_argument(
        "--monitor", action="store_true", help="Monitor CPU and memory usage"
    )
    return parser.parse_args()


def load_config(config_path):
    import yaml

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    args = parse_args()

    config_path = args.config if args.config else default_config_path

    config = load_config(config_path)

    if args.data_path:
        config["data"]["data_dir"] = args.data_path

    if args.checkpoint_dir:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.monitor:
        config["monitor_resources"] = True

    trainer = Trainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    except Exception as e:
        trainer.handle_error(e)
        trainer.save_training_state(
            os.path.join(config["training"]["checkpoint_dir"], "training_state.pth")
        )


if __name__ == "__main__":
    main()
