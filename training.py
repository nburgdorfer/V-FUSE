import os
import sys
import argparse
from random import seed
import numpy as np
import torch

from src.config import load_config, load_scene_list, load_invalid_frames
from src.vfuse import VFUSE

# argument parsing
parser = argparse.ArgumentParser(description="V-FUSE Network training.")
parser.add_argument("--config_path", type=str, help="Configuration path.")
parser.add_argument("--dataset", type=str, help='Current dataset being used.', choices=["scannet", "replica", "dtu", "tnt"], required=True)
ARGS = parser.parse_args()

def main():
    #### Load Configuration ####
    cfg = load_config(os.path.join(ARGS.config_path, f"{ARGS.dataset}.yaml"))
    cfg["mode"] = "training"

    # set random seed
    torch.manual_seed(cfg["seed"])
    seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    #### Load Scene Lists ####
    ts = load_scene_list(os.path.join(cfg["scene_list_path"], "training.txt"))
    vs = load_scene_list(os.path.join(cfg["scene_list_path"], "validation.txt"))

    #### TRAINING ####
    pipeline = VFUSE(cfg, training_scenes=ts, validation_scenes=vs)
    pipeline.training()

if __name__ == '__main__':
    main()
