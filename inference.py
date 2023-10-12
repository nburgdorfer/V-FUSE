import os
import sys
import argparse
from random import seed
import numpy as np
import torch

from src.config import load_config, load_scene_list, load_invalid_frames
from src.vfuse import VFUSE
from src.tools.consensus_filtering import consensus_filtering
from src.evaluation.dtu.eval import dtu_point_eval

# argument parsing
parser = argparse.ArgumentParser(description="V-FUSE Network inference.")
parser.add_argument("--config_path", type=str, help="Configuration path.")
parser.add_argument("--dataset", type=str, help='Current dataset being used.', choices=["blendedmvs", "dtu", "tnt"], required=True)
ARGS = parser.parse_args()

def main():
    #### Load Configuration ####
    cfg = load_config(os.path.join(ARGS.config_path, f"{ARGS.dataset}.yaml"))
    cfg["mode"] = "inference"

    # set random seed
    torch.manual_seed(cfg["seed"])
    seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    #### Load Scene Lists ####
    scenes_list = os.path.join(cfg["scene_list_path"], "inference.txt")
    with open(scenes_list,'r') as sf:
        scenes = sf.readlines()
        scenes = [s.strip() for s in scenes]


    # metric used for evaluation
    acc_sum = 0.0
    comp_sum = 0.0
    ovr_sum = 0.0

    for scene in scenes:
        print(f"\n----Running V-FUSE on {scene}----")
        #### INFERENCE ####
        pipeline = VFUSE(cfg, inference_scene=[scene])
        pipeline.inference()
        consensus_filtering(cfg, scene, pipeline.dataset.get_cluster_list_file(scene), f"{scene}.ply")


        #### EVALUATION ####
        if cfg["eval"]["run_eval"]:
            if (ARGS.dataset == "dtu"):
                acc, comp, ovr, prec, rec = dtu_point_eval(cfg, scene)

                acc_sum += acc
                comp_sum += comp
                ovr_sum += ovr


    if cfg["eval"]["run_eval"]:
        print("\n---Total Results---")
        print(f"Acc: {acc_sum/len(scenes):0.3f}")
        print(f"Comp: {comp_sum/len(scenes):0.3f}")
        print(f"Ovr: {ovr_sum/len(scenes):0.3f}")


if __name__ == '__main__':
    main()
