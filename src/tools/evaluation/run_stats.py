import os
import time
import sys
import math
import argparse
from random import randint, seed
import cv2
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision.transforms as T
import matplotlib.pyplot as plt

from utils import *
from data_utils import *
from models import *
from loss_functions import * 

torch.manual_seed(5)
seed(5)
np.random.seed(5)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# argument parsing
parse = argparse.ArgumentParser(description="V-FUSE statistics.")
parse.add_argument("-r", "--data_root_dir", default="/data/dtu", type=str, help="Root path to dataset.")
parse.add_argument("-v", "--view_num", default=11, type=int, help="Number of images (1 ref image and view_num - 1 view images).")
parse.add_argument("-j", "--num_workers", default=4, type=int, help="Number of data loading workers.")
parse.add_argument("-d", "--max_d", default=8, type=int, help="Maximum depth step when training.")
parse.add_argument("-b", "--batch_size", default=2, type=int, help="Model batch size.")
parse.add_argument("-s", "--scale", default=1, type=float, help="Scale for rescaling inputs.")
parse.add_argument("-g", "--load_gt", action="store_true", help="If present, ground truth data will be loaded for the test set.")
ARGS = parse.parse_args()
device = torch.device("cpu")

def load_data(scan, mode):
    transform = T.Compose([T.ToTensor()])
    load_set = DtuDataLoader(ARGS.data_root_dir, ARGS.view_num, ARGS.max_d, transform=transform, mode=mode, device=device, scale=ARGS.scale, load_gt=ARGS.load_gt, eval_scan=scan)
    sample_size = load_set.__len__()
    print ('Sample number: ', sample_size)
    data_loader = torch.utils.data.DataLoader(load_set, batch_size=ARGS.batch_size, shuffle=False, num_workers=ARGS.num_workers, pin_memory=True)
    return data_loader

def compute_stats(data_loader, scan, mode):
    stats_path = os.path.join(ARGS.data_root_dir, "Stats_{}".format(mode))
    scan_path = os.path.join(stats_path, "scan{:03d}".format(scan))
    stats_file = os.path.join(scan_path, "stats.txt")

    # create stats dir
    if not os.path.exists(scan_path):
        os.makedirs(scan_path)

    """ compute statistics of training data """
    ########## run statistic computation ##########
    with tqdm(data_loader, desc="Computing {} Data Statistics".format(mode), unit="batches") as loader:
        for i, data in enumerate(loader):
            depths, _, _, _, _, gt_depth, ref_view_num = data

            depths = depths.to(device)
            gt_depth = gt_depth.to(device)

            # test input bias
            train_abs_error(depths, gt_depth, scan_path, stats_file, ref_view_num, device)
        avg_train_stats(stats_path, stats_file, scan)

    return

def main():
    """ program entrance """
    ########## Compute statistics for evaluation data ##########
    eval_set = np.array([1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118])

    for e in eval_set:
        eval_loader = load_data(e, "evaluation")
        compute_stats(eval_loader, e, "Eval")


    #   ########## Compute statistics for training data ##########
    #   train_set = np.array([2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
    #               45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
    #               74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    #               101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
    #               121, 122, 123, 124, 125, 126, 127, 128])

    #   for t in train_set:
    #       train_loader = load_data(t, "evaluation")
    #       compute_stats(train_loader, t, "Train")


    ########## Compute statistics for validation data ##########
    #   val_set = np.array([3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117])

    #   for v in val_set:
    #       valid_loader = load_data(v, "evaluation")
    #       compute_stats(valid_loader, v, "Valid")


if __name__ == '__main__':
    main()
