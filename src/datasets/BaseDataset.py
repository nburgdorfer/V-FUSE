import os
import random
import numpy as np
import torch.utils.data as data
import sys
import torch

from src.utils.io import read_single_cam_sfm, read_pfm
from src.utils.common import warp
from src.utils.preprocess import scale_mvs_input, scale_image

def build_dataset(cfg, mode, scenes):
    if cfg["dataset"] == 'TNT':
        from src.datasets.TNT import TNT as Dataset
    elif cfg["dataset"] == 'BlendedMVS':
        from src.datasets.BlendedMVS import BlendedMVS as Dataset
    elif cfg["dataset"] == 'DTU':
        from src.datasets.DTU import DTU as Dataset
    else:
        raise Exception(f"Unknown Dataset {self.cfg['dataset']}")

    return Dataset(cfg, mode, scenes)

class BaseDataset(data.Dataset):
    def __init__(self, cfg, mode, scenes):
        self.cfg = cfg
        self.mode = mode
        self.data_path = self.cfg["data_path"]
        self.device = self.cfg["device"]
        self.scenes = scenes

        if self.mode == "training":
            self.num_frame = self.cfg["training"]["num_frame"]
            self.scale = self.cfg["training"]["scale"]
        else:
            self.num_frame = self.cfg["eval"]["num_frame"]
            self.scale = self.cfg["eval"]["scale"]

        self.poses = {}
        self.K = {}

        self.build_samples()

    def build_samples(self, frame_spacing):
        raise NotImplementedError()

    def load_intrinsics(self):
        raise NotImplementedError()

    def get_pose(self, pose_file):
        raise NotImplementedError()

    def get_metadata(self, pose_file):
        raise NotImplementedError()

    def get_image(self, image_file):
        raise NotImplementedError()

    def get_depth(self, depth_file):
        raise NotImplementedError()

    def compute_projection_matrix(self, K_pool, poses):
        proj_mats = np.zeros((self.num_frame, 6, 4, 4), dtype=np.float32)
        for i in range(self.num_frame):
            for j,k in enumerate(K_pool):
                K44 = np.eye(4)
                K44[:3, :3] = k
                proj_mats[i,j] = np.matmul(K44, poses[i])
        return proj_mats

    def get_intrinsic_pools(self, K):
        K_pool = np.zeros((6,3,3), dtype=np.float32)
        for i in range(6):
            K_pool[i] = K.copy()
            K_pool[i][:2, :] /= 2**i

        K_inv_pool = np.zeros((6,4,4), dtype=np.float32)
        for i,k in enumerate(K_pool):
            K44 = np.eye(4)
            K44[:3, :3] = k
            K_inv_pool[i] = np.linalg.inv(K44)

        return K_inv_pool[0], K_pool, K_inv_pool

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        sample = self.samples[idx]
        scene = sample["scene"]
        ref_index = sample["ref_index"]

        # load and compute intrinsics
        K = self.K[scene]
        depths = []
        images = []
        confs = []
        cams = []
        for i in range(len(sample["cams"])):
            images.append(self.get_image(sample["images"][i]))
            depths.append(self.get_depth(sample["depths"][i]))
            confs.append(self.get_conf(sample["confs"][i]))
            
            cam = np.zeros((2,4,4))
            cam[0] = self.get_pose(sample["cams"][i])
            cam[1,:3,:3] = K
            cam[1,3,:] = self.get_metadata(sample["cams"][i])
            cams.append(cam)
        if self.mode == "training":
            gt_depth = self.get_depth(sample["gt_depth"])

        # convert to numpy array
        depths = np.asarray(depths)
        images = np.asarray(images)
        confs = np.asarray(confs)
        cams = np.asarray(cams)

        # process input data
        depths[np.isnan(depths)] = 0.0
        confs[np.isnan(confs)] = 0.0
        confs = np.clip(confs, 0.0, 1.0)
        depths,confs,cams = scale_mvs_input(depths, confs, cams, scale=self.scale)
        if self.mode == "training":
            gt_depth = scale_image(gt_depth, scale=self.scale)

        # fix depth range and adapt depth sample number
        cams[:,1,3,1] = (cams[:,1,3,3] - cams[:,1,3,0]) / 256 

        # store original (un-rendered) data
        orig_confs = np.copy(confs)
        orig_depths = np.copy(depths)

        # transform all inputs 
        # [N, C, H, W]
        confs = torch.Tensor(confs).unsqueeze(1).to(torch.float32)
        depths = torch.Tensor(depths).unsqueeze(1).to(torch.float32)
        images = torch.movedim(torch.Tensor(images), (1,2,3), (2,3,1)).to(torch.float32)
        orig_confs = torch.Tensor(orig_confs).unsqueeze(1).to(torch.float32)
        orig_depths = torch.Tensor(orig_depths).unsqueeze(1).to(torch.float32)
        cams = torch.Tensor(cams).to(torch.float32) # [N, 2, 4, 4]
        if self.mode == "training":
            gt_depth = torch.Tensor(gt_depth).unsqueeze(0).to(torch.float32) # [C, H, W]

        depths, confs = warp(depths, confs, torch.clone(cams))
        depths = depths.unsqueeze(1)
        confs = confs.unsqueeze(1)

        # load data dict
        data = {}
        data["ref_index"] = ref_index
        data["scene"] = scene
        data["K"] = K
        data["depths"] = depths
        data["orig_depths"] = orig_depths
        data["images"] = images
        data["confs"] = confs
        data["orig_confs"] = orig_confs
        data["cams"] = cams
        data['num_frame'] = self.num_frame

        if self.mode == "training":
            data["gt_depth"] = gt_depth

        return data
