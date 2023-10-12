import os
import random
import numpy as np
import torch
import cv2
import sys
import json

from src.utils.io import read_single_cam_sfm, read_pfm
from src.datasets.BaseDataset import BaseDataset

class ScanNet(BaseDataset):
    def __init__(self, cfg, mode, scenes):
        super(ScanNet, self).__init__(cfg, mode, scenes)

    def get_frame_count(self, scene):
        image_files = os.listdir(os.path.join(self.data_path, scene, "color"))
        image_files = [img for img in image_files if img[-4:]==".png"]
        image_files.sort()
        return len(image_files)

    def build_samples(self):
        self.samples = []
        self.frame_count = 0
        for scene in self.scenes:
            # build samples dict for other data
            curr_frame_count = self.get_frame_count(scene)
            self.samples.extend(self.build_samples_helper(scene, curr_frame_count))
            self.frame_count += curr_frame_count

            # load pose and intrinsics for current scene
            self.load_all_poses(scene, curr_frame_count)
            self.load_intrinsics(scene)

    def build_samples_helper(self, scene, frame_count):
        samples = []

        frame_offset = ((self.num_frame-1)//2)
        radius = frame_offset*self.frame_spacing
        for ref_frame in range(0, frame_count):
            start = ref_frame - radius
            end = ref_frame + radius

            while(start < 0):
                start += self.frame_spacing
                end += self.frame_spacing
            while(end >= frame_count):
                start -= self.frame_spacing
                end -= self.frame_spacing

            frame_inds = [i for i in range(start, end+1, self.frame_spacing) if i != ref_frame]

            frame_inds.insert(0, ref_frame)
            image_files = [ os.path.join(self.data_path, scene, 'color', f"{ind:06d}.png") for ind in frame_inds ]
            depth_files = [ os.path.join(self.data_path, scene, 'depth', f"{ind:06d}.png") for ind in frame_inds ]

            samples.append({"scene": scene,
                            "frame_inds": frame_inds,
                            "image_files": image_files,
                            "depth_files": depth_files,
                            })
        return samples

    def load_all_poses(self, scene, frame_count):
        self.poses[scene] = []
        for ind in range(frame_count):
            path = os.path.join(self.data_path, scene, 'pose', f"{ind:06d}.txt")
            pose = np.loadtxt(path).astype('float32')
            pose = np.linalg.inv(pose)
            self.poses[scene].append(pose)

    def load_intrinsics(self, scene):
        intrinsics_file = os.path.join(self.data_path, scene, "intrinsics.txt")
        K = np.loadtxt(intrinsics_file).astype('float32')
        self.K[scene] = K[:3,:3]
        self.H = self.cfg["camera"]["height"]
        self.W = self.cfg["camera"]["width"]

    def get_pose(self, scene, frame_id):
        return self.poses[scene][frame_id]
        
    def get_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.
        do_color_aug = False
        return image

    def get_depth(self, depth_file):
        gt_depth = cv2.imread(depth_file, 2) / 1000.0
        return gt_depth.astype(np.float32)
