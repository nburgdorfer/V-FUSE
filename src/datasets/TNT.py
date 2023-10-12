import os
import random
import numpy as np
import torch
import cv2
import sys
import json

from src.utils.io import read_single_cam_sfm, read_pfm, load_cluster_list
from src.datasets.BaseDataset import BaseDataset

class TNT(BaseDataset):
    def __init__(self, cfg, mode, scenes):
        super(TNT, self).__init__(cfg, mode, scenes)

    def get_frame_count(self, scene):
        camera_files = os.listdir(os.path.join(self.data_path, scene, "cams"))
        camera_files = [cam for cam in camera_files if cam[-8:]=="_cam.txt"]
        return len(camera_files)
    
    def get_cluster_list_file(self, scene):
        return os.path.join(self.cfg["data_path"], f"Cameras/{scene}/pair.txt")        

    def build_samples(self):
        self.samples_from_cluster_list()

    def samples_from_cluster_list(self):
        self.samples = []
        self.frame_count = 0

        for scene in self.scenes:
            self.load_intrinsics(scene)
            cluster_list = load_cluster_list(os.path.join(self.data_path, f"Cameras/{scene}/pair.txt"))

            cam_folder = os.path.join(self.data_path, f"Cameras/{scene}")
            conf_folder = os.path.join(self.data_path, f"Confs/{scene}")
            depth_folder = os.path.join(self.data_path, f"Depths/{scene}")
            image_folder = os.path.join(self.data_path, f"Images/{scene}")
            if (self.mode != "evaluation" or self.load_gt):
                gt_depth_folder = os.path.join(self.data_path, f"GT_Depths/{scene}")

            # for each reference image
            num_views,_ = cluster_list.shape
            for n in range(num_views):
                ref_index = int(cluster_list[n,0])
                max_src_views = int(cluster_list[n,1])
                ref_conf_path = os.path.join(conf_folder, f"{ref_index:08d}_conf.pfm")
                ref_depth_path = os.path.join(depth_folder, f"{ref_index:08d}_depth.pfm")
                ref_cam_path = os.path.join(cam_folder, f"{ref_index:08d}_cam.txt")
                ref_image_path = os.path.join(image_folder, f"{ref_index:08d}.png")

                # ground truth depth path
                sample = {
                        "ref_index": ref_index,
                        "scene": scene,
                        "depths": [ref_depth_path],
                        "images": [ref_image_path],
                        "cams": [ref_cam_path],
                        "confs": [ref_conf_path],
                        }
                if (self.mode != "evaluation" or self.load_gt):
                    gt_depth_path = os.path.join(gt_depth_folder, f"{ref_index:08d}_depth.pfm")
                    sample["gt_depth"] = gt_depth_path

                # target views
                num_src_views = min(self.num_frame, max_src_views)
                if(num_src_views < 1):
                    continue

                for view in range(1,num_src_views):
                    tgt_index = int(cluster_list[n, ((2*view) + 2)])
                    tgt_conf_path = os.path.join(conf_folder, f"{tgt_index:08d}_conf.pfm")
                    tgt_depth_path = os.path.join(depth_folder, f"{tgt_index:08d}_depth.pfm")
                    tgt_image_path = os.path.join(image_folder, f"{tgt_index:08d}.png")
                    tgt_cam_path = os.path.join(cam_folder, f"{tgt_index:08d}_cam.txt")

                    sample["depths"].append(tgt_depth_path)
                    sample["images"].append(tgt_image_path)
                    sample["cams"].append(tgt_cam_path)
                    sample["confs"].append(tgt_conf_path)

                # append sample to samples list
                self.samples.append(sample)

    def samples_from_stream(self):
        self.samples = []
        self.frame_count = 0
        for scene in self.scenes:
            # build samples dict for other data
            curr_frame_count = self.get_frame_count(scene)

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

            self.samples.extend(samples)
            self.frame_count += curr_frame_count

    def load_intrinsics(self, scene):
        cam_file = os.path.join(self.data_path, f"Cameras/{scene}/00000000_cam.txt")
        cam = read_single_cam_sfm(cam_file)
        self.K[scene] = cam[1,:3,:3]

    def get_pose(self, pose_file):
        cam_file = os.path.join(pose_file)
        cam = read_single_cam_sfm(cam_file)
        return cam[0]

    def get_metadata(self, pose_file):
        cam_file = os.path.join(pose_file)
        cam = read_single_cam_sfm(cam_file)
        return cam[1,3,:]
        
    def get_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_depth(self, depth_file):
        depth = read_pfm(depth_file)
        return depth.astype(np.float32)

    def get_conf(self, conf_file):
        conf = read_pfm(conf_file)
        return conf.astype(np.float32)
