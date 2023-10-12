import os
import time
import random
import sys
import cv2
import numpy as np
from random import seed
import torch.utils.data as data
import torch
import torchvision.transforms as T
from tqdm import tqdm

from src.utils.common import *
from src.utils.io import read_extrinsics_tum, read_stereo_intrinsics_yaml, read_pfm
from src.preprocess import *

dirname = os.path.dirname(__file__)
rel_path = os.path.join(dirname, './cpp_utils/render/build/')
sys.path.append(rel_path)
import render_tgt_volume as rtv

torch.manual_seed(5)
seed(5)
np.random.seed(5)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class DtuDataLoader(data.Dataset):
    def __init__(self, data_root, view_num, max_d, mode="training", transform=T.Compose([T.ToTensor]), device="cpu", scale=1, eval_scan=1, load_gt=False):
        self.root = data_root
        self.view_num = view_num
        self.max_d = max_d
        self.mode = mode
        self.transform = transform
        self.device = device
        self.scale = scale
        self.samples = []
        self.cluster_list = np.array(open(os.path.join(data_root,"Cameras/pair.txt"), 'r').read().split())
        self.training_set = np.array([2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128])
        self.validation_set = np.array([3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117])
        self.evaluation_set = np.array([eval_scan])
        self.load_gt = load_gt

        self.build_samples()

    def build_samples(self):
        """ generate data paths for dtu dataset """
        data_set = []
        if self.mode == "training":
            data_set = self.training_set
        elif self.mode == "validation":
            data_set = self.validation_set
        elif self.mode == "evaluation":
            data_set = self.evaluation_set
        else:
            print("ERROR: invalid dataset mode received: '{}'".format(self.mode))
            sys.exit()


        # for each dataset
        for i in data_set:
            cam_folder = os.path.join(self.root, "Cameras")
            conf_folder = os.path.join(self.root, ("Confs/scan{:03d}".format(i)))
            depth_folder = os.path.join(self.root, ("Depths/scan{:03d}".format(i)))
            if (self.mode != "evaluation" or self.load_gt):
                gt_depth_folder = os.path.join(self.root, ("GT_Depths/scan{:03d}".format(i)))
                #gt_visibility_folder = os.path.join(self.root, ("GT_Visibility/scan{:03d}".format(i)))

            # for each reference image
            # TODO: Reading of pair file needs updating (see TNT data loader below)...
            #       Updates were needed for TNT (especially advanced set) since
            #       the pair file entries sometimes did not contain 10 supporting views.
            for p in range(0, int(self.cluster_list[0])):
                # reference view
                ref_index = int(self.cluster_list[22 * p + 1])
                ref_conf_path = os.path.join(conf_folder, ("{:08d}_conf.pfm".format(ref_index)))
                ref_depth_path = os.path.join(depth_folder, ("{:08d}_depth.pfm".format(ref_index)))
                ref_cam_path = os.path.join(cam_folder, ("{:08d}_cam.txt".format(ref_index)))

                # ground truth depth path
                if (self.mode != "evaluation" or self.load_gt):
                    gt_depth_path = os.path.join(gt_depth_folder, ("{:08d}_depth.pfm".format(ref_index)))
                    #gt_visibility_path = os.path.join(gt_visibility_folder, ("{:08d}.pfm".format(ref_index)))
                    #sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "gt_depth": gt_depth_path, "gt_visibility": gt_visibility_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}
                    sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "gt_depth": gt_depth_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}
                else:
                    sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}

                # target views
                tgt_ind = []
                for view in range(10):
                    tgt_ind.append(int(self.cluster_list[22 * p + 2 * view + 3]))
                
                if(self.mode == "training"):
                    tgt_ind = np.random.choice(tgt_ind, size=(self.view_num-1), replace=False) # random-k view selection
                    #tgt_ind = tgt_ind[:self.view_num-1] # best-k view selection
                else:
                    tgt_ind = tgt_ind[:self.view_num-1] # best-k view selection

                for index in tgt_ind:
                    tgt_conf_path = os.path.join(conf_folder, ("{:08d}_conf.pfm".format(index)))
                    tgt_depth_path = os.path.join(depth_folder, ("{:08d}_depth.pfm".format(index)))
                    tgt_cam_path = os.path.join(cam_folder, ("{:08d}_cam.txt".format(index)))
                    sample["tgt_confs"].append(tgt_conf_path)
                    sample["tgt_depths"].append(tgt_depth_path)
                    sample["tgt_cams"].append(tgt_cam_path)

                # append sample to samples list
                self.samples.append(sample)

        self.samples = np.array(self.samples)

    def load_data(self,index):
        sample = self.samples[index]

        # read input data
        ref_view_num = int(sample["ref_depth"][-18:-10])

        ref_conf = load_pfm(open(sample["ref_conf"],'rb'))
        ref_depth = load_pfm(open(sample["ref_depth"],'rb'))
        ref_cam = np.asarray(load_cam(open(sample["ref_cam"]), self.max_d) )
        if (self.mode != "evaluation" or self.load_gt):
            gt_depth = load_pfm(open(sample["gt_depth"],'rb'))
            gt_depth = scale_gt(gt_depth, scale=self.scale)
            #gt_visibility = load_pfm(open(sample["gt_visibility"],'rb'))
            #gt_visibility = scale_gt(gt_visibility, scale=self.scale)

        tgt_confs = np.array([load_pfm(open(tgt_conf,'rb')) for tgt_conf in sample["tgt_confs"]])
        tgt_depths = np.array([load_pfm(open(tgt_depth,'rb')) for tgt_depth in sample["tgt_depths"]])
        tgt_cams = np.array([load_cam(open(tgt_cam), self.max_d) for tgt_cam in sample["tgt_cams"]])

        # collect all confs, depths, and cams
        confs = np.concatenate((np.asarray([ref_conf]), tgt_confs), axis=0)
        depths = np.concatenate((np.asarray([ref_depth]), tgt_depths), axis=0)
        cams = np.concatenate((np.asarray([ref_cam]), tgt_cams), axis=0)

        # process input data
        depths[np.isnan(depths)] = 0.0
        confs[np.isnan(confs)] = 0.0
        depths,confs,cams = scale_mvs_input(depths, confs, cams, scale=self.scale)
        confs = np.clip(confs, 0.0, 1.0)

        # store original (un-rendered) data
        orig_confs = np.copy(confs)
        orig_depths = np.copy(depths)

        # render target views into reference view
        depths, confs = render_into_ref(depths, confs, np.copy(cams))
        
        # set depth range to [425, 937] (DTU only)
        cams[0,1,3,0] = 425
        cams[0,1,3,3] = 937

        # fix depth range and adapt depth sample number
        cams[:,1,3,1] = (cams[:,1,3,3] - cams[:,1,3,0]) / 256 

        # mask out-of-range ground-truth depth pixels
        if (self.mode != "evaluation" or self.load_gt):
            depth_start = cams[0,1,3,0] + cams[0,1,3,1]
            depth_end = cams[0,1,3,3] - cams[0,1,3,1]
            gt_depth = mask_depth_image(gt_depth, depth_start, depth_end)

        # transform all inputs
        confs = torch.Tensor(confs)
        depths = torch.Tensor(depths)
        orig_confs = torch.Tensor(orig_confs)
        orig_depths = torch.Tensor(orig_depths)

        if (self.mode != "evaluation" or self.load_gt):
            gt_depth = self.transform(gt_depth)
            #gt_visibility = self.transform(gt_visibility)
            #data = (depths, confs, orig_depths, orig_confs, cams, gt_depth, gt_visibility, ref_view_num)
            data = (depths, confs, orig_depths, orig_confs, cams, gt_depth, ref_view_num)
        else:
            data = (depths, confs, orig_depths, orig_confs, cams, ref_view_num)

        return data

    def __getitem__(self,index):
        # get sample for current index
        sample = self.samples[index]

        # load all other data
        data = self.load_data(index)
        return data

    def __len__(self):
        return len(self.samples)


class TntDataLoader(data.Dataset):

    def __init__(self, data_root, view_num, max_d, mode="training", transform=T.Compose([T.ToTensor]), device="cpu", tnt_scene="Ignatius", load_gt=False, scale=1):
        self.root = data_root
        self.view_num = view_num
        self.max_d = max_d
        self.mode = mode
        self.transform = transform
        self.device = device
        self.samples = []
        self.training_set = np.array(["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"])
        self.intermediate_set = np.array(["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"])
        self.advanced_set = np.array(["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"])
        self.evaluation_set = np.array([tnt_scene])
        self.load_gt = load_gt
        self.scale=scale

        self.build_samples()

    def build_samples(self):
        """ generate data paths for tnt dataset """
        data_set = []
        if self.mode == "training":
            data_set = self.training_set
        elif self.mode == "intermediate":
            data_set = self.validation_set
        elif self.mode == "advanced":
            data_set = self.advanced_set
        elif self.mode == "evaluation":
            data_set = self.evaluation_set
        else:
            print("ERROR: invalid dataset mode received: '{}'".format(mode))
            sys.exit()

        # for each dataset
        for d in data_set:
            with open(os.path.join(self.root,"Cameras/{}/pair.txt".format(d)),'r') as cf:
                lines = cf.readlines()
            num_views = int(lines[0])
            cluster_list=np.zeros((num_views,22)) # 22 here is for the...  ref number, number of sup views, max 10 view numbers, max 10 view scores
            i = 1
            j = 0
            while i < len(lines)-1:
                cluster_list[j,0] = np.asarray(lines[i].strip().split())
                lst = np.asarray(lines[i+1].strip().split())
                cluster_list[j,1:len(lst)+1] = lst
                i+=2
                j+=1

            cam_folder = os.path.join(self.root, "Cameras/{}".format(d))
            conf_folder = os.path.join(self.root, ("Confs/{}".format(d)))
            depth_folder = os.path.join(self.root, ("Depths/{}".format(d)))
            if (self.mode != "evaluation" or self.load_gt):
                gt_depth_folder = os.path.join(self.root, ("GT_Depths/{}".format(d)))

            # for each reference image
            num_views,_ = cluster_list.shape
            for r in range(num_views):
                # reference view
                ref_index = int(cluster_list[r,0])
                max_src_views = int(cluster_list[r,1])
                ref_conf_path = os.path.join(conf_folder, ("{:08d}_conf.pfm".format(ref_index)))
                ref_depth_path = os.path.join(depth_folder, ("{:08d}_depth.pfm".format(ref_index)))
                ref_cam_path = os.path.join(cam_folder, ("{:08d}_cam.txt".format(ref_index)))

                # ground truth depth path
                if (self.mode != "evaluation" or self.load_gt):
                    gt_depth_path = os.path.join(gt_depth_folder, ("{:08d}_depth.pfm".format(ref_index)))
                    sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "gt_depth": gt_depth_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}
                else:
                    sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}

                # target views
                num_src_views = min(self.view_num, max_src_views)
                if(num_src_views < 1):
                    continue

                for view in range(num_src_views):
                    tgt_index = int(cluster_list[r, ((2*view) + 2)])
                    tgt_conf_path = os.path.join(conf_folder, ("{:08d}_conf.pfm".format(tgt_index)))
                    tgt_depth_path = os.path.join(depth_folder, ("{:08d}_depth.pfm".format(tgt_index)))
                    tgt_cam_path = os.path.join(cam_folder, ("{:08d}_cam.txt".format(tgt_index)))
                    sample["tgt_confs"].append(tgt_conf_path)
                    sample["tgt_depths"].append(tgt_depth_path)
                    sample["tgt_cams"].append(tgt_cam_path)

                # append sample to samples list
                self.samples.append(sample)

        self.samples = np.array(self.samples)

    def load_data(self,index):
        sample = self.samples[index]

        ###### read input data ######
        ref_view_num = int(sample["ref_depth"][-18:-10])

        ref_conf = load_pfm(open(sample["ref_conf"],'rb'))
        ref_depth = load_pfm(open(sample["ref_depth"],'rb'))
        ref_cam = load_cam(open(sample["ref_cam"]), self.max_d)
        confs = np.expand_dims(ref_conf,0)
        depths = np.expand_dims(ref_depth,0)
        cams = np.expand_dims(ref_cam,0)

        if (self.mode != "evaluation" or self.load_gt):
            gt_depth = load_pfm(open(sample["gt_depth"],'rb'))
            gt_depth = scale_gt(gt_depth, scale=self.scale)

        if (len(sample['tgt_confs']) > 0):
            tgt_confs = np.array([load_pfm(open(tgt_conf,'rb')) for tgt_conf in sample["tgt_confs"]])
            tgt_depths = np.array([load_pfm(open(tgt_depth,'rb')) for tgt_depth in sample["tgt_depths"]])
            tgt_cams = np.array([load_cam(open(tgt_cam), self.max_d) for tgt_cam in sample["tgt_cams"]])
            confs = np.concatenate((confs, tgt_confs), axis=0)
            depths = np.concatenate((depths, tgt_depths), axis=0)
            cams = np.concatenate((cams, tgt_cams), axis=0)

        # process input data
        depths[np.isnan(depths)] = 0.0
        confs[np.isnan(confs)] = 0.0
        depths,confs,cams = scale_mvs_input(depths, confs, cams, scale=self.scale)
        confs = np.clip(confs, 0.0, 1.0)

        # store original (un-rendered) data
        orig_confs = np.copy(confs)
        orig_depths = np.copy(depths)

        # render target views into reference view
        depths, confs = render_into_ref(depths, confs, np.copy(cams))

        # fix depth range and adapt depth sample number
        cams[:,1,3,3] = (cams[:,1,3,1]*cams[:,1,3,2]) + cams[:,1,3,0]

        # mask out-of-range depth pixels (in a relaxed range)
        if (self.mode != "evaluation" or self.load_gt):
            depth_start = cams[0,1,3,0] + cams[0,1,3,1]
            depth_end = cams[0,1,3,3] - cams[0,1,3,1]
            gt_depth = mask_depth_image(gt_depth, depth_start, depth_end)

        # transform all inputs
        confs = torch.Tensor(confs)
        depths = torch.Tensor(depths)
        orig_confs = torch.Tensor(orig_confs)
        orig_depths = torch.Tensor(orig_depths)

        if (self.mode != "evaluation" or self.load_gt):
            gt_depth = self.transform(gt_depth)
            data = (depths, confs, orig_depths, orig_confs, cams, gt_depth, ref_view_num)
        else:
            data = (depths, confs, orig_depths, orig_confs, cams, ref_view_num)

        return data

    def __getitem__(self,index):
        # get sample for current index
        sample = self.samples[index]

        # load all other data
        data = self.load_data(index)
        return data

    def __len__(self):
        return len(self.samples)


class BlendedDataLoader(data.Dataset):

    def __init__(self, data_root, view_num, max_d, mode="training", transform=T.Compose([T.ToTensor]), device="cpu", scale=1, blended_scene=106, load_gt=False):
        self.root = data_root
        self.view_num = view_num
        self.max_d = max_d
        self.mode = mode
        self.transform = transform
        self.device = device
        self.scale = scale
        self.samples = []
        self.training_set = np.asarray(range(0,96))
        self.validation_set = np.asarray(range(96,106))
        self.evaluation_set = np.asarray([blended_scene])
        self.load_gt = load_gt

        self.build_samples()

    def build_samples(self):
        """ generate data paths for dtu dataset """
        data_set = []
        if self.mode == "training":
            data_set = self.training_set
        elif self.mode == "validation":
            data_set = self.validation_set
        elif self.mode == "evaluation":
            data_set = self.evaluation_set
        else:
            print("ERROR: invalid dataset mode received: '{}'".format(self.mode))
            sys.exit()


        # for each dataset
        for d in data_set:
            with open(os.path.join(self.root,"Cameras/scene{:03d}/pair.txt".format(d)),'r') as cf:
                lines = cf.readlines()
            num_views = int(lines[0])
            cluster_list=np.zeros((num_views,22)) # 22 here is for the...  ref number, number of sup views, max 10 view numbers, max 10 view scores
            i = 1
            j = 0
            while i < len(lines)-1:
                cluster_list[j,0] = np.asarray(lines[i].strip().split())
                lst = np.asarray(lines[i+1].strip().split())
                cluster_list[j,1:len(lst)+1] = lst
                i+=2
                j+=1

            cam_folder = os.path.join(self.root, "Cameras/scene{:03d}".format(d))
            conf_folder = os.path.join(self.root, ("Confs/scene{:03d}".format(d)))
            depth_folder = os.path.join(self.root, ("Depths/scene{:03d}".format(d)))
            if (self.mode != "evaluation" or self.load_gt):
                gt_depth_folder = os.path.join(self.root, ("GT_Depths/scene{:03d}".format(d)))

            # for each reference image
            num_views,_ = cluster_list.shape
            for r in range(num_views):
                # reference view
                ref_index = int(cluster_list[r,0])
                max_src_views = int(cluster_list[r,1])

                # skip any views that do not have enough 
                if (max_src_views <= self.view_num):
                    continue

                ref_conf_path = os.path.join(conf_folder, ("{:08d}_conf.pfm".format(ref_index)))
                ref_depth_path = os.path.join(depth_folder, ("{:08d}_depth.pfm".format(ref_index)))
                ref_cam_path = os.path.join(cam_folder, ("{:08d}_cam.txt".format(ref_index)))

                # ground truth depth path
                if (self.mode != "evaluation" or self.load_gt):
                    gt_depth_path = os.path.join(gt_depth_folder, ("{:08d}.pfm".format(ref_index)))
                    sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "gt_depth": gt_depth_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}
                else:
                    sample = {"ref_depth": ref_depth_path, "ref_cam": ref_cam_path, "ref_conf": ref_conf_path, "tgt_depths": [], "tgt_cams": [], "tgt_confs": []}

                # target views
                num_src_views = min(self.view_num, max_src_views)
                if(num_src_views < 1):
                    continue

                for view in range(num_src_views):
                    tgt_index = int(cluster_list[r, ((2*view) + 2)])
                    tgt_conf_path = os.path.join(conf_folder, ("{:08d}_conf.pfm".format(tgt_index)))
                    tgt_depth_path = os.path.join(depth_folder, ("{:08d}_depth.pfm".format(tgt_index)))
                    tgt_cam_path = os.path.join(cam_folder, ("{:08d}_cam.txt".format(tgt_index)))
                    sample["tgt_confs"].append(tgt_conf_path)
                    sample["tgt_depths"].append(tgt_depth_path)
                    sample["tgt_cams"].append(tgt_cam_path)

                # append sample to samples list
                self.samples.append(sample)

        self.samples = np.array(self.samples)

    def load_data(self,index):
        sample = self.samples[index]

        ###### read input data ######
        ref_view_num = int(sample["ref_depth"][-18:-10])

        ref_conf = load_pfm(open(sample["ref_conf"],'rb'))
        ref_depth = load_pfm(open(sample["ref_depth"],'rb'))
        ref_cam = load_cam(open(sample["ref_cam"]), self.max_d)
        confs = np.expand_dims(ref_conf,0)
        depths = np.expand_dims(ref_depth,0)
        cams = np.expand_dims(ref_cam,0)

        if (self.mode != "evaluation" or self.load_gt):
            gt_depth = load_pfm(open(sample["gt_depth"],'rb'))
            gt_depth = scale_gt(gt_depth, scale=self.scale)

        if (len(sample['tgt_confs']) > 0):
            tgt_confs = np.array([load_pfm(open(tgt_conf,'rb')) for tgt_conf in sample["tgt_confs"]])
            tgt_depths = np.array([load_pfm(open(tgt_depth,'rb')) for tgt_depth in sample["tgt_depths"]])
            tgt_cams = np.array([load_cam(open(tgt_cam), self.max_d) for tgt_cam in sample["tgt_cams"]])
            confs = np.concatenate((confs, tgt_confs), axis=0)
            depths = np.concatenate((depths, tgt_depths), axis=0)
            cams = np.concatenate((cams, tgt_cams), axis=0)

        # process input data
        depths[np.isnan(depths)] = 0.0
        confs[np.isnan(confs)] = 0.0
        depths,confs,cams = scale_mvs_input(depths, confs, cams, scale=self.scale)
        confs = np.clip(confs, 0.0, 1.0)

        # store original (un-rendered) data
        orig_confs = np.copy(confs)
        orig_depths = np.copy(depths)

        # render target views into reference view
        depths, confs = render_into_ref(depths, confs, np.copy(cams))

        # fix depth range and adapt depth sample number
        cams[:,1,3,3] = (cams[:,1,3,1]*cams[:,1,3,2]) + cams[:,1,3,0]

        # mask out-of-range depth pixels (in a relaxed range)
        if (self.mode != "evaluation" or self.load_gt):
            depth_start = cams[0,1,3,0] + cams[0,1,3,1]
            depth_end = cams[0,1,3,3] - cams[0,1,3,1]
            gt_depth = mask_depth_image(gt_depth, depth_start, depth_end)

        # transform all inputs
        confs = torch.Tensor(confs)
        depths = torch.Tensor(depths)
        orig_confs = torch.Tensor(orig_confs)
        orig_depths = torch.Tensor(orig_depths)

        if (self.mode != "evaluation" or self.load_gt):
            gt_depth = self.transform(gt_depth)
            data = (depths, confs, orig_depths, orig_confs, cams, gt_depth, ref_view_num)
        else:
            data = (depths, confs, orig_depths, orig_confs, cams, ref_view_num)

        return data

    def __getitem__(self,index):
        # get sample for current index
        sample = self.samples[index]

        # load all other data
        data = self.load_data(index)
        return data

    def __len__(self):
        return len(self.samples)


class UnderwaterDataLoader(data.Dataset):

    def __init__(self, data_root, view_num, max_d, mode="testing", transform=T.Compose([T.ToTensor]), device="cpu", scale=1, eval_scan="Florida", load_gt=False):
        self.root = data_root
        self.view_num = view_num
        self.max_d = max_d
        self.mode = mode
        self.transform = transform
        self.device = device
        self.scale = scale
        self.samples = []
        self.evaluation_set = np.array([eval_scan])

        self.build_samples()

    def build_samples(self):
        """ generate data paths for dtu dataset """
        data_set = self.evaluation_set
        frame_offset = ((self.view_num-1)//2)
        frame_spacing = 1
        radius = frame_offset*frame_spacing

        samples = []
        for scene in data_set:
            pose_file = os.path.join(self.root, scene, "poses.txt")
            intrinsics_file = os.path.join(self.root, scene, "intrinsics.yaml")

            with open(pose_file, 'r') as pf:
                lines = pf.readlines()
                filenames = []
                for line in lines:
                    fn = line.strip().split()[0]
                    fn = fn.split('.')
                    filenames.append(fn[0] + fn[1])
                    

            depth_folder = os.path.join(self.root, scene, "depth_maps")
            #depth_folder = os.path.join(self.root, scene, "fused_depths")
            depth_files = os.listdir(depth_folder)
            depth_files.sort()

            conf_folder = os.path.join(self.root, scene, "conf_maps")
            conf_files = os.listdir(conf_folder)
            conf_files.sort()

            conf_files = [cf for cf in conf_files if (cf in depth_files) ]

            poses = read_extrinsics_tum(pose_file)
            K_33, _,_,_,_,_ = read_stereo_intrinsics_yaml(intrinsics_file)
            K = np.zeros((4,4))
            K[:3,:3] = K_33
            K[3,3] = 1
            self.cams = np.asarray([np.asarray([P,K]) for P in poses])

            ind = 0
            for fn in filenames:
                if ind >= len(depth_files):
                    self.cams = np.delete(self.cams, ind, axis=0)
                elif depth_files[ind][:-4] != fn:
                    self.cams = np.delete(self.cams, ind, axis=0)
                else:
                    ind+=1

            #   depth_files = depth_files[-80:]
            #   conf_files = conf_files[-80:]
            #   self.cams = self.cams[-80:]

            #   depth_files = depth_files[750:800]
            #   conf_files = conf_files[750:800]
            #   self.cams = self.cams[750:800]

            count = len(self.cams)
            for ref_view in range(0,count):
                start = ref_view - radius
                end = ref_view + radius

                while(start < 0):
                    start += frame_spacing
                    end += frame_spacing
                while(end >= count):
                    start -= frame_spacing
                    end -= frame_spacing

                src_views = [i for i in range(start, end+1, frame_spacing) if i != ref_view]

                # get data paths
                ref_conf_path = os.path.join(conf_folder, conf_files[ref_view])
                ref_depth_path = os.path.join(depth_folder, depth_files[ref_view])
                sample = {"ref_view": ref_view, "ref_depth": ref_depth_path, "ref_conf": ref_conf_path, "tgt_depths": [], "tgt_inds": [], "tgt_confs": []}

                for index in src_views:
                    tgt_conf_path = os.path.join(conf_folder, conf_files[index])
                    tgt_depth_path = os.path.join(depth_folder, depth_files[index])
                    sample["tgt_confs"].append(tgt_conf_path)
                    sample["tgt_depths"].append(tgt_depth_path)
                    sample["tgt_inds"].append(index)

                #if ref_view > 1020 and ref_view < 1061:
                # append sample to samples list
                self.samples.append(sample)

        self.samples = np.array(self.samples)

    def load_data(self,index):
        sample = self.samples[index]

        # read input data
        ref_view_num = int(sample["ref_view"])

        ref_conf = read_pfm(sample["ref_conf"])[:,:800]
        ref_depth = read_pfm(sample["ref_depth"])
        ref_cam = self.cams[ref_view_num]

        tgt_confs = np.array([read_pfm(tgt_conf)[:,:800] for tgt_conf in sample["tgt_confs"]])
        tgt_depths = np.array([read_pfm(tgt_depth) for tgt_depth in sample["tgt_depths"]])
        tgt_cams = np.array([self.cams[src_view] for src_view in sample["tgt_inds"]])

        # collect all confs, depths, and cams
        confs = np.concatenate((np.asarray([ref_conf]), tgt_confs), axis=0)
        depths = np.concatenate((np.asarray([ref_depth]), tgt_depths), axis=0)
        cams = np.concatenate((np.asarray([ref_cam]), tgt_cams), axis=0)

        # process input data
        depths[np.isnan(depths)] = 0.0
        confs[np.isnan(confs)] = 0.0
        depths,confs,cams = scale_mvs_input(depths, confs, cams, scale=self.scale)
        depths = np.where(depths > 4.0, 0.0, depths)
        confs = np.clip(confs, 0.0, 1.0)

        # store original (un-rendered) data
        orig_confs = np.copy(confs)
        orig_depths = np.copy(depths)

        # render target views into reference view
        depths, confs = render_into_ref(depths, confs, np.copy(cams))
        
        cams[0,1,3,0] = 1.0
        cams[0,1,3,3] = 4.0

        # fix depth range and adapt depth sample number
        cams[:,1,3,1] = (cams[:,1,3,3] - cams[:,1,3,0]) / 256 

        # transform all inputs
        confs = torch.Tensor(confs)
        depths = torch.Tensor(depths)
        orig_confs = torch.Tensor(orig_confs)
        orig_depths = torch.Tensor(orig_depths)

        data = (depths, confs, orig_depths, orig_confs, cams, ref_view_num)
        return data

    def __getitem__(self,index):
        # get sample for current index
        sample = self.samples[index]

        # load all other data
        data = self.load_data(index)
        return data

    def __len__(self):
        return len(self.samples)
