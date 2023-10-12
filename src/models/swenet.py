import sys
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.common import *
from src.models.layers import * 
from src.utils.preprocess import *
from src.models.confidence import confidence_estimation

######################################################################
# Search Window Estimation Network
######################################################################
class SWENet(nn.Module):
    def __init__(self, cfg):
        super(SWENet, self).__init__()
        self.cfg = cfg
        self.device = self.cfg["device"]

        in_channels = self.cfg["swenet"]["in_channels"]
        base_channels = self.cfg["swenet"]["base_channels"]
        out_channels = self.cfg["swenet"]["out_channels"]
        kernel_size = self.cfg["swenet"]["kernel_size"]

        self.max_radius_scale = nn.parameter.Parameter(torch.tensor(self.cfg["swenet"]["max_radius"]), requires_grad=False)
        self.min_radius_scale = nn.parameter.Parameter(torch.tensor(self.cfg["swenet"]["min_radius"]), requires_grad=False)

        # 2D ConvNet for binary intial depth estimation
        self.conv1 = conv2d_bn(in_channels, base_channels, kernel_size=kernel_size, stride=1)
        self.conv2 = conv2d_bn(base_channels, base_channels, kernel_size=kernel_size, stride=1)
        self.conv3 = conv2d_bn(base_channels, base_channels * 2, kernel_size=kernel_size, stride=2)
        self.conv4 = conv2d_bn(base_channels * 2, base_channels * 2, kernel_size=kernel_size, stride=1)
        self.conv5 = conv2d_bn(base_channels * 2, base_channels * 4, kernel_size=kernel_size, stride=2)
        self.conv6 = deconv2d_bn(base_channels * 4, base_channels * 2, kernel_size=kernel_size, stride=2)
        self.conv7 = conv2d_bn(base_channels * 2, base_channels * 2, kernel_size=kernel_size, stride=1)
        self.conv8 = deconv2d_bn(base_channels * 2, base_channels, kernel_size=kernel_size, stride=2)
        self.conv9 = conv2d_bn(base_channels, base_channels, kernel_size=kernel_size, stride=1)
        self.conv10 = conv2d(base_channels, out_channels, kernel_size=kernel_size, stride=1, nonlinearity="sigmoid")


        # 2D ConvNet for confidence-based window radius estimation
        self.conv11 = conv2d_bn(in_channels, base_channels, kernel_size=kernel_size, stride=1)
        self.conv12 = conv2d_bn(base_channels, base_channels * 2, kernel_size=kernel_size, stride=2)
        self.conv13 = conv2d_bn(base_channels * 2, base_channels * 4, kernel_size=kernel_size, stride=2)
        self.conv14 = deconv2d_bn(base_channels * 4, base_channels * 2, kernel_size=kernel_size, stride=2)
        self.conv15 = deconv2d_bn(base_channels * 2, base_channels, kernel_size=kernel_size, stride=2)
        self.conv16 = conv2d(base_channels, out_channels, kernel_size=kernel_size, stride=1, nonlinearity="sigmoid")

    def forward(self, data):
        batch_size, views, _, height, width = data["depths"].shape
        near_depth = data["cams"][:,0,1,3,0]
        far_depth = data["cams"][:,0,1,3,3]
        depth_range = far_depth - near_depth
        depth_mode = self.cfg["swenet"]["depth_mode"]

        depths = data["depths"].squeeze(2)
        confs = data["confs"].squeeze(2)

        if (depth_mode=="mean"):
            # use mean depth as initial center value
            initial_depth = torch.div(torch.sum(depths * confs, dim=1, keepdim=True), torch.sum(confs, dim=1, keepdim=True))
        elif(depth_mode=="conf"):
            # use most confident depth as initial center value
            max_conf_ind = torch.argmax(confs, dim=1, keepdim=True)
            initial_depth = torch.gather(depths, dim=1, index=max_conf_ind)

        # expand ref_depth
        depth_range = depth_range.reshape(batch_size,1,1,1).repeat(1,1,height,width)
        near_depth = near_depth.reshape(batch_size,1,1,1).repeat(1,1,height,width)

        # compute number of valid (non-hole) values per pixel, across views
        valid_values = torch.ne(depths, 0.0).to(torch.float32).to(self.device)
        valid_count = torch.sum(valid_values, dim=1, keepdim=True)

        # normalize depth values
        norm_depths = torch.clamp((depths-near_depth) / depth_range, min=0.0, max=1.0).to(torch.float32)
        norm_initial_depth = torch.clamp((initial_depth-near_depth) / depth_range, min=0.0, max=1.0).to(torch.float32)

        # compute mean and std for depths
        depth_mean = torch.div(torch.sum(norm_depths,dim=1, keepdim=True), valid_count)
        depth_std = non_zero_std(norm_depths, self.device, dim=1, keepdim=True)

        # normalize std
        dsm = torch.max(depth_std)
        depth_std = torch.div(depth_std, dsm + 1e-5)

        # compute mean and std for confs
        conf_mean = torch.div(torch.sum(confs,dim=1, keepdim=True), valid_count)
        conf_std = non_zero_std(confs, self.device, dim=1, keepdim=True)

        # normalize std
        csm = torch.max(conf_std)
        conf_std = torch.div(conf_std, csm + 1e-5)

        # concatenate ref conf with average target conf
        inpt = torch.cat((depth_mean, depth_std, conf_mean, conf_std), dim=1)    # 4D: [Batch_Size, 4, Height, Width]

        # clean up gpu mem
        del depth_mean
        del depth_std
        del conf_mean
        del conf_std
        del valid_values
        del valid_count
        del norm_depths
        torch.cuda.empty_cache()

        c_map = norm_initial_depth
        center_inpt = torch.cat((c_map, inpt), dim=1)

        # run convolution
        output11 = self.conv11(center_inpt)
        output12 = self.conv12(output11)
        output13 = self.conv13(output12)
        output14 = self.conv14(output13)
        output14 = torch.add(output12, output14)
        output15 = self.conv15(output14)
        output15 = torch.add(output11, output15)
        window_output = self.conv16(output15)

        # compute center depth value
        center_depth = torch.mul(c_map, depth_range)
        center_depth = torch.add(center_depth, near_depth)

        # compute dynamic step-size
        max_radius = torch.mul(self.max_radius_scale, depth_range)
        min_radius = torch.mul(self.min_radius_scale, depth_range)
        window_radius = torch.add(torch.mul(window_output, max_radius), min_radius)

        # Compute start and end depth values per pixel
        low_bound = torch.sub(center_depth, window_radius)
        high_bound = torch.add(center_depth, window_radius)

        # clean up gpu mem
        del inpt
        del window_output
        del max_radius
        del min_radius
        del near_depth
        del far_depth
        del center_inpt
        torch.cuda.empty_cache()

        return torch.cat((low_bound, high_bound), dim=1)
