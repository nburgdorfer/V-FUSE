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
from src.models.swenet import SWENet
from src.models.confidence import confidence_estimation

#from src.cpp_utils.render.build import render_tgt_volume as rtv

######################################################################
# Fusion Network
######################################################################
class Fusion(nn.Module):
    def __init__(self, cfg, depth_planes):
        super(Fusion, self).__init__()
        # parameter setup
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.depth_planes = depth_planes

        in_channels = self.cfg["fusion"]["in_channels"]
        base_channels = self.cfg["fusion"]["base_channels"]
        out_channels = self.cfg["fusion"]["out_channels"]
        kernel_size = self.cfg["fusion"]["kernel_size"]

        self.sup_planes = nn.parameter.Parameter(torch.tensor(self.cfg["fusion"]["sup_planes"]), requires_grad=True)
        self.scale = nn.parameter.Parameter(torch.tensor(self.cfg["fusion"]["sigmoid_scale"]), requires_grad=True)

        # Search Window Estimation Network
        self.swo_net = SWENet(self.cfg)

        # 3D UNet Cost Regularization
        self.conv1_0 = conv3d_bn(in_channels, base_channels * 2, kernel_size=kernel_size, stride=2)
        self.conv2_0 = conv3d_bn(base_channels*2, base_channels*4, kernel_size=kernel_size, stride=2)
        self.conv3_0 = conv3d_bn(base_channels*4, base_channels*8, kernel_size=kernel_size, stride=2)
        self.conv0_1 = conv3d_bn(in_channels, base_channels, kernel_size=kernel_size, stride=1)
        self.conv1_1 = conv3d_bn(base_channels*2, base_channels*2, kernel_size=kernel_size, stride=1)
        self.conv2_1 = conv3d_bn(base_channels*4, base_channels*4, kernel_size=kernel_size, stride=1)
        self.conv3_1 = conv3d_bn(base_channels*8, base_channels*8, kernel_size=kernel_size, stride=1)
        self.conv4_0 = deconv3d_bn(base_channels*8, base_channels*4, kernel_size=kernel_size, stride=2)
        self.conv5_0 = deconv3d_bn(base_channels*4, base_channels*2, kernel_size=kernel_size, stride=2)
        self.conv6_0 = deconv3d_bn(base_channels*2, base_channels, kernel_size=kernel_size, stride=2)
        self.conv6_2 = conv3d(base_channels, out_channels, kernel_size=kernel_size, stride=1)

    def build_cost_volume(self, data, depth_bounds):
        depth_intervals = torch.div(torch.sub(depth_bounds[:,1], depth_bounds[:,0]), self.depth_planes-1)

        # read in depth information
        batch_size, views, _, height, width = data["depths"].shape
        full_interval = (data["cams"][0,0,1,3,3] - data["cams"][0,0,1,3,0]).item()

        depths = data["depths"].squeeze(2)
        odepths = data["orig_depths"].squeeze(2)
        confs = data["confs"].squeeze(2)
        oconfs = data["orig_confs"].squeeze(2)
        cams = data["cams"].clone()

        # initialize cost volume: shape is [ B, 3, M, H, W ] (batch size, channels, depth planes, height, width)
        cost_volume = torch.zeros(batch_size, 3, self.depth_planes, height, width, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            depth_vol = torch.zeros((batch_size, self.depth_planes, height, width), device=self.device)
            depth_vol[:,0] = depth_bounds[:,0]

            ## Compute depth values vector
            for p in range(self.depth_planes-1):
                depth_vol[:,p+1] = torch.add(depth_vol[:,p], depth_intervals)

        # compute sigmoid multiplier based on search window diameter
        interval_ratio = torch.div(full_interval, depth_intervals)
        sig_scale = self.scale * interval_ratio * self.depth_planes
        sig_scale = sig_scale.reshape(batch_size, 1, height, width)
        sig_scale = sig_scale.repeat(1,self.depth_planes,1,1)

        ## Free Space Violations (need to compute fsv first before data re-shaping)
        fsv_cost = torch.zeros((batch_size, views, self.depth_planes, height, width))
        for v in range(views):
            if(v==0):
                cost = torch.sub(depth_vol, odepths[:,0].unsqueeze(1))
                cost = torch.div(cost, full_interval)
                cost = torch.mul(cost, sig_scale)
                cost = torch.sigmoid(cost)
                fsv_cost[:,0] = torch.mul(oconfs[:,0].unsqueeze(1), cost)

                # clean up gpu mem
                del cost
                torch.cuda.empty_cache()
            else:
                cost, warped_conf = warp_to_tgt(odepths[:,v].unsqueeze(1), oconfs[:,v].unsqueeze(1), cams[:,0], cams[:,v], self.depth_planes, depth_vol)
                cost = torch.div(cost, full_interval)
                cost = torch.mul(cost, sig_scale)
                cost = torch.sigmoid(cost)
                fsv_cost[:,v] = torch.mul(warped_conf, cost)

                # clean up gpu mem
                del cost
                del warped_conf
                torch.cuda.empty_cache()

        valid_values = torch.ne(fsv_cost, 0.0).to(torch.float32)
        valid_count = torch.sum(valid_values, dim=1, keepdim=True) + 1e-5
        cost_volume[:,2,:] = torch.div(torch.sum(fsv_cost,dim=1, keepdim=True), valid_count).squeeze(1)

        # compute support sigma based on search window diameter
        # support is ~70% in +- sigma from gt depth
        # so ~70% of the support response is within +- n depth-planes (n==self.sup_planes)
        sup_perc = self.sup_planes * (1/(self.depth_planes-1))
        sigma = torch.mul(sup_perc, depth_intervals)
        sigma = torch.div(sigma, full_interval)

        # clean up gpu mem
        del fsv_cost
        del depth_intervals
        torch.cuda.empty_cache()

        ## Reshape all data to perform high dimensional tensor computations
        sigma = sigma.reshape(batch_size, 1, 1, height, width)
        sigma = sigma.repeat(1,views,self.depth_planes,1,1)
        sig_scale = sig_scale.reshape(batch_size, 1, self.depth_planes, height, width)
        sig_scale = sig_scale.repeat(1,views,1,1,1)
        depths = depths.unsqueeze(2)
        depths = depths.repeat(1,1,self.depth_planes,1,1)
        confs = confs.unsqueeze(2)
        confs = confs.repeat(1,1,self.depth_planes,1,1)
        depth_vol = depth_vol.unsqueeze(1)
        depth_vol = depth_vol.repeat(1,views,1,1,1)

        ## Support - compute gaussian -> exp( -(d-d_v)^2 / (2*sigma^2) )
        sup_cost = torch.sub(depths, depth_vol)
        sup_cost = torch.div(sup_cost, full_interval)
        sup_cost = torch.square(sup_cost)
        sup_cost = torch.div(sup_cost, (2*torch.square(sigma))+1e-7)
        sup_cost = torch.mul(sup_cost, -1)
        sup_cost = torch.exp(sup_cost)
        sup_cost = torch.mul(sup_cost,confs)
        valid_values = torch.ne(sup_cost, 0.0).to(torch.float32).to(self.device)
        valid_count = torch.sum(valid_values, dim=1, keepdim=True) + 1e-5
        cost_volume[:,0,:] = torch.div(torch.sum(sup_cost,dim=1, keepdim=True), valid_count).squeeze(1)

        # clean up gpu mem
        del sup_cost
        torch.cuda.empty_cache()
        
        ## Occlusions
        occ_cost = torch.sub(depths, depth_vol)
        occ_cost = torch.div(occ_cost, full_interval)
        occ_cost = torch.mul(occ_cost, sig_scale)
        occ_cost = torch.sigmoid(occ_cost)
        occ_cost = torch.mul(occ_cost, confs)
        valid_values = torch.ne(occ_cost, 0.0).to(torch.float32).to(self.device)
        valid_count = torch.sum(valid_values, dim=1, keepdim=True) + 1e-5
        cost_volume[:,1,:] = torch.div(torch.sum(occ_cost,dim=1, keepdim=True), valid_count).squeeze(1)

        # clean up gpu mem
        del occ_cost
        del depth_vol
        del depths
        del confs
        torch.cuda.empty_cache()

        return cost_volume

    def forward(self, data):
        batch_size, views, _, height, width = data["depths"].shape

        # estimate search window bounds from confidence values
        depth_bounds = self.swo_net(data)

        # compute cost volume
        cost_volume = self.build_cost_volume(data, depth_bounds)

        # run cost regularization
        output1_0 = self.conv1_0(cost_volume)
        output2_0 = self.conv2_0(output1_0)
        output3_0 = self.conv3_0(output2_0)
        output0_1 = self.conv0_1(cost_volume)
        output1_1 = self.conv1_1(output1_0)
        output2_1 = self.conv2_1(output2_0)
        output3_1 = self.conv3_1(output3_0)
        output4_0 = self.conv4_0(output3_1)
        output4_1 = add(output4_0, output2_1)
        output5_0 = self.conv5_0(output4_1)
        output5_1 = add(output5_0, output1_1)
        output6_0 = self.conv6_0(output5_1)
        output6_1 = add(output6_0, output0_1)
        final_output = self.conv6_2(output6_1)

        # apply negative softmax (i.e. softmin) on depth planes dimension of cost volume
        prob_volume = F.softmax(final_output, dim=2)

        # clean up gpu mem
        del cost_volume
        del final_output
        torch.cuda.empty_cache()

        # regress depth map from prob volume
        depth_intervals = torch.div(torch.sub(depth_bounds[:,1], depth_bounds[:,0]), self.depth_planes-1)
        depth_vol = torch.zeros((batch_size, self.depth_planes, height, width), device=self.device)
        depth_vol[:,0] = depth_bounds[:,0]
        for p in range(self.depth_planes-1):
            depth_vol[:,p+1] = torch.add(depth_vol[:,p], depth_intervals)
        fused_depth = torch.sum(depth_vol.unsqueeze(1) * prob_volume, dim=2)

        ## get conf map from prob volume
        with torch.no_grad():
            fused_conf = confidence_estimation(prob_volume.squeeze(1), fused_depth.squeeze(1), depth_bounds[:,0], depth_intervals, self.device, method=self.cfg["fusion"]["conf_method"])

        # clean up gpu mem
        del depth_vol
        del depth_intervals
        torch.cuda.empty_cache()

        outputs = {
                "fused_depth": fused_depth,
                "fused_conf": fused_conf,
                "prob_volume": prob_volume,
                "depth_bounds": depth_bounds
                }

        return outputs

