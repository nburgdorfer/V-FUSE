import sys
import math
import numpy as np
from random import seed
import cv2
import torch
import torch.nn.functional as F

from src.utils.common import *

def mean_absolute_diff(gt_depth, estimated_depth, depth_interval):
    """ non zero mean absolute loss for one batch """
    shape = estimated_depth.shape

    # get loss only where we have GT data
    gt_mask = (torch.ne(gt_depth, 0.0)).to(torch.float32).cuda()
    num_gt = torch.sum(gt_mask, dim=[1, 2, 3]) + 1e-7

    # compute L1
    abs_error = torch.abs(gt_depth - estimated_depth)

    # apply gt mask
    abs_error = torch.mul(gt_mask, abs_error)
    mae_loss = torch.sum(abs_error, dim=[1, 2, 3])

    mae_loss = torch.mean(torch.div(mae_loss, num_gt), dim=0)

    return mae_loss

def swr_loss(depth_ranges, gt_depth):
    # compute mean and half-width of the ranges
    window_center = torch.mean(depth_ranges, dim=1, keepdim=True)
    window_radius = torch.div(torch.sub(depth_ranges[:,1], depth_ranges[:,0]).unsqueeze(1), 2)

    # compute the distance from the GT value to the mean
    abs_error = torch.abs(torch.sub(window_center, gt_depth))
    coverage_loss = torch.div(abs_error, window_radius)

    # sum the loss and divde by the number of GT data points
    radius_loss = window_radius

    # get loss only where we have GT data
    gt_mask = (torch.ne(gt_depth, 0.0)).to(torch.float32).cuda()
    num_gt = torch.sum(gt_mask, dim=[1, 2, 3]) + 1e-7

    # apply gt mask
    coverage_loss = torch.mul(gt_mask, coverage_loss)
    radius_loss = torch.mul(gt_mask, radius_loss)

    # average loss values across batches
    radius_loss = torch.sum(radius_loss, dim=[1,2,3])
    radius_loss = torch.mean(torch.div(radius_loss, num_gt), dim=0)
    coverage_loss = torch.sum(coverage_loss,dim=[1,2,3])
    coverage_loss = torch.mean(torch.div(coverage_loss, num_gt), dim=0)

    return radius_loss, coverage_loss

def accuracy(y_true, y_pred, interval, th):
    shape = y_pred.shape
    mask_true = (torch.ne(y_true, 0.0)).to(torch.float32)
    denom = torch.sum(mask_true) + 1e-7
    abs_diff_image = torch.abs(y_true - y_pred) / interval
    less_one_image = mask_true * (torch.le(abs_diff_image, th)).to(torch.float32)
    return torch.sum(less_one_image) / denom

def compute_loss(data, outputs, cfg):
    # compute loss terms
    depth_loss = mean_absolute_diff(data["gt_depth"], outputs["fused_depth"], 1)
    radius_loss, coverage_loss = swr_loss(outputs["depth_bounds"], data["gt_depth"])

    # weight loss terms
    depth_loss *= cfg["loss"]["depth_weight"]
    radius_loss *= cfg["loss"]["radius_weight"]
    coverage_loss *= cfg["loss"]["coverage_weight"]

    # compute accuracy
    depth_interval = data["cams"][0,0,1,3,1].item()
    less_one_accuracy = accuracy(data["gt_depth"], outputs["fused_depth"], depth_interval, th=1.0)
    less_three_accuracy = accuracy(data["gt_depth"], outputs["fused_depth"], depth_interval, th=3.0)

    # sum losses
    total_loss = 0.0
    total_loss += depth_loss
    total_loss += radius_loss
    total_loss += coverage_loss

    loss = {
            "loss": total_loss,
            "depth": depth_loss,
            "radius": radius_loss,
            "coverage": coverage_loss,
            "one_acc": less_one_accuracy,
            "three_acc": less_three_accuracy
            }

    return loss
