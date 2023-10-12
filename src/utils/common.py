import numpy as np
import torch
import sys
import os
import cv2
from random import randint, seed
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
from scipy.linalg import null_space
import time
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#import src.cpp_utils.render.build.render_tgt_volume as rtv

def round_nearest(num: float, decimal: int = 0) -> int:
    """Rounds a floating point number to the nearest decimal place.

    Args:
        num: Float to be rounded.
        decimal: Decimal place to round to.

    Returns:
        The given number rounded to the nearest decimal place.

    Examples:
        >>> round_nearest(11.1)
        11
        >>> round_nearest(15.7)
        16
        >>> round_nearest(2.5)
        2
        >>> round_nearest(3.5)
        3
        >>> round_nearest(14.156, 1)
        14.2
        >>> round_nearest(15.156, 1)
        15.2
        >>> round_nearest(15.156, 2)
        15.16
    """

    return np.round(num+10**(-len(str(num))-1), decimal)


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def non_zero_std(maps, device, dim=1, keepdim=False):
    batch_size, views, height, width = maps.shape
    valid_map = torch.ne(maps, 0.0).to(torch.float32).to(device)
    valid_count = torch.sum(valid_map, dim=1, keepdim=keepdim)+1e-7
    mean = torch.div(torch.sum(maps,dim=1, keepdim=keepdim), valid_count).reshape(batch_size, 1, height, width).repeat(1,views,1,1)
    mean = torch.mul(valid_map, mean)

    std = torch.sub(maps, mean)
    std = torch.square(std)
    std = torch.sum(std, dim=1, keepdim=keepdim)
    std = torch.div(std, valid_count)
    std = torch.sqrt(std)

    return std

def avg_stats(path, stats_file, output_file, avg_path):
    mean_abs_err = []
    in_mean_abs_err = []
    in_mean_depth = []
    ae = []
    se = []
    ise = []
    rad = []
    wce_flat = []

    gt_cover_perc = []
    input_cover_perc = []

    window_radius = []
    window_radius_perc = []
    wce_clipped = []

    in_prec = []
    out_prec = []
    in_prec_oracle = []
    out_prec_oracle = []

    in_rec = []
    out_rec = []
    in_rec_oracle = []
    out_rec_oracle = []

    # store data
    with open(stats_file,'r') as f:
        lines = f.readlines()

        num_lines = len(lines)
        i = 0
        N = 0
        while( i<num_lines ):
            N = int(lines[i])+1
            mean_abs_err.append(float(lines[i+1]))
            in_mean_abs_err.append(float(lines[i+2]))

            a = [ float(a) for a in lines[i+3].split(',') ]
            ae.extend(a)

            s = [ float(s) for s in lines[i+4].split(',') ]
            se.extend(s)

            ia = [ float(ia) for ia in lines[i+5].split(',') ]
            ise.extend(ia)

            ra = [ float(ra) for ra in lines[i+6].split(',') ]
            rad.extend(ra)

            wc = [ float(wc) for wc in lines[i+7].split(',') ]
            wce_flat.extend(wc)

            gt_cover_perc.append(float(lines[i+8]))
            input_cover_perc.append(float(lines[i+9]))

            # range width percentage
            rp = [ float(s) for s in lines[i+10].split(',') ]
            window_radius_perc.extend(rp)

            # standard deviation from center
            wcec = [ float(wcec) for wcec in lines[i+11].split(',') ]
            wce_clipped.extend(wcec)

            # Precision
            ip = [ float(s) for s in lines[i+12].split(',') ]
            in_prec.append(ip)
            op = [ float(s) for s in lines[i+13].split(',') ]
            out_prec.append(op)
            ipo = [ float(s) for s in lines[i+14].split(',') ]
            in_prec_oracle.append(ipo)
            opo = [ float(s) for s in lines[i+15].split(',') ]
            out_prec_oracle.append(opo)

            # Recall
            irec = [ float(s) for s in lines[i+16].split(',') ]
            in_rec.append(irec)
            orec = [ float(s) for s in lines[i+17].split(',') ]
            out_rec.append(orec)
            iro = [ float(s) for s in lines[i+18].split(',') ]
            in_rec_oracle.append(iro)
            oro = [ float(s) for s in lines[i+19].split(',') ]
            out_rec_oracle.append(oro)

            i += 20

    mae = np.mean(mean_abs_err)
    in_mae = np.mean(in_mean_abs_err)
    gip = np.mean(gt_cover_perc)
    iip = np.mean(input_cover_perc)

    in_prec = np.mean(in_prec, 0)
    out_prec = np.mean(out_prec, 0)
    in_prec_oracle = np.mean(in_prec_oracle, 0)
    out_prec_oracle = np.mean(out_prec_oracle, 0)

    in_rec = np.mean(in_rec, 0)
    out_rec = np.mean(out_rec, 0)
    in_rec_oracle = np.mean(in_rec_oracle, 0)
    out_rec_oracle = np.mean(out_rec_oracle, 0)
    perc = np.array(list(range(5,105,5)))

    with open(output_file, 'w') as of:
        of.write("Mean Abs Error: {:0.5f}\n".format(mae))
        of.write("Input Mean Abs Error: {:0.5f}\n".format(in_mae))
        of.write("GT Inclusion: {:0.2f}\n".format(gip))
        of.write("Input Inclusion: {:0.2f}\n".format(iip))


    ### Signed Error Histogram ###
    plt.hist(se, bins=100)
    plt.title("Signed Error")
    plt.xlabel("distance")
    plt.axvline(np.mean(se), linestyle='--', color='red')
    plt.savefig(os.path.join(avg_path,"signed_error.png"))
    plt.close()

    # input signed error histogram
    plt.hist(ise, bins=100)
    plt.title("Input Signed Error")
    plt.xlabel("distance")
    plt.axvline(np.mean(ise), linestyle='--', color='red')
    plt.savefig(os.path.join(avg_path,"signed_error_input.png"))
    plt.close()

    # window center error histogram
    wce = np.asarray(wce_clipped)
    under = np.where((wce < -1), 1, 0)
    num_under = np.sum(under)
    over = np.where((wce > 1), 1, 0)
    num_over = np.sum(over)
    wce_len = len(wce)
    num_inside = wce_len - (num_under+num_over)
    plt.hist(wce, bins=100)
    plt.title("Window Center Error ({:0.2f}|{:0.2f}|{:0.2f})".format((num_under/wce_len), (num_inside/wce_len), (num_over/wce_len)))
    plt.xlabel("error / window-radius")
    plt.axvline(np.mean(wce), linestyle='--', color='red')
    plt.savefig(os.path.join(avg_path,"window_center_error.png"))
    plt.close()

    # window radius percentage
    plt.hist(window_radius_perc, bins=100)
    plt.title("Window Radius Percentage")
    plt.xlabel("radius percentage")
    plt.savefig(os.path.join(avg_path,"window_radius_perc.png"))
    plt.close()

    # radius vs abs error
    ratio = np.asarray(ae) / np.asarray(rad)
    ratio = np.clip(ratio, 0, 4)
    plt.hist(ratio, bins=100)
    plt.title("Absolute Error vs. Window Radius")
    plt.xlabel("error/radius")
    plt.savefig(os.path.join(avg_path,"radius_error.png"))
    plt.close()

    # window center error vs abs error
    wcae_flat = np.abs(np.asarray(wce_flat))
    plt.scatter(ae, wcae_flat, s=0.5)
    plt.title("Absolute Error vs. Window Center Absolute Error")
    plt.xlabel("absolute error")
    plt.ylabel("window center absolute error")
    plt.savefig(os.path.join(avg_path,"wc_error.png"))
    plt.close()

    # plot Precision
    plt.plot(perc, in_prec, label="input")
    plt.plot(perc, in_prec_oracle, label="input_oracle")
    plt.plot(perc, out_prec, label="output")
    plt.plot(perc, out_prec_oracle, label="output_oracle")
    plt.title("Precision")
    plt.xlabel("density")
    plt.ylabel("inlier %")
    plt.legend()
    plt.savefig(os.path.join(avg_path,"precision.png"))
    plt.close()

    # plot Recall
    plt.plot(perc, in_rec, label="input")
    plt.plot(perc, in_rec_oracle, label="input_oracle")
    plt.plot(perc, out_rec, label="output")
    plt.plot(perc, out_rec_oracle, label="output_oracle")
    plt.title("Recall")
    plt.xlabel("density")
    plt.ylabel("inlier %")
    plt.legend()
    plt.savefig(os.path.join(avg_path,"recall.png"))
    plt.close()

def window_visuals(input_depth, input_conf, fused_depth, fused_conf, gt_depth, depth_bounds, view_num, path, n=10, offset=100):
    _,_,h,w = fused_depth.shape
    np.random.seed(int(time.time()))
    rows = np.random.randint(offset, h-offset, n)
    cols = np.random.randint(offset, w-offset, n)

    for r,c in zip(rows,cols):
        min_b,max_b = depth_bounds[0,:,r,c].cpu().numpy()
        radius = float((max_b-min_b) / 2.0)
        center = (min_b+max_b)/2

        in_depths = input_depth[0,:,r,c].cpu().numpy() 
        min_d = np.min(in_depths)
        max_d = np.max(in_depths)
        spread = max_d - min_d
        in_confs = input_conf[0,:,r,c].cpu().numpy() 

        inputs = np.asarray([ (d,c) for d,c in zip(in_depths,in_confs) if d > 0 ])
        gt = gt_depth[0,0,r,c].cpu().numpy()

        if(gt<=0):
            continue

        fd = fused_depth[0,0,r,c].cpu().numpy()
        x1, y1 = [center-radius, center-radius], [1.0, -0.1]
        x2, y2 = [center+radius, center+radius], [1.0, -0.1]

        #fg,ax = plt.subplots()
        #ax.set_aspect(10*spread)
        plt.bar(inputs[:,0], inputs[:,1], color="#547972", edgecolor="black", width=(0.04*radius), linewidth=(0.01*radius))
        plt.arrow(center-radius-(0.1*radius), 0, 2*(radius+(0.2*radius)), 0, shape="right", width=(0.001*radius), head_width=(0.005*radius), head_length=(0.05*radius), color="black")
        plt.plot(x1,y1,x2,y2,color="maroon", linestyle="--")
        plt.scatter(fd, 0.01, s=(10*radius), marker="s", label="fused depth", color="#BFA7A3", edgecolor="black")
        plt.scatter(gt, 0.01, s=(10*radius), marker="^", label="ground truth", color="#EAB464", edgecolor="black")
        #plt.legend()
        #plt.axis("off")
        plt.savefig(os.path.join(path,"{:04d}_window_[{}-{}]_.png".format(view_num,r,c)),dpi=300,bbox_inches="tight",pad_inches=0)
        plt.close()

def error_stats(input_depth, fused_depth, gt_depth, view_num, scene, path, plot=False):
    if (scene == "Barn"):
        th = 0.005
    elif(scene == "Caterpillar"):
        th = 0.0025
    elif(scene == "Church"):
        th = 0.0125
    elif(scene == "Courthouse"):
        th = 0.0125
    elif (scene =="Ignatius"):
        th = 0.0015
    elif(scene == "Meetingroom"):
        th = 0.005
    elif(scene == "Truck"):
        th = 0.0025
    else:
        th = 0.125 #DTU

    ### Setup ###
    input_depth = input_depth.cpu().numpy()
    fused_depth = fused_depth.cpu().numpy()
    gt_depth = gt_depth.cpu().numpy()

    batch_size, channels, height, width = gt_depth.shape
    max_dist = 1
    error_th = 0.3

    # compute gt mask and number of valid pixels
    gt_mask = np.not_equal(gt_depth, 0.0).astype(np.double)
    gt_mask_flat = np.copy(gt_mask).flatten()
    num_gt = np.sum(gt_mask, axis=(1, 2, 3)) + 1e-7

    ### Error ###
    # output
    signed_error = fused_depth - gt_depth
    abs_error = np.abs(signed_error)

    # measure error percentages
    ae = np.copy(abs_error).flatten()
    ae_flat = np.array( [x for x,g in zip(ae,gt_mask_flat) if g != 0] )
    e_0_125 = np.sum(np.where(ae_flat < th, 1,0))
    e_0_25 = np.sum(np.where(ae_flat < 2*th, 1,0))
    e_0_5 = np.sum(np.where(ae_flat < 4*th, 1,0))
    e_1_0 = np.sum(np.where(ae_flat < 8*th, 1,0))
    pe = np.asarray([e_0_125,e_0_25,e_0_5,e_1_0])


    # input
    input_signed_error = input_depth - gt_depth
    input_abs_error = np.abs(input_signed_error)

    # measure input error percentages
    iae = np.copy(input_abs_error).flatten()
    iae_flat = np.array( [x for x,g in zip(iae,gt_mask_flat) if g != 0] )
    ie_0_125 = np.sum(np.where(iae_flat < th, 1,0))
    ie_0_25 = np.sum(np.where(iae_flat < 2*th, 1,0))
    ie_0_5 = np.sum(np.where(iae_flat < 4*th, 1,0))
    ie_1_0 = np.sum(np.where(iae_flat < 8*th, 1,0))
    ipe = np.asarray([ie_0_125,ie_0_25,ie_0_5,ie_1_0])


    ### Apply GT Mask ###
    # output
    signed_error = gt_mask * signed_error
    abs_error = gt_mask * abs_error
    mean_abs_error = np.mean(abs_error[0,0])

    # input
    input_signed_error = gt_mask * input_signed_error
    input_abs_error = gt_mask * input_abs_error
    input_mean_abs_error = np.mean(input_abs_error[0,0])


    ### Plots ###
    if (plot):
        # output absolute error
        ad = abs_error[0,0,:,:]
        ad = np.clip(ad, 0, max_dist)
        # input absolute error
        ise = input_abs_error[0,0,:,:]
        ise = np.clip(ise, 0, max_dist)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, aspect='equal')
        shw = plt.imshow(ad)
        plt.hot()
        plt.axis('off')
        #divider = make_axes_locatable(ax)
        #cax1 = divider.append_axes("right", size="5%", pad=0.05)
        #bar = plt.colorbar(shw, cax=cax1)
        #bar.set_label("error (mm)")
        plt.savefig(os.path.join(path,"{:04d}_abs_error.png".format(view_num)), dpi=300, bbox_inches='tight')
        plt.close()

        shw = plt.imshow(ise)
        plt.hot()
        plt.axis('off')
        plt.savefig(os.path.join(path,"{:04d}_in_abs_error.png".format(view_num)), dpi=300, bbox_inches='tight')
        plt.close()

        #   # output binary error
        #   under_est = np.less_equal(signed_error, (-error_th))
        #   over_est = np.greater_equal(signed_error, error_th) * 0.5
        #   binary_error = under_est + over_est
        #   # input binary error
        #   under_est = np.less_equal(input_signed_error, (-error_th))
        #   over_est = np.greater_equal(input_signed_error, error_th) * 0.5
        #   input_binary_error = under_est + over_est

        #   plt.subplot(1,2,1)
        #   cmap = matplotlib.colors.ListedColormap(['black','red','yellow'])
        #   plt.imshow(binary_error[0,0,:,:], cmap=cmap)
        #   plt.title("Fused".format(error_th))
        #   plt.axis('off')
        #   plt.subplot(1,2,2)
        #   plt.imshow(input_binary_error[0,0,:,:], cmap=cmap)
        #   plt.title("Input".format(error_th))
        #   #bar = plt.colorbar()
        #   #bar.set_label("Error")
        #   plt.axis('off')
        #   plt.savefig(os.path.join(path,"{:04d}_binary_error.png".format(view_num)), dpi=300, bbox_inches='tight')
        #   plt.close()

    return (mean_abs_error, input_mean_abs_error), pe, ipe, num_gt

def compute_auc(input_depth, fused_depth, input_conf, fused_conf, gt_depth, path, view_num, plot=False):
    # convert to numpy
    input_depth = input_depth.cpu().numpy()
    fused_depth = fused_depth.cpu().numpy()
    input_conf = input_conf.cpu().numpy()
    fused_conf = fused_conf.cpu().numpy()
    gt_depth = gt_depth.cpu().numpy()

    height, width = input_depth.shape

    gt_mask = np.not_equal(gt_depth, 0.0)
    gt_count = int(np.sum(gt_mask)+1e-7)

    # flatten to 1D tensor
    input_depth = input_depth.flatten()
    fused_depth = fused_depth.flatten()
    input_conf = input_conf.flatten()
    fused_conf = fused_conf.flatten()
    gt_depth = gt_depth.flatten()

    ##### INPUT #####
    # sort all tensors by confidence value
    indices = np.argsort(input_conf)
    indices = indices[::-1]
    input_depth = np.take(input_depth, indices=indices, axis=0)
    input_gt_depth = np.take(gt_depth, indices=indices, axis=0)
    input_gt_indices = np.nonzero(input_gt_depth)[0]
    input_depth = np.take(input_depth, indices=input_gt_indices, axis=0)
    input_gt_depth = np.take(input_gt_depth, indices=input_gt_indices, axis=0)
    input_error = np.abs(input_depth-input_gt_depth)


    # sort orcale curves by error
    indices = np.argsort(input_error)
    input_oracle = np.take(input_error, indices=indices, axis=0)
    input_oracle_gt = np.take(input_gt_depth, indices=indices, axis=0)
    input_oracle_indices = np.nonzero(input_oracle_gt)[0]
    input_oracle_error = np.take(input_oracle, indices=input_oracle_indices, axis=0)

    ##### OUTPUT #####
    # sort all tensors by confidence value
    indices = np.argsort(fused_conf)
    indices = indices[::-1]
    fused_depth = np.take(fused_depth, indices=indices, axis=0)
    output_gt_depth = np.take(gt_depth, indices=indices, axis=0)
    output_gt_indices = np.nonzero(output_gt_depth)[0]
    fused_depth = np.take(fused_depth, indices=output_gt_indices, axis=0)
    output_gt_depth = np.take(output_gt_depth, indices=output_gt_indices, axis=0)
    output_error = np.abs(fused_depth - output_gt_depth)

    # sort orcale curves by error
    indices = np.argsort(output_error)
    output_oracle = np.take(output_error, indices=indices, axis=0)
    output_oracle_gt = np.take(output_gt_depth, indices=indices, axis=0)
    output_oracle_indices = np.nonzero(output_oracle_gt)[0]
    output_oracle_error = np.take(output_oracle, indices=output_oracle_indices, axis=0)


    # build density vector
    num_gt_points = input_gt_depth.shape[0]
    perc = np.array(list(range(5,105,5)))
    density = np.array((perc/100) * (num_gt_points), dtype=np.int32)

    input_roc = np.zeros(density.shape)
    input_oracle = np.zeros(density.shape)
    output_roc = np.zeros(density.shape)
    output_oracle = np.zeros(density.shape)

    for i,k in enumerate(density):
        # compute input absolute error chunk
        ie = input_error[0:k]
        ioe = input_oracle_error[0:k]
        if (ie.shape[0] == 0):
            input_roc[i] = 0.0
        else :
            input_roc[i] = np.mean(ie)

        if (ioe.shape[0] == 0):
            input_oracle[i] = 0.0
        else:
            input_oracle[i] = np.mean(ioe)

        # compute input absolute error chunk
        oe = output_error[0:k]
        ooe = output_oracle_error[0:k]
        if (oe.shape[0] == 0):
            output_roc[i] = 0.0
        else :
            output_roc[i] = np.mean(oe)

        if (ooe.shape[0] == 0):
            output_oracle[i] = 0.0
        else:
            output_oracle[i] = np.mean(ooe)

    # comput AUC
    input_auc = np.trapz(input_roc, dx=5)
    output_auc = np.trapz(output_roc, dx=5)

    if(plot):
        # plot ROC density errors
        plt.plot(perc, input_roc, label="input")
        plt.plot(perc, input_oracle, label="input_oracle")
        plt.plot(perc, output_roc, label="output")
        plt.plot(perc, output_oracle, label="output_oracle")
        plt.title("ROC Error")
        plt.xlabel("density")
        plt.ylabel("absolte error")
        plt.legend()
        plt.savefig(os.path.join(path,"{:04d}_roc_error.png".format(view_num)))
        plt.close()

    return (input_auc, output_auc)
        
def roc_curve(in_depth, out_depth, in_conf, out_conf, gt_depth, path, stats_file, view_num, scene, device):
    if (scene == "Barn"):
        th = 0.01
    elif(scene == "Caterpillar"):
        th = 0.005
    elif(scene == "Church"):
        th = 0.025
    elif(scene == "Courthouse"):
        th = 0.025
    elif (scene =="Ignatius"):
        th = 0.003
    elif(scene == "Meetingroom"):
        th = 0.01
    elif(scene == "Truck"):
        th = 0.005
    elif(scene == "none"): # DTU
        th = 2
    else:
        th = 0.02

    height, width = in_depth.shape

    valid_map = torch.ne(gt_depth, 0.0).to(torch.float32).to(device)
    valid_count = torch.sum(valid_map)+1e-7

    # flatten to 1D tensor
    in_depth = torch.flatten(in_depth)
    in_conf = torch.flatten(in_conf)
    out_depth = torch.flatten(out_depth)
    out_conf = torch.flatten(out_conf)
    gt_depth = torch.flatten(gt_depth)

    ##### INPUT #####
    # sort all tensors by confidence value
    (in_conf,indices) = in_conf.sort(descending=True)
    in_depth = torch.gather(in_depth, dim=0, index=indices)
    in_gt_depth = torch.gather(gt_depth, dim=0, index=indices)
    # pull only gt values
    in_gt_indices = torch.nonzero(in_gt_depth).flatten()
    in_depth = torch.index_select(in_depth, dim=0, index=in_gt_indices)
    in_gt_depth = torch.index_select(in_gt_depth, dim=0, index=in_gt_indices)

    # sort orcale curves by error
    in_oracle = torch.abs(in_depth-in_gt_depth)
    (in_oracle,indices) = in_oracle.sort(descending=False)
    in_oracle_gt = torch.gather(in_gt_depth, dim=0, index=indices)
    # pull only gt values
    in_oracle_indices = torch.nonzero(in_oracle_gt).flatten()
    in_oracle = torch.index_select(in_oracle, dim=0, index=in_oracle_indices)

    ##### OUTPUT #####
    # sort all tensors by confidence value
    (out_conf,indices) = out_conf.sort(descending=True)
    out_depth = torch.gather(out_depth, dim=0, index=indices)
    out_gt_depth = torch.gather(gt_depth, dim=0, index=indices)
    # pull only gt values
    out_gt_indices = torch.nonzero(out_gt_depth).flatten()
    out_depth = torch.index_select(out_depth, dim=0, index=out_gt_indices)
    out_gt_depth = torch.index_select(out_gt_depth, dim=0, index=out_gt_indices)

    out_oracle = torch.abs(out_depth-out_gt_depth)
    (out_oracle,indices) = out_oracle.sort(descending=False)
    out_oracle_gt = torch.gather(out_gt_depth, dim=0, index=indices)
    # pull only gt values
    out_oracle_indices = torch.nonzero(out_oracle_gt).flatten()
    out_oracle = torch.index_select(out_oracle, dim=0, index=out_oracle_indices)

    # build density vector
    num_gt_points = in_gt_depth.shape[0]
    perc = np.array(list(range(5,105,5)))
    density = np.array((perc/100) * (num_gt_points), dtype=np.int32)

    in_prec = np.zeros(density.shape)
    in_prec_oracle = np.zeros(density.shape)
    out_prec = np.zeros(density.shape)
    out_prec_oracle = np.zeros(density.shape)

    in_rec = np.zeros(density.shape)
    in_rec_oracle = np.zeros(density.shape)
    out_rec = np.zeros(density.shape)
    out_rec_oracle = np.zeros(density.shape)

    for i,k in enumerate(density):
        # compute input absolute error chunk
        iae = torch.abs(in_gt_depth[0:k] - in_depth[0:k])
        num_inliers = torch.sum(torch.le(iae, th).to(torch.float32).to(device))

        in_prec[i] = num_inliers / k
        in_rec[i] = num_inliers / valid_count

        # compute input oracle chunk
        in_prec_oracle[i] = torch.sum(torch.le(in_oracle[0:k], th).to(torch.float32).to(device)) / k
        in_rec_oracle[i] = torch.sum(torch.le(in_oracle[0:k], th).to(torch.float32).to(device)) / valid_count

        # compute output absolute error chunk
        oae = torch.abs(out_gt_depth[0:k] - out_depth[0:k])
        num_inliers = torch.sum(torch.le(oae, th).to(torch.float32).to(device))
        out_prec[i] = num_inliers / k
        out_rec[i] = num_inliers / valid_count

        # compute output oracle chunk
        out_prec_oracle[i] = torch.sum(torch.le(out_oracle[0:k], th).to(torch.float32).to(device)) / k
        out_rec_oracle[i] = torch.sum(torch.le(out_oracle[0:k], th).to(torch.float32).to(device)) / valid_count

    in_prec_str = ",".join([ "{:0.5f}".format(r) for r in in_prec ])
    out_prec_str = ",".join([ "{:0.5f}".format(r) for r in out_prec ])
    in_prec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in in_prec_oracle ])
    out_prec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in out_prec_oracle ])

    in_rec_str = ",".join([ "{:0.5f}".format(r) for r in in_rec ])
    out_rec_str = ",".join([ "{:0.5f}".format(r) for r in out_rec ])
    in_rec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in in_rec_oracle ])
    out_rec_oracle_str = ",".join([ "{:0.5f}".format(r) for r in out_rec_oracle ])

    # store data
    with open(stats_file,'a') as f:
        f.write("{}\n".format(in_prec_str))
        f.write("{}\n".format(out_prec_str))
        f.write("{}\n".format(in_prec_oracle_str))
        f.write("{}\n".format(out_prec_oracle_str))

        f.write("{}\n".format(in_rec_str))
        f.write("{}\n".format(out_rec_str))
        f.write("{}\n".format(in_rec_oracle_str))
        f.write("{}\n".format(out_rec_oracle_str))

    return

def train_abs_error(in_depth, gt_depth, path, stats_file, view_num, device):
    batch_size, channels, height, width = gt_depth.shape
    gt_mask = (torch.ne(gt_depth, 0.0)).to(torch.float32).to(device)
    denom = torch.sum(gt_mask, dim=[1, 2, 3]) + 1e-7

    ### Signed-MAE Input Histogram ###
    max_dist = 10
    signed_in_dist = torch.flatten(torch.sub(in_depth, gt_depth)).cpu().numpy()
    mask = torch.flatten(gt_mask).cpu().numpy()
    signed_in_dist = np.array( [x for x,g in zip(signed_in_dist,mask) if g != 0] )
    signed_in_dist = np.clip(signed_in_dist, -max_dist, max_dist)

    # store data
    ise_str = ",".join([ "{:0.5f}".format(r) for r in signed_in_dist ])

    with open(stats_file,'a') as f:
        f.write("{}\n".format(ise_str))

    ### Signed Error Input Histogram ###
    #plt.hist(signed_in_dist, bins=100)
    #plt.title("Input Signed Error")
    #plt.xlabel("distance")
    #plt.axvline(np.mean(signed_in_dist), linestyle='--', color='red')
    #plt.savefig(os.path.join(path,"{:04d}_signed_error_input.png".format(view_num)))
    #plt.close()

def avg_train_stats(path, stats_file, scan_num):
    iad = []
    # store data
    with open(stats_file,'r') as f:
        lines = f.readlines()

        num_lines = len(lines)
        i = 0
        while( i<num_lines ):
            ia = [ float(ia) for ia in lines[i].split(',') ]
            iad.extend(ia)
            i += 1

    ### Signed Error Input Histogram ###
    plt.hist(iad, bins=100)
    plt.title("Input Signed Error")
    plt.xlabel("distance")
    plt.axvline(np.mean(iad), linestyle='--', color='red')
    plt.savefig(os.path.join(path,"signed_error_input_{:03d}.png".format(scan_num)))
    plt.close()

def print_gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t- (a+r)
    print("Free: {:0.4f} GB".format(f/(1024*1024*1024)))

def show_distrib(volume, index, name, path="./distribs"):
    if not os.path.exists(path):
        os.makedirs(path)

    row,col = index
    volume = volume.cpu().numpy()
    distrib = volume[0,0,:,row,col]
    plt.plot(distrib)

    filename = "{}_({}-{}).png".format(name,row,col)
    file_path = os.path.join(path, filename)

    plt.savefig(file_path)
    plt.close()

def save_slices(volume, index, name, path="./slices"):
    volume = volume.cpu().numpy()

    path = os.path.join(path,name)

    if not os.path.exists(path):
        os.makedirs(path)

    for vslice in range(volume.shape[3]):
        slc = np.flip(volume[0,index,:,vslice,:], axis=0)
        m = np.amax(slc)
        slc = slc/m*255

        slc_path = os.path.join(path, "{:04d}_{}.png".format(vslice, name))
        cv2.imwrite(slc_path, slc)

def pre_filter_conf(depths, confs, device='cuda:0', min_conf=0.9):
    mask = (torch.ge(confs, min_conf)).to(torch.float32).to(device)
    return depths*mask

def post_filter_conf(depth_map, conf_map, device='cuda:0', min_conf=0.8):
    mask = (torch.ge(conf_map, min_conf)).to(torch.float32).to(device)
    return depth_map*mask, mask

def post_filter_gt(depth_map, conf_map, gt_depth, device='cuda:0'):
    mask = (torch.ne(gt_depth, 0.0)).to(torch.float32).to(device)
    return depth_map*mask, mask

def topk_filter(depth_map, conf_map, device='cuda:0', percent=0.3):
    height, width = depth_map.shape

    # calculate k number of points to keep
    valid_map = torch.ne(conf_map, 0.0).to(torch.float32)
    valid_count = torch.sum(valid_map)
    k = int(percent * valid_count)

    # flatten and grab top-k indices
    filter_prob = conf_map.reshape(-1)
    (vals, indices) = torch.topk(filter_prob, k=k, dim=0)

    # get min confidence value
    min_conf = torch.min(vals)

    # filter by min conf value
    filt = (torch.ge(conf_map, min_conf)).to(torch.float32).to(device)

    return depth_map*filt, filt

def topk_strict_filter(depth_map, filter_prob, device='cuda:0', percent=0.3):
    height, width = depth_map.shape

    # calculate k number of points to keep
    valid_map = torch.ne(filter_prob, 0.0).to(torch.float32)
    valid_count = torch.sum(valid_map)
    k = int(percent * valid_count)

    # flatten and grab top-k indices
    filter_prob = filter_prob.reshape(-1)
    (vals, indices) = torch.topk(filter_prob, k=k, dim=0)

    # calculate the row and column given each index
    row_indices = torch.div(indices, width, rounding_mode="floor").unsqueeze(-1)
    col_indices = torch.remainder(indices, width).unsqueeze(-1)

    # concatenate the [r,c] indices into a single tensor
    indices = torch.cat((row_indices, col_indices), dim=1)
    filt = torch.zeros((height,width), dtype=torch.uint8).to(device)

    # set top-k indices to 1
    for r,c in indices:
        filt[r,c] = 1

    return depth_map*filt, filt

#   def render_into_ref(depths, confs, cams):
#       shape = depths.shape
#       views = shape[0]
#       rcam = cams[0].flatten().tolist()
#   
#       rendered_depths = [depths[0]]
#       rendered_confs = [confs[0]]
#   
#       for v in range(1,views):
#           tcam = cams[v].flatten().tolist()
#           depth_map = depths[v].flatten().tolist()
#           conf_map = confs[v].flatten().tolist()
#   
#           rendered_map = np.array(rtv.render_to_ref(list([shape[1],shape[2]]),depth_map,conf_map,rcam,tcam))
#           rendered_depth = rendered_map[:(shape[1]*shape[2])].reshape((shape[1],shape[2]))
#           rendered_conf = rendered_map[(shape[1]*shape[2]):].reshape((shape[1],shape[2]))
#   
#           rendered_depths.append(rendered_depth)
#           rendered_confs.append(rendered_conf)
#   
#       rendered_depths = np.asarray(rendered_depths)
#       rendered_confs = np.asarray(rendered_confs)
#   
#       return rendered_depths, rendered_confs

def warp(depths, confs, cams):
    views, _, height, width = depths.shape

    rendered_depths = [depths[0]]
    rendered_confs = [confs[0]]

    # grab intrinsics and extrinsics from reference view
    P_ref = cams[0,0,:,:]
    K_ref = cams[0,1,:,:]
    K_ref[3,:] = torch.tensor([0,0,0,1])

    R_ref = P_ref[:3,:3]
    t_ref = P_ref[:3,3:4]
    C_ref = torch.matmul(-R_ref.transpose(0,1), t_ref)
    z_ref = R_ref[2:3,:3].reshape(1,1,1,3).repeat(height,width,1,1)
    C_ref = C_ref.reshape(1,1,3).repeat(height, width, 1)

    for v in range(1,views):
        depth_map = depths[v]
        conf_map = confs[v]

        # get intrinsics and extrinsics from target view
        P_tgt = cams[v,0,:,:]
        K_tgt = cams[v,1,:,:]
        K_tgt[3,:] = torch.tensor([0,0,0,1])

        bwd_proj = torch.matmul(torch.inverse(P_tgt), torch.inverse(K_tgt)).to(torch.float32)
        fwd_proj = torch.matmul(K_ref, P_ref).to(torch.float32)
        bwd_rot = bwd_proj[:3,:3]
        bwd_trans = bwd_proj[:3,3:4]
        proj = torch.matmul(fwd_proj, bwd_proj)
        rot = proj[:3,:3]
        trans = proj[:3,3:4]

        y, x = torch.meshgrid([ torch.arange(0, height,dtype=torch.float32),
                                torch.arange(0, width, dtype=torch.float32)],
                                indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height*width), x.reshape(height*width)
        homog = torch.stack((x,y,torch.ones_like(x)))

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords * depth_map.reshape(1,-1)
        world_coords = world_coords + bwd_trans.reshape(3,1)
        world_coords = torch.movedim(world_coords, 0, 1)
        world_coords = world_coords.reshape(height, width,3)

        # get pixel projection
        rot_coords = torch.matmul(rot, homog)
        proj_3d = rot_coords * depth_map.reshape(1,-1)
        proj_3d = proj_3d + trans.reshape(3,1)
        proj_2d = proj_3d[:2,:] / proj_3d[2:3,:]
        proj_2d = (torch.movedim(proj_2d,0,1)).to(torch.long)
        proj_2d = torch.flip(proj_2d, dims=(1,))

        # compute projected depth
        proj_depth = torch.sub(world_coords, C_ref).unsqueeze(-1)
        proj_depth = torch.matmul(z_ref, proj_depth).reshape(height,width)
        proj_depth = proj_depth.reshape(-1,1)

        # mask out invalid indices
        mask =  torch.where(proj_2d[:,0] < height, 1, 0) * \
                torch.where(proj_2d[:,0] >= 0, 1, 0) * \
                torch.where(proj_2d[:,1] < width, 1, 0) * \
                torch.where(proj_2d[:,1] >= 0, 1, 0)
        inds = torch.where(mask)[0]
        proj_2d = torch.index_select(proj_2d, dim=0, index=inds)
        proj_2d = (proj_2d[:,0] * width) + proj_2d[:,1]
        proj_depth = torch.index_select(proj_depth, dim=0, index=inds).squeeze()
        proj_conf = torch.index_select(conf_map.flatten(), dim=0, index=inds).squeeze()
        
        warped_depth = torch.zeros(height * width)
        warped_depth[proj_2d] = proj_depth
        warped_depth = warped_depth.reshape(height,width)

        warped_conf = torch.zeros(height * width)
        warped_conf[proj_2d] = proj_conf
        warped_conf = warped_conf.reshape(height,width)

        rendered_depths.append(warped_depth.unsqueeze(0))
        rendered_confs.append(warped_conf.unsqueeze(0))

    rendered_depths = torch.cat(rendered_depths)
    rendered_confs = torch.cat(rendered_confs)

    return rendered_depths, rendered_confs

def warp_to_tgt(tgt_depth, tgt_conf, ref_cam, tgt_cam, depth_planes, depth_vol):
    batch_size, views, height, width = tgt_depth.shape
    # grab intrinsics and extrinsics from reference view
    P_ref = ref_cam[:,0,:,:]
    K_ref = ref_cam[:,1,:,:]
    K_ref[:,3,:] = torch.tensor([0,0,0,1])

    # get intrinsics and extrinsics from target view
    P_tgt = tgt_cam[:,0,:,:]
    K_tgt = tgt_cam[:,1,:,:]
    K_tgt[:,3,:] = torch.tensor([0,0,0,1])

    R_tgt = P_tgt[:,:3,:3]
    t_tgt = P_tgt[:,:3,3:4]
    C_tgt = torch.matmul(-R_tgt.transpose(1,2), t_tgt)
    z_tgt = R_tgt[:,2:3,:3].reshape(batch_size,1,1,1,1,3).repeat(1,depth_planes, height,width,1,1)
    
    with torch.no_grad():
        # shape camera center vector
        C_tgt = C_tgt.reshape(batch_size,1,1,1,3).repeat(1, depth_planes, height, width, 1)

        bwd_proj = torch.matmul(torch.inverse(P_ref), torch.inverse(K_ref)).to(torch.float32)
        fwd_proj = torch.matmul(K_tgt, P_tgt).to(torch.float32)

        bwd_rot = bwd_proj[:,:3,:3]
        bwd_trans = bwd_proj[:,:3,3:4]

        proj = torch.matmul(fwd_proj, bwd_proj)
        rot = proj[:,:3,:3]
        trans = proj[:,:3,3:4]

        y, x = torch.meshgrid([torch.arange(0, height,dtype=torch.float32,device=tgt_depth.device),
                                     torch.arange(0, width, dtype=torch.float32, device=tgt_depth.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.reshape(height*width), x.reshape(height*width)
        homog = torch.stack((x,y,torch.ones_like(x)))
        homog = torch.unsqueeze(homog, 0).repeat(batch_size,1,1)

        # get world coords
        world_coords = torch.matmul(bwd_rot, homog)
        world_coords = world_coords.unsqueeze(2).repeat(1,1,depth_planes,1)
        depth_vol = depth_vol.reshape(batch_size,1,depth_planes,-1)
        world_coords = world_coords * depth_vol
        world_coords = world_coords + bwd_trans.reshape(batch_size,3,1,1)
        world_coords = torch.movedim(world_coords, 1, 3)
        world_coords = world_coords.reshape(batch_size, depth_planes, height, width,3)

        # get pixel projection
        rot_coords = torch.matmul(rot, homog)
        rot_coords = rot_coords.unsqueeze(2).repeat(1,1,depth_planes,1)
        proj_3d = rot_coords * depth_vol
        proj_3d = proj_3d + trans.reshape(batch_size,3,1,1)
        proj_2d = proj_3d[:,:2,:,:] / proj_3d[:,2:3,:,:]

        proj_x = proj_2d[:,0,:,:] / ((width-1)/2) - 1
        proj_y = proj_2d[:,1,:,:] / ((height-1)/2) - 1
        proj_2d = torch.stack((proj_x, proj_y), dim=3)
        grid = proj_2d

    proj_depth = torch.sub(world_coords, C_tgt).unsqueeze(-1)
    proj_depth = torch.matmul(z_tgt, proj_depth).reshape(batch_size,depth_planes,height,width)

    warped_depth = F.grid_sample(tgt_depth, grid.reshape(batch_size, depth_planes*height, width, 2), mode='bilinear', padding_mode="zeros", align_corners=False)
    warped_depth = warped_depth.reshape(batch_size, depth_planes, height, width)

    warped_conf = F.grid_sample(tgt_conf, grid.reshape(batch_size, depth_planes*height, width, 2), mode='bilinear', padding_mode="zeros", align_corners=False)
    warped_conf = warped_conf.reshape(batch_size, depth_planes, height, width)

    depth_diff = torch.sub(proj_depth, warped_depth)

    return depth_diff, warped_conf


def project_depth_map(depth: torch.Tensor, cam: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Projects a depth map into a list of 3D points

    Parameters:
        depth: Input depth map to project.
        cam: Camera parameters for input depth map.

    Returns:
        A float Tensor of 3D points corresponding to the projected depth values.
    """
    depth = depth.squeeze(1)

    batch_size, height, width = depth.shape
    cam_shape = cam.shape

    # get camera extrinsics and intrinsics
    P = cam[:,0,:,:]
    K = cam[:,1,:,:]
    K[:,3,:] = torch.tensor([0,0,0,1])

    # construct back-projection from invers matrices
    # separate into rotation and translation components
    bwd_projection = torch.matmul(torch.inverse(P), torch.inverse(K)).to(torch.float32)
    bwd_rotation = bwd_projection[:,:3,:3]
    bwd_translation = bwd_projection[:,:3,3:4]

    # build 2D homogeneous coordinates tensor: [B, 3, H*W]
    with torch.no_grad():
        row_span = torch.arange(0, height, dtype=torch.float32).cuda()
        col_span = torch.arange(0, width, dtype=torch.float32).cuda()
        r,c = torch.meshgrid(row_span, col_span, indexing="ij")
        r,c = r.contiguous(), c.contiguous()
        r,c = r.reshape(height*width), c.reshape(height*width)
        coords = torch.stack((c,r,torch.ones_like(c)))
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1)

    # compute 3D coordinates using the depth map: [B, H*W, 3]
    world_coords = torch.matmul(bwd_rotation, coords)
    depth = depth.reshape(batch_size, 1, -1)
    world_coords = world_coords * depth
    world_coords = world_coords + bwd_translation

    #TODO: make sure index select is differentiable
    #       (there is a backward function but need to find the code..)
    if (mask != None):
        world_coords = torch.index_select(world_coords, dim=2, index=non_zero_inds)
        world_coords = torch.movedim(world_coords, 1, 2)

    # reshape 3D coordinates back into 2D map: [B, H, W, 3]
    #   coords_map = world_coords.reshape(batch_size, height, width, 3)

    return world_coords

