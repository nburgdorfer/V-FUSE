import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt
import open3d as o3d
from time import time
from random import seed
import argparse
import scipy.io as sio
from sklearn.neighbors import KDTree
from tqdm import tqdm

from src.utils.io import read_point_cloud, write_point_cloud
from src.utils.preprocess import downsample_cloud
from src.utils.common import round_nearest

seed(5)
np.random.seed(5)
milliseconds = int(time() * 1000)

def remove_close_points(ply, min_point_dist):
    # build KD-Tree of estimated point cloud for querying
    tree = KDTree(np.asarray(ply.points), leaf_size=40)
    (dists, inds) = tree.query(np.asarray(ply.points), k=2)

    # ignore first nearest neighbor (since it is the point itself)
    dists = dists[:,1]
    inds = inds[:,1]

    # get unique set of indices
    ind_pairs = np.asarray([ np.asarray([min(i,inds[i]),max(i,inds[i])]) for i in range(len(inds)) ])
    close_inds = np.where(dists < min_point_dist)[0]
    remove_inds = set()
    pair_set = set()
    
    for ind in tqdm(close_inds):
        i,j = ind_pairs[ind]

        if ((i,j) not in pair_set):
            remove_inds.add(i)
            pair_set.add((i,j))

    remove_inds = list(remove_inds)

    #   for ind in tqdm(close_inds):
    #       # check if index is in index pair list
    #       same_inds = np.where(ind_pairs == ind)[0]
    #       if (len(same_inds) > 0):
    #           # add index to unique removal index list
    #           unique_inds.append(ind)

    #           for offset,i in enumerate(same_inds):
    #               index = i - offset
    #               ind_pairs = np.delete(ind_pairs, index, axis=0)


    cloud = ply.select_by_index(remove_inds, invert=True)
    return cloud

def build_est_points_filter(est_ply, data_path, scan_num):
    # read in matlab bounding box, mask, and resolution
    mask_filename = "ObsMask{}_10.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    bounds = np.asarray(data["BB"])
    min_bound = bounds[0,:]
    max_bound = bounds[1,:]
    mask = np.asarray(data["ObsMask"])
    res = int(data["Res"])

    points = np.asarray(est_ply.points).transpose()
    shape = points.shape
    mask_shape = mask.shape
    filt = np.zeros(shape[1])

    min_bound = min_bound.reshape(3,1)
    min_bound = np.tile(min_bound, (1,shape[1]))

    qv = points
    qv = (points - min_bound) / res
    qv = round_nearest(qv).astype(int)

    # get all valid points
    in_bounds = np.asarray(np.where( ((qv[0,:]>=0) & (qv[0,:] < mask_shape[0]) & (qv[1,:]>=0) & (qv[1,:] < mask_shape[1]) & (qv[2,:]>=0) & (qv[2,:] < mask_shape[2])))).squeeze(0)
    valid_points = qv[:,in_bounds]

    # convert 3D coords ([x,y,z]) to appropriate flattened coordinate ((x*mask_shape[1]*mask_shape[2]) + (y*mask_shape[2]) + z )
    mask_inds = np.ravel_multi_index(valid_points, dims=mask.shape, order='C')

    # further trim down valid points by mask value (keep point if mask is True)
    mask = mask.flatten()
    valid_mask_points = np.asarray(np.where(mask[mask_inds] == True)).squeeze(0)

    # add 1 to indices where we want to keep points
    filt[in_bounds[valid_mask_points]] = 1

    return filt

def build_gt_points_filter(ply, data_path, scan_num):
    # read in matlab gt plane 
    mask_filename = "Plane{}.mat".format(scan_num)
    mask_path = os.path.join(data_path, "ObsMask", mask_filename)
    data = sio.loadmat(mask_path)
    P = np.asarray(data["P"])

    points = np.asarray(ply.points).transpose()
    shape = points.shape

    # compute iner-product between points and the defined plane
    Pt = P.transpose()

    points = np.concatenate((points, np.ones((1,shape[1]))), axis=0)
    plane_prod = (Pt @ points).squeeze(0)

    # get all valid points
    filt = np.asarray(np.where((plane_prod > 0), 1, 0))

    return filt

def filter_outlier_points(est_ply, gt_ply, outlier_th):
    dists_est = np.asarray(est_ply.compute_point_cloud_distance(gt_ply))
    valid_dists = np.where(dists_est <= outlier_th)[0]
    return est_ply.select_by_index(valid_dists)

def compare_point_clouds(est_ply, gt_ply, mask_th, max_dist, min_dist, est_filt=None, gt_filt=None):
    mask_gt = 20.0
    inlier_th = 0.5

    # compute bi-directional chamfer distance between point clouds
    dists_est = np.asarray(est_ply.compute_point_cloud_distance(gt_ply))
    valid_inds_est = set(np.where(est_filt == 1)[0])
    valid_dists = set(np.where(dists_est <= mask_th)[0])
    valid_inds_est.intersection_update(valid_dists)
    inlier_inds_est = set(np.where(dists_est < inlier_th)[0])
    inlier_inds_est.intersection_update(valid_inds_est)
    outlier_inds_est = set(np.where(dists_est >= inlier_th)[0])
    outlier_inds_est.intersection_update(valid_inds_est)
    valid_inds_est = np.asarray(list(valid_inds_est))
    inlier_inds_est = np.asarray(list(inlier_inds_est))
    outlier_inds_est = np.asarray(list(outlier_inds_est))
    dists_est = dists_est[valid_inds_est]

    dists_gt = np.asarray(gt_ply.compute_point_cloud_distance(est_ply))
    valid_inds_gt = set(np.where(gt_filt == 1)[0])
    valid_dists = set(np.where(dists_gt <= mask_gt)[0])
    valid_inds_gt.intersection_update(valid_dists)
    inlier_inds_gt = set(np.where(dists_gt < inlier_th)[0])
    inlier_inds_gt.intersection_update(valid_inds_gt)
    outlier_inds_gt = set(np.where(dists_gt >= inlier_th)[0])
    outlier_inds_gt.intersection_update(valid_inds_gt)
    valid_inds_gt = np.asarray(list(valid_inds_gt))
    inlier_inds_gt = np.asarray(list(inlier_inds_gt))
    outlier_inds_gt = np.asarray(list(outlier_inds_gt))
    dists_gt = dists_gt[valid_inds_gt]

    # compute accuracy and competeness
    acc = np.mean(dists_est)
    comp = np.mean(dists_gt)

    # measure incremental precision and recall values with thesholds from (0, 10*max_dist)
    th_vals = np.linspace(0, 3*max_dist, num=50)
    prec_vals = [ (len(np.where(dists_est <= th)[0]) / len(dists_est)) for th in th_vals ]
    rec_vals = [ (len(np.where(dists_gt <= th)[0]) / len(dists_gt)) for th in th_vals ]

    # compute precision and recall for given distance threshold
    prec = len(np.where(dists_est <= max_dist)[0]) / len(dists_est)
    rec = len(np.where(dists_gt <= max_dist)[0]) / len(dists_gt)

    # color point cloud for precision
    valid_est_ply = est_ply.select_by_index(valid_inds_est)
    est_size = len(valid_est_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_est, max_dist) / max_dist)[:, :3]
    valid_est_ply.colors = o3d.utility.Vector3dVector(colors)

    # color invalid points precision
    invalid_est_ply = est_ply.select_by_index(valid_inds_est, invert=True)
    cmap = plt.get_cmap("winter")
    colors = cmap(np.ones(len(invalid_est_ply.points)))[:, :3]
    invalid_est_ply.colors = o3d.utility.Vector3dVector(colors)

    # color point cloud for recall
    valid_gt_ply = gt_ply.select_by_index(valid_inds_gt)
    gt_size = len(valid_gt_ply.points)
    cmap = plt.get_cmap("hot_r")
    colors = cmap(np.minimum(dists_gt, max_dist) / max_dist)[:, :3]
    valid_gt_ply.colors = o3d.utility.Vector3dVector(colors)

    # color invalid points recall
    invalid_gt_ply = gt_ply.select_by_index(valid_inds_gt, invert=True)
    cmap = plt.get_cmap("winter")
    colors = cmap(np.ones(len(invalid_gt_ply.points)))[:, :3]
    invalid_gt_ply.colors = o3d.utility.Vector3dVector(colors)

    # color accuracy outliers
    inlier_est_ply = est_ply.select_by_index(inlier_inds_est)
    outlier_est_ply = est_ply.select_by_index(outlier_inds_est)
    inlier_cmap = plt.get_cmap("binary")
    outlier_cmap = plt.get_cmap("cool")
    inlier_colors = inlier_cmap(np.ones(len(inlier_est_ply.points)))[:, :3]
    outlier_colors = outlier_cmap(np.ones(len(outlier_est_ply.points)))[:, :3]
    inlier_est_ply.colors = o3d.utility.Vector3dVector(inlier_colors)
    outlier_est_ply.colors = o3d.utility.Vector3dVector(outlier_colors)

    # color completeness outliers
    inlier_gt_ply = gt_ply.select_by_index(inlier_inds_gt)
    outlier_gt_ply = gt_ply.select_by_index(outlier_inds_gt)
    inlier_cmap = plt.get_cmap("binary")
    outlier_cmap = plt.get_cmap("cool")
    inlier_colors = inlier_cmap(np.ones(len(inlier_gt_ply.points)))[:, :3]
    outlier_colors = outlier_cmap(np.ones(len(outlier_gt_ply.points)))[:, :3]
    inlier_gt_ply.colors = o3d.utility.Vector3dVector(inlier_colors)
    outlier_gt_ply.colors = o3d.utility.Vector3dVector(outlier_colors)

    precision_ply = valid_est_ply + invalid_est_ply
    recall_ply = valid_gt_ply + invalid_gt_ply
    acc_outliers_ply = inlier_est_ply+outlier_est_ply+invalid_est_ply
    comp_outliers_ply = inlier_gt_ply+outlier_gt_ply+invalid_gt_ply

    return  (precision_ply, recall_ply) \
            ,(acc_outliers_ply, comp_outliers_ply) \
            ,(acc,comp), (prec, rec) \
            ,(th_vals, prec_vals, rec_vals) \
            ,(est_size, gt_size)

def dtu_point_eval(cfg, scan, method="vfuse"):
    eval_data_path = cfg["eval"]["data_path"]
    output_path = cfg["output_path"]
    mask_th = cfg["eval"]["mask_th"]
    max_dist = cfg["eval"]["max_dist"]
    min_dist = cfg["eval"]["min_dist"]
    min_point_dist = cfg["eval"]["min_point_dist"]
    resolution = cfg["eval"]["resolution"]
    scan_num = int(scan[-3:])

    start_total = time()
    print("\nEvaluating scan{:03d}...".format(scan_num))

    # read in point clouds
    est_ply_filename = "{}.ply".format(scan)
    points_path = os.path.join(output_path, "scan{}".format(str(scan_num).zfill(3)), "point_clouds")
    est_ply_path = os.path.join(points_path, est_ply_filename)
    est_ply = read_point_cloud(est_ply_path)
    est_ply = downsample_cloud(est_ply, min_point_dist)
    #est_ply = remove_close_points(est_ply, min_point_dist)

    gt_ply_filename = "stl{:03d}_total.ply".format(scan_num)
    gt_ply_path = os.path.join(eval_data_path, "Points", f"stl_{resolution}", gt_ply_filename)
    gt_ply = read_point_cloud(gt_ply_path)

    # build points filter based on input mask
    est_ply = filter_outlier_points(est_ply, gt_ply, mask_th)
    est_filt = build_est_points_filter(est_ply, eval_data_path, scan_num)
    gt_filt = build_gt_points_filter(gt_ply, eval_data_path, scan_num)

    # compute distances between point clouds
    (precision_ply, recall_ply) \
    ,(acc_outliers_ply, comp_outliers_ply) \
    ,(acc,comp) \
    ,(prec, rec) \
    ,(th_vals, prec_vals, rec_vals) \
    ,(est_size, gt_size) = \
    compare_point_clouds(est_ply, gt_ply, mask_th, max_dist, min_dist, est_filt, gt_filt)

    end_total = time()
    dur = end_total-start_total

    # display current evaluation
    print("Num Est: {}".format(int(est_size)))
    print("Num GT: {}".format(int(gt_size)))
    print("Accuracy: {:0.4f}".format(acc))
    print("Completeness: {:0.4f}".format(comp))
    print("Overall: {:0.4f}".format((acc+comp)/2.0))
    print("Precision: {:0.4f}".format(prec))
    print("Recall: {:0.4f}".format(rec))
    print("Elapsed time: {:0.3f} s".format(dur))

    ##### Save metrics #####
    eval_path = os.path.join(points_path, "eval")
    if (os.path.exists(eval_path)):
        shutil.rmtree(eval_path)
    os.mkdir(eval_path)

    # save precision point cloud
    precision_path = os.path.join(eval_path, "precision.ply")
    write_point_cloud(precision_path, precision_ply)

    # save recall point cloud
    recall_path = os.path.join(eval_path, "recall.ply")
    write_point_cloud(recall_path, recall_ply)

    # save binary accuracy point cloud
    acc_outliers_path = os.path.join(eval_path, "acc_outliers.ply")
    write_point_cloud(acc_outliers_path, acc_outliers_ply)

    # save recall point cloud
    comp_outliers_path = os.path.join(eval_path, "comp_outliers.ply")
    write_point_cloud(comp_outliers_path, comp_outliers_ply)

    # create plots for incremental threshold values
    plot_filename = os.path.join(eval_path, "eval.png")
    plt.plot(th_vals, prec_vals, th_vals, rec_vals)
    plt.title("Precision and Recall (t={}mm)".format(max_dist))
    plt.xlabel("threshold")
    plt.vlines(max_dist, 0, 1, linestyles='dashed', label='t')
    plt.legend(("precision", "recall"))
    plt.grid()
    plt.savefig(plot_filename)
    plt.close()

    # write all metrics to the evaluation file
    stats_file = os.path.join(eval_path, "metrics.txt")
    with open(stats_file, 'w') as f:
        f.write("Method: {}\n".format(method))
        f.write("Min_point_dist: {:0.3f}mm | Max distance threshold: {:0.3f}mm | Min distance threshold: {:0.3f}mm | Mask threshold: {:0.3f}mm\n".format(min_point_dist, max_dist, min_dist, mask_th))
        f.write("Source point cloud size: {}\n".format(est_size))
        f.write("Target point cloud size: {}\n".format(gt_size))
        f.write("Accuracy: {:0.3f}mm\n".format(acc))
        f.write("Completness: {:0.3f}mm\n".format(comp))
        f.write("Overall: {:0.4f}\n".format((acc+comp)/2.0))
        f.write("Precision: {:0.3f}\n".format(prec))
        f.write("Recall: {:0.3f}\n".format(rec))

    return acc, comp, (acc+comp)/2.0, prec, rec
