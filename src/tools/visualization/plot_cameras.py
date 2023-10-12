import sys
import os
import numpy as np
import cv2
import argparse
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# custom imports
dirname = os.path.dirname(__file__)
rel_path = os.path.join(dirname, '../common_utilities')
sys.path.append(rel_path)
from utils import *

# argument parsing
parse = argparse.ArgumentParser(description="Camera Plotting Tool.")

parse.add_argument("-d", "--data_path", default="/data/dtu/cams", type=str, help="Path to the camera data.")
parse.add_argument("-n", "--num_cams", default=0, type=int, help="Number of cameras to plot.")
parse.add_argument("-f", "--format", default="mvsnet", type=str, help="The format type for the stored camera data (ex. mvsnet, colmap, ...).")
parse.add_argument("-o", "--output_file", default="data/cam.ply", type=str, help="The output point cloud file name/path.")
parse.add_argument("-a", "--alignment", default=None, type=str, help="The alignment file for the cameras.")
parse.add_argument("-s", "--scale", default=0.0002, type=float, help="Camera scale for camera pyramid construction.")

ARGS = parse.parse_args()

def plot_cameras(cams, num_cams, scale, A, output_file):
    # grab the requested number of cameras and apply the alignment
    P = cams[:num_cams]
    P = np.array([ A @ np.linalg.inv(p) for p in P ])

    # create 'num_cams' intrinsic matrices (just used for point-cloud camera pyramid geometry)
    k = np.array([[233.202, 0.0, 144.753],[0.0, 233.202, 108.323],[0.0, 0.0, 1.0]])
    K = np.array([ k for p in P])

    # build list of camera pyramid points
    pyr_pts = []
    for k,p in zip(K,P):
        pyr_pt = build_cam_pyr(scale, k)
        pyr_pt = p @ pyr_pt
        pyr_pts.append(pyr_pt)

    # build point cloud using camera centers
    build_pyr_point_cloud(pyr_pts, output_file)

    return

def main():
    # get the cameras and scale for the given format
    if (ARGS.format == "mvsnet"):
        cams = load_mvsnet_cams(ARGS.data_path)
        scale = ARGS.scale
    elif(ARGS.format == "colmap"):
        cams = load_colmap_cams(ARGS.data_path)
        scale = 5*ARGS.scale
    else:
        print("ERROR: unknown format type '{}'".format(form))
        sys.exit()

    # get number of cameras to plot
    if(ARGS.num_cams <= 0 or ARGS.num_cams > len(cams)):
        num_cams = len(cams)
    else:
        num_cams = ARGS.num_cams

    # read alignment file
    if(ARGS.alignment == None):
        A = np.eye(4)
    else:
        A = read_matrix(ARGS.alignment)

    # plot cameras into point-cloud
    plot_cameras(cams, num_cams, scale, A, ARGS.output_file)


if __name__=="__main__":
    main()
