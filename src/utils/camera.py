import os
import numpy as np
import math
import open3d as o3d
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation

import torch

def to_opengl_pose(pose):
    """
    OpenGL pose: (right-up-back) (cam-to-world)
    """
    pose = torch.linalg.inv(pose)
    pose[:3,1] *= -1
    pose[:3,2] *= -1
    return pose

def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3).expand(*batch_dims,3,3).to(data)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)


def build_o3d_traj(poses, K, width, height):
    trajectory = o3d.camera.PinholeCameraTrajectory()
    for pose in poses:
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
        camera_params.extrinsic = pose
        trajectory.parameters += [(camera_params)]
    return trajectory


def camera_center(cam: np.ndarray) -> np.ndarray:
    """Computes the center of a camera in world coordinates.

    Args:
        cam: The extrinsics matrix (4x4) of a given camera.

    Returns:
        The camera center vector (3x1) in world coordinates.
    """
    C = null_space(cam[:3,:4])
    C /= C[3,:]

    return C

def relative_transform(cams_1: np.ndarray, cams_2: np.ndarray) -> np.ndarray:
    """Computes the relative transformation between two sets of cameras.

    Args:
        cams_1: Array of the first set of cameras (Nx4x4).
        cams_2: Array of the second set of cameras (Nx4x4).

    Returns:
        The relative transformation matrix (4x4) between the two trajectories.
    """
    centers_1 = np.squeeze(np.array([ camera_center(c) for c in cams_1 ]), axis=2)
    centers_2 = np.squeeze(np.array([ camera_center(c) for c in cams_2 ]), axis=2)

    ### determine scale
    # grab first camera pair
    c1_0 = centers_1[0][:3]
    c2_0 = centers_2[0][:3]

    # grab one-hundreth camera pair
    c1_1 = centers_1[99][:3]
    c2_1 = centers_2[99][:3]

    # calculate the baseline between both sets of cameras
    baseline_1 = np.linalg.norm(c1_0 - c1_1)
    baseline_2 = np.linalg.norm(c2_0 - c2_1)

    # compute the scale based on the baseline ratio
    scale = baseline_2/baseline_1

    ### determine 1->2 Rotation 
    b1 = np.array([c[:3] for c in centers_1])
    b2 = np.array([c[:3] for c in centers_2])
    R = Rotation.align_vectors(b2,b1)[0].as_matrix()
    R = scale * R

    ### create transformation matrix
    M = np.eye(4)
    M[:3,:3] = R

    ### determine 1->2 Translation
    num_cams = len(cams_1)
    t = np.array([ c2-(M@c1) for c1,c2 in zip(centers_1,centers_2) ])
    t = np.mean(t, axis=0)

    ### add translation
    M[:3,3] = t[:3]

    return M


def sfm_to_trajectory(cams: np.ndarray, log_file: str) -> None:
    """Convert a set of cameras from SFM format to Trajectory File format.

    Args:
        cams: Array of camera extrinsics (Nx4x4) to be converted.
        log_file: Output path to the *.log file that is to be created.
    """
    num_cams = len(cams)

    with open(log_file, 'w') as f:
        for i,cam in enumerate(cams):
            # write camera to output_file
            f.write("{} {} 0\n".format(str(i),str(i)))
            for row in cam:
                for c in row:
                    f.write("{} ".format(str(c)))
                f.write("\n")
        
    return

def trajectory_to_sfm(log_file: str, camera_path: str, intrinsics: np.ndarray) -> None:
    """Convert a set of cameras from Trajectory File format to SFM format.

    Args:
        log_file: Input *.log file that stores the trajectory information.
        camera_path: Output path where the SFM camera files will be written.
        intrinsics: Array of intrinsics matrices (Nx3x3) for each camera.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        num_lines = len(lines)
        i = 0

        while(i < num_lines-5):
            view_num = int(lines[i].strip().split(" ")[0])
            
            cam = np.zeros((2,4,4))
            cam[0,0,:] = np.asarray(lines[i+1].strip().split(" "), dtype=float)
            cam[0,1,:] = np.asarray(lines[i+2].strip().split(" "), dtype=float)
            cam[0,2,:] = np.asarray(lines[i+3].strip().split(" "), dtype=float)
            cam[0,3,:] = np.asarray(lines[i+4].strip().split(" "), dtype=float)
            cam[0,:,:] = np.linalg.inv(cam[0,:,:])

            cam_file = "{:08d}_cam.txt".format(view_num)
            cam_path = os.path.join(camera_path, cam_file)

            cam[1,:,:] = intrinsics[view_num]

            write_cam(cam_path, cam)
            i = i+5
    return

def y_axis_rotation(P: np.ndarray, theta: float) -> np.ndarray:
    """Applies a rotation to the given camera extrinsics matrix along the y-axis.

    Parameters:
        P: Initial extrinsics camera matrix.
        theta: Angle (in radians) to rotate the camera.

    Returns:
        The rotated extrinsics matrix for the camera.
    """
    R = np.eye(4)
    R[0,0] = math.cos(theta)
    R[0,2] = math.sin(theta)
    R[2,0] = -(math.sin(theta))
    R[2,2] = math.cos(theta)

    P_rot = R @ P

    return P_rot
