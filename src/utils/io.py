import os
import sys
import numpy as np
import cv2
import re
import math
import open3d as o3d
from typing import List, Tuple

import torch

from scipy.spatial.transform import Rotation as rot

from src.utils.camera import y_axis_rotation

def load_cluster_list(cluster_file):
    with open(cluster_file, 'r') as cf:
        lines = cf.readlines()
    num_views = int(lines[0])
    cluster_list=np.zeros((num_views,22))
    i = 1
    j = 0
    while i < len(lines)-1:
        cluster_list[j,0] = np.asarray(lines[i].strip().split())
        lst = np.asarray(lines[i+1].strip().split())
        cluster_list[j,1:len(lst)+1] = lst
        i+=2
        j+=1

    return cluster_list


def save_model(model, cfg, name="ckpt_model.pth"):
    """Saves model weights to disk.
    """
    ckpt_path = cfg["model"]["ckpt"]
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Saving model checkpoint to {ckpt_path}...")
    model_path = os.path.join(save_folder, name)
    torch.save(model.state_dict(), model_path)

def load_pretrained_model(model, ckpt):
    """Loads model weights from disk.
    """
    print(f"Loading model from: {ckpt}...")
    try:
        model.load_state_dict(torch.load(ckpt))
    except Exception as e:
        print(e)
        print("Failed loading network weights...")
        sys.exit()

def read_cams_sfm(camera_path: str, extension: str = "cam.txt") -> np.ndarray:
    """Reads an entire directory of camera files in SFM format.

    Parameters:
        camera_path: Path to the directory of camera files.
        extension: File extension being used for the camera files.

    Returns:
        Array of camera extrinsics, intrinsics, and view metadata (Nx2x4x4).
    """
    cam_files = os.listdir(camera_path)
    cam_files.sort()

    cams = []
    
    for cf in cam_files:
        if (cf[-7:] != extension):
            continue

        cam_path = os.path.join(camera_path,cf)
        with open(cam_path,'r') as f:
            cam = read_single_cam_sfm(f, 256)
            cams.append(cam)

    return np.asarray(cams)

def read_cams_trajectory(log_file: str) -> np.ndarray:
    """Reads camera file in Trajectory File format.

    Parameters:
        log_file: Input *.log file to be read.

    Returns:
        Array of camera extrinsics, intrinsics, and view metadata (Nx2x4x4).
    """
    cams = []
    
    with open(log_file,'r') as f:
        lines = f.readlines()

        for i in range(0,len(lines),5):
            cam = np.zeros((4, 4))
            # read extrinsic
            for j in range(1, 5):
                cam[j-1,:] = np.array([float(l.strip()) for l in lines[i+j].split()])
            cam = np.linalg.inv(cam)
                
            cams.append(cam)

    return cams


def read_cams_traj(trajectory_file: str, frames: int = -1) -> np.ndarray:
    """Reads extrinsic camera pose from the traj.txt file for the Replica dataset.
    """
    poses = []
    with open(trajectory_file, "r") as f:
        lines = f.readlines()
    if (frames == -1):
        n = len(lines)
    else:
        n = frames
    for i in range(n):
        line = lines[i]
        P = np.array(list(map(float, line.split()))).reshape(4, 4)
        P[:3, 1] *= -1
        P[:3, 2] *= -1
        poses.append(P)
    return np.asarray(poses)

def read_exr(filename):
    """Read depth data from EXR image file.
    Parameters:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

def read_extrinsics_tum(tum_file: str, key_frames: List[int] = None) -> np.ndarray:
    """Reads extrinsic camera trajectories in TUM format [timestamp tx ty tz qx qy qz qw].

    Parameters:
        tum_file: Input extrinsics file.
        key_frames: Indices corresponding to the desired keyframes.

    Returns:
        Array of camera extrinsics (Nx4x4).
    """
    rot_interval = 30
    max_rot_angle = math.pi / 3

    extrinsics = []
    with open(tum_file,"r") as tf:
        lines = tf.readlines()
        
        for i,line in enumerate(lines):
            l = np.asarray(line.strip().split(" "), dtype=float)
            l = l[1:]
            t = l[:3]
            q = l[3:]

            R = rot.from_quat(q).as_matrix()
            R = R.transpose()
            t = -R@t
            P = np.zeros((4,4))
            P[:3,:3] = R
            P[:3,3] = t.transpose()
            P[3,3] = 1

            extrinsics.append(P)

            if((key_frames == None) or (i in key_frames)):
                left = np.linspace(0.0, max_rot_angle, rot_interval)
                right = np.linspace(max_rot_angle, -(max_rot_angle), rot_interval*2)
                center = np.linspace(-(max_rot_angle), 0.0, rot_interval)
                thetas = np.concatenate((left,right,center))

                for theta in thetas:
                    new_P = y_axis_rotation(P,theta)
                    extrinsics.append(new_P)

    return np.asarray(extrinsics)

def read_matrix(mat_file: str) -> np.ndarray:
    """Reads a single matrix of float values from a file.

    Parameters:
        mat_file: Input file for the matrix to be read.

    Returns:
        The matrix stored in the given file.
    """
    with open(mat_file, 'r') as f:
        lines = f.readlines()
        M = []

        for l in lines:
            row = l.split()
            row = [float(s) for s in row]
            M.append(row)
        M = np.array(M)

    return M

def read_mesh(mesh_file: str) -> o3d.geometry.TriangleMesh:
    """Reads a mesh from a file.

    Parameters:
        mesh_file: Input mesh file.

    Returns:
        The mesh stored in the given file.
    """
    return o3d.io.read_triangle_mesh(mesh_file)

def read_cluster_list(filename: str) -> List[Tuple[int,List[int]]]:
    """Reads a cluster list file encoding supporting camera viewpoints.

    Parameters:
        filename: Input file encoding per-camera viewpoints.

    Returns:
        An array of tuples encoding (ref_view, [src_1,src_2,..])
    """
    data = []
    with open(filename) as f:
        num_views = int(f.readline())
        all_views = list(range(0,num_views))

        for view_idx in range(num_views):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) == 0:
                continue
            data.append((ref_view, src_views))
    return data

def read_pfm(pfm_file: str) -> np.ndarray:
    """Reads a file in *.pfm format.

    Parameters:
        pfm_file: Input *.pfm file to be read.

    Returns:
        Data map that was stored in the *.pfm file.
    """
    with open(pfm_file, 'rb') as pfm_file:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = pfm_file.readline().decode('iso8859_15').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('iso8859_15'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((pfm_file.readline()).decode('iso8859_15').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = pfm_file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
    return data

def read_point_cloud(point_cloud_file: str) -> o3d.geometry.PointCloud:
    """Reads a point cloud from a file.

    Parameters:
        point_cloud_file: Input point cloud file.

    Returns:
        The point cloud stored in the given file.
    """
    return o3d.io.read_point_cloud(point_cloud_file)

def read_single_cam_sfm(cam_file: str, depth_planes: int = 256) -> np.ndarray:
    """Reads a single camera file in SFM format.

    Parameters:
        cam_file: Input camera file to be read.
        depth_planes: Number of depth planes to store in the view metadata.

    Returns:
        Camera extrinsics, intrinsics, and view metadata (2x4x4).
    """
    cam = np.zeros((2, 4, 4))

    with open(cam_file, 'r') as cam_file:
        words = cam_file.read().split()

    words_len = len(words)

    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0,i,j] = float(words[extrinsic_index])

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1,i,j] = float(words[intrinsic_index])

    if words_len == 29:
        cam[1,3,0] = float(words[27])
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = depth_planes
        cam[1,3,3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 30:
        cam[1,3,0] = float(words[27])
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = float(words[29])
        cam[1,3,3] = cam[1][3][0] + (cam[1][3][1] * cam[1][3][2])
    elif words_len == 31:
        cam[1,3,0] = words[27]
        cam[1,3,1] = float(words[28])
        cam[1,3,2] = float(words[29])
        cam[1,3,3] = float(words[30])
    else:
        cam[1,3,0] = 0
        cam[1,3,1] = 0
        cam[1,3,2] = 0
        cam[1,3,3] = 1

    return cam

def read_stereo_intrinsics_yaml(intrinsics_file: str) -> Tuple[ np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray, \
                                                                np.ndarray]:
    """Reads intrinsics information for a stereo camera pair from a *.yaml file.

    Parameters:
        intrinsics_file: Input *.yaml file storing the intrinsics information.

    Returns:
        K_left: Intrinsics matrix (3x3) of left camera.
        D_left: Distortion coefficients vector (1x4) of left camera.
        K_right: Intrinsics matrix (3x3) of right camera.
        D_right: Distortion coefficients vector (1x4) of right camera.
        R: Relative rotation matrix (3x3) from left -> right cameras.
        T: Relative translation vector (1x3) from left -> right cameras.
    """
    K_left = np.zeros((3,3))
    D_left = np.zeros((1,4))
    K_right = np.zeros((3,3))
    D_right = np.zeros((1,4))
    R = np.zeros((3,3))
    T = np.zeros((1,3))

    cv_file = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)

    left = cv_file.getNode("left")
    K_left = left.getNode("K").mat()
    D_left = left.getNode("D").mat()

    right = cv_file.getNode("right")
    K_right = right.getNode("K").mat()
    D_right = right.getNode("D").mat()

    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()

    cv_file.release()

    return [K_left, D_left, K_right, D_right, R, T]

def write_cam_sfm(cam_file: str, cam: np.ndarray) -> None:
    """Writes intrinsic and extrinsic camera parameters to a file in sfm format.

    Parameters:
        cam_file: The file to be writen to.
        cam: Camera extrinsic and intrinsic data to be written.
    """
    with open(cam_file, "w") as f:
        f.write('extrinsic\n')
        for i in range(0, 4):
            for j in range(0, 4):
                f.write(str(cam[0][i][j]) + ' ')
            f.write('\n')
        f.write('\n')

        f.write('intrinsic\n')
        for i in range(0, 3):
            for j in range(0, 3):
                f.write(str(cam[1][i][j]) + ' ')
            f.write('\n')

        f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

def write_matrix(M: np.ndarray, mat_file: str) -> None:
    """Writes a single matrix to a file.

    Parameters:
        M: Matrix to be stored.
        mat_file: Output file where the given matrix is to be writen.
    """
    with open(mat_file, "w") as f:
        for row in M:
            for e in row:
                f.write("{} ".format(e))
            f.write("\n")

def write_mesh(mesh_file: str, mesh: o3d.geometry.TriangleMesh) -> None:
    """Writes a mesh to a file.

    Parameters:
        mesh_file: Output mesh file.
        mesh: Mesh to be stored.
    """
    return o3d.io.write_triangle_mesh(mesh_file, mesh)

def write_pfm(pfm_file: str, data_map: np.ndarray, scale: float = 1.0) -> None:
    """Writes a data map to a file in *.pfm format.

    Parameters:
        pfm_file: Output *.pfm file to store the data map.
        data_map: Data map to be stored.
        scale: Value used to scale the data map.
    """
    with open(pfm_file, 'wb') as pfm_file:
        color = None

        if data_map.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        data_map = np.flipud(data_map)

        if len(data_map.shape) == 3 and data_map.shape[2] == 3: # color data_map
            color = True
        elif len(data_map.shape) == 2 or (len(data_map.shape) == 3 and data_map.shape[2] == 1): # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        a = 'PF\n' if color else 'Pf\n'
        b = '%d %d\n' % (data_map.shape[1], data_map.shape[0])
        
        pfm_file.write(a.encode('iso8859-15'))
        pfm_file.write(b.encode('iso8859-15'))

        endian = data_map.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        c = '%f\n' % scale
        pfm_file.write(c.encode('iso8859-15'))

        data_map_string = data_map.tostring()
        pfm_file.write(data_map_string)

def display_map(filename, disp_map, mx, mn):
    disp_map = ((disp_map-mn)/(mx-mn))*255
    cv2.imwrite(filename, disp_map)


def write_ply(fn, point, normal=None, color=None):

    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)
    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)
    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(fn, ply)

def write_point_cloud(fn, cloud):
    o3d.io.write_point_cloud(fn, cloud)
