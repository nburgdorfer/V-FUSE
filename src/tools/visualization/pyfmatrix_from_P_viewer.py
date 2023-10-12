# Philippos 4/17/2018
# based on Noah's code
# assumption: all relevant datasets, store intrinsics and extrinsics separately
# assumption 2: K is constant throughout

import numpy as np
import cv2
import skimage.transform as transform
import sys
from pyfmatrix_viewer import *


def fundamentalFromKP(K,P1,P2) :
    R1 = P1[0:3,0:3]
    t1 = P1[0:3,3]
    R2 = P2[0:3,0:3]
    t2 = P2[0:3,3]

    t1aug = np.array([t1[0], t1[1], t1[2], 1])
    epi2 = np.matmul(P2,t1aug)
    epi2 = np.matmul(K,epi2[0:3])
    print('epipole 2: {} {}'.format(epi2[0]/epi2[2],epi2[1]/epi2[2]))


    R = np.matmul(R2,np.transpose(R1))
    t= t2- np.matmul(R,t1)
    #print(R)
    #print(t)
    K1inv = np.linalg.inv(K)
    K2invT = np.transpose(K1inv)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.matmul(K2invT,np.matmul(tx,np.matmul(R,K1inv)))
    F = F/np.amax(F)
    return F

# pyfmatrix_from_P_viewer.py ~/data/t_n_t/Barn_all_frames/Barn4/004839.jpg ~/data/t_n_t/Barn_all_frames/Barn4/004840.jpg ~/data/t_n_t/Barn/subseqs/K_tnt.txt ~/data/t_n_t/Barn/subseqs/004839.txt ~/data/t_n_t/Barn/subseqs/004840.txt

# NOTE P matrix/frame numbering inconsistency + inversion for COLMAP frames
# pyfmatrix_from_P_viewer.py ~/data/t_n_t/Barn_img/000304.jpg ~/data/t_n_t/Barn_img/000305.jpg ~/data/t_n_t/Barn/subseqs/K_tnt.txt ~/data/t_n_t/Barn/subseqs/sparse304.txt ~/data/t_n_t/Barn/subseqs/sparse305.txt

# now mix sequences
# pyfmatrix_from_P_viewer.py ~/data/t_n_t/Barn_img/000087.jpg ~/data/t_n_t/Barn_all_frames/Barn4/002561.jpg ~/data/t_n_t/Barn/subseqs/K_tnt.txt ~/data/t_n_t/Barn/subseqs/sparse087.txt ~/data/t_n_t/Barn/subseqs/dense2561.txt

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: pyfmatrix_from_P_viewer.py img1_path img2_path K_path P1_path P2_path [scale]")
        print("Examples:")
        print("   pyfmatrix_from_P_viewer.py data/264.bmp data/435.bmp data/K_tnt.txt data/P-264.txt data/P-435.txt")
        print("   pyfmatrix_from_P_viewer.py data/264.bmp data/435.bmp data/K_tnt.txt data/P-264.txt data/P-435.txt 0.25")
        quit(-1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    K_path = sys.argv[3]
    P1_path = sys.argv[4]
    P2_path = sys.argv[5]
    scale = 0.5
    if len(sys.argv) > 6:
        scale = float(sys.argv[6])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    K =  np.loadtxt(K_path)
    K = np.reshape(K,(3,3))
    P1 = np.loadtxt(P1_path)
    P1 = np.reshape(P1,(3,4)) # 4x4 can be ignored later
    print(P1)
    P2 = np.loadtxt(P2_path)
    P2 = np.reshape(P2,(3,4))

    # for COLMAP log, invert
    #P1 = np.linalg.inv(P1)
    #print(P1)
    #P2 = np.linalg.inv(P2)
    #print(P2)

    f = fundamentalFromKP(K,P1,P2)

    fmat_demo(img1, img2, f, scale)
