import numpy as np
import cv2
import scipy.ndimage as ndimage
import skimage.transform as transform
import sys

f_matrix = None
img2_line = None
img1 = None
img2 = None

CENTERED = False


def mouse1_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        #print("mouse1 X: " + str(x) + " Y: " + str(y))
#        y = img1.shape[0] - y
        if CENTERED:
            x -= img1.shape[1] / 2
            y -= img1.shape[0] / 2
        mouse1_pt = np.asarray([x, y, 1.0])
        i2ray = np.dot(f_matrix, mouse1_pt)
        if CENTERED:
            i2ray[2] = i2ray[2] - i2ray[0]*img2.shape[1]/2 - i2ray[1]*img2.shape[0]/2 
        #print(i2ray)
        i2pt1_x = 0
        i2pt2_x = img2.shape[1]
        i2pt1_y = int(-(i2ray[2] + i2ray[0] * i2pt1_x) / i2ray[1])
        i2pt2_y = int(-(i2ray[2] + i2ray[0] * i2pt2_x) / i2ray[1])

        global img2_line
#        img2_line = ((i2pt1_x, img1.shape[0] - i2pt1_y), (i2pt2_x, img1.shape[0] - i2pt2_y))
        img2_line = ((i2pt1_x, i2pt1_y), (i2pt2_x, i2pt2_y))

def draw_line(img, line):
    if line is None:
        return img
    # Clip the line to image bounds
    ret, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), line[0], line[1])
    if ret:
        img = img.copy()
        cv2.line(img, p1, p2, (255, 255, 0), 1)
    return img


def scale_img(img, scale):
    # This function takes an integer image in range of 0-255
    # and returns a float64 image in range of 0-1
    img_scaled = transform.resize(img,
                    [img.shape[0] * scale, img.shape[1] * scale], 
                    mode="constant")
    # Cast it to float32 to appease OpenCV later on
    return img_scaled.astype(np.float32)


def scale_f_mat(mat, scale):
    mat[:, 2] *= scale
    mat[2, :] *= scale
    return mat


def fmat_demo(img1l, img2l, fl, scale=1.0):
    global img1, img2, f_matrix

    img1 = scale_img(img1l, scale)
    img2 = scale_img(img2l, scale)
    f_matrix = scale_f_mat(fl, scale)

    f_matrix = f_matrix/np.max(f_matrix)

    print("Img1: " + str(img1.shape))
    print("Img2: " + str(img2.shape))
    #print(f)

    cv2.namedWindow("img1")
    cv2.namedWindow("img2")
    cv2.setMouseCallback("img1", mouse1_callback)

    while True:
        show2 = draw_line(img2, img2_line)
        show1 = img1
        cv2.imshow("img1", cv2.cvtColor(show1, cv2.COLOR_BGR2RGB))
        cv2.imshow("img2", cv2.cvtColor(show2, cv2.COLOR_BGR2RGB))
        cv2.waitKey(50)
        

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: pyfmatrix_viewer.py img1_path img2_path F_path [scale]")
        print("Examples:")
        print("   pyfmatrix_viewer.py data/264.bmp data/435.bmp data/f-264-435.txt")
        print("   pyfmatrix_viewer.py data/264.bmp data/435.bmp data/f-264-435.txt 0.25")
        quit(-1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    f_path = sys.argv[3]
    scale = 0.25
    if len(sys.argv) > 4:
        scale = float(sys.argv[4])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    f = np.loadtxt(f_path)

    fmat_demo(img1, img2, f, scale)
