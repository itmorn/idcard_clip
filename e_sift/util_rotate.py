import time
import numpy as np
import cv2


def rotate(ps, m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x], target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), M


def calc_rotate_angle(best_dst):
    p1 = best_dst[0] # 身份证左上角(x,y)
    p2 = best_dst[-1] # 身份证右上角(x,y)
    dx,  dy= p2 - p1
    theta = dx / ((dx ** 2 + dy ** 2) ** 0.5)
    theta2 = np.arccos(theta) / np.pi * 180
    # 如果角度差不多等于0度或180度，就强制转为该度数，防止像素偏差
    if theta2 < 5:
        theta2 = 0
    elif 85 < theta2 < 95:
        theta2 = 90
    elif theta2 > 175:
        theta2 = 180

    if dy > 0:
        # 需要顺时针旋转
        theta2 = 360 - theta2
    return theta2
