import cv2
import numpy as np


def rotate(ps, m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x], target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point

"""
rotate_img_and_point函数：

  功能 ：对输入的图片和坐标点进行相同的仿射变化，返回变化之后的图和坐标点

参数：

   img ：输入图片 numpy数组

points：需要计算的坐标点，[[x1,y1],[x2,y2]......]

angle:旋转的度数 注意是角度，不是弧度

center_x:旋转的中心点的x坐标

center_y：旋转的中心点的y坐标

resize_rate:缩放的比例，默认为1
"""
def rotate_img_and_point(img, points, angle, center_x_center_y, resize_rate=1.0):
    h, w, c = img.shape
    M = cv2.getRotationMatrix2D(center_x_center_y, angle, resize_rate)
    res_img = cv2.warpAffine(img, M, (w, h))
    out_points = rotate(points, M)
    return res_img, out_points
