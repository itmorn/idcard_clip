"""
@Auth: itmorn
@Date: 2022/6/19-14:39
@Email: 12567148@qq.com
"""
import copy
import os
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

from e_sift.util_rotate import rotate_img_and_point

sift = cv2.SIFT_create()
MIN_MATCH_COUNT = 5  # 设置最低特征点匹配数量为10

# 创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


class UtilSift:
    def __init__(self):
        # 初始化模板向量
        a = time.time()
        self.max_len = 1000  # 最长边转为该值
        self.lst_tmp_vec = self.get_lst_tmp_vec(dir_tmp="pic_tmps/")
        print("初始化模板：", time.time() - a)

    def get_lst_tmp_vec(self, dir_tmp):
        lst_tmp_vec = []
        lst_url_tmp = [dir_tmp + i for i in os.listdir(dir_tmp)]
        for url_tmp in lst_url_tmp:
            template = cv2.imread(url_tmp, 0)  # queryImage
            kp1, des1 = sift.detectAndCompute(template, None)
            lst_tmp_vec.append([kp1, des1, template])
        return lst_tmp_vec

    def calc_good_match(self, target, target_rgb, is_show=False):
        h, w = target.shape
        self.scale_factor = self.max_len / max(h, w)
        target = cv2.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        target_rgb = cv2.resize(target_rgb, (0, 0), fx=self.scale_factor, fy=self.scale_factor,
                                interpolation=cv2.INTER_LINEAR)
        # 计算待检测的图像的SIFT向量
        a = time.time()
        kp2, des2 = sift.detectAndCompute(target, None)
        print("初始化待检测的图像：", time.time() - a)
        # 与每一个模板进行匹配
        a = time.time()
        self.best_score = 0
        self.best_dst = None
        for idx, (kp1, des1, template) in enumerate(self.lst_tmp_vec):
            matches = flann.knnMatch(des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            # 舍弃大于0.7的匹配
            for m, n in matches:
                if m.distance < 1 * n.distance:
                    good.append(m)
            if len(good) >= MIN_MATCH_COUNT:
                # 获取关键点的坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                # 计算变换矩阵和MASK
                # 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()  # 先将mask变成一维，再将矩阵转化为列表
                h, w = template.shape
                # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                sum_score = np.sum(matchesMask)
                if is_show:
                    target2 = copy.deepcopy(target)
                    cv2.polylines(target2, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=None,
                                       matchesMask=matchesMask,
                                       flags=2)
                    result = cv2.drawMatches(template, kp1, target2, kp2, good, None, **draw_params)
                    cv2.imwrite(f"res{idx}_{sum_score}.png", result)
            else:
                print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
                continue
            if sum_score > self.best_score:
                self.best_score = sum_score
                self.best_dst = dst.reshape(4, 2)
        print("匹配耗时：", time.time() - a)
        return self.best_score, self.best_dst,target_rgb

    def rotate_and_crop(self, target_rgb):
        p1 = self.best_dst[0]
        p2 = self.best_dst[1]
        dy, dx = p2 - p1
        theta = dx / ((dx ** 2 + dy ** 2) ** 0.5)
        theta2 = np.arccos(theta) / np.pi * 180
        center = np.sum(self.best_dst, axis=0) / 4
        # 如果角度差不多等于0度或180度，就强制转为该度数，防止像素偏差
        if theta2<5:
            theta2 = 0
        elif 85<theta2<95:
            theta2 = 90
        elif theta2>175:
            theta2 = 180

        a = time.time()
        if dy > 0:
            # 需要顺时针旋转
            res_img, out_points = rotate_img_and_point(target_rgb, self.best_dst,360 -  theta2, tuple(center))
        else:
            # 逆时针旋转
            res_img, out_points = rotate_img_and_point(target_rgb, self.best_dst, theta2, tuple(center))

        print("旋转图像耗时：", time.time() - a)
        cv2.imwrite("2.png", res_img)

        pass


if __name__ == '__main__':
    utilSift = UtilSift()
    target_rgb = cv2.imread('../img_idcard/1254139.jpg', 1)  # trainImage
    target = cv2.imread('../img_idcard/1254139.jpg', 0)  # trainImage
    # 输入待检测图像，计算最佳匹配
    best_score, best_dst,target_rgb = utilSift.calc_good_match(target, target_rgb, is_show=True)

    # 如果没有足够多的匹配点，则停止
    if best_score < 50:
        9 / 0

    # 照片原图进行旋转，然后裁剪
    target_crop = utilSift.rotate_and_crop(target_rgb)
    print()
