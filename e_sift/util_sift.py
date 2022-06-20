"""
@Auth: itmorn
@Date: 2022/6/19-14:39
@Email: 12567148@qq.com
"""
import copy
import os

import numpy as np
import cv2

sift = cv2.SIFT_create()
MIN_MATCH_COUNT = 5  # 设置最低特征点匹配数量为10

# 创建设置FLANN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def get_lst_tmp_vec(dir_tmp):
    print(1)
    lst_tmp_vec = []
    lst_url_tmp = [dir_tmp + i for i in os.listdir(dir_tmp)]
    for url_tmp in lst_url_tmp:
        template = cv2.imread(url_tmp, 0)  # queryImage
        kp1, des1 = sift.detectAndCompute(template, None)
        lst_tmp_vec.append([kp1, des1, template])
    return lst_tmp_vec


class UtilSift:
    max_len = 1000  # 最长边转为该值
    dir_tmp = "pic_tmps/"
    lst_tmp_vec = get_lst_tmp_vec(dir_tmp)

    @classmethod
    def calc_good_match(cls, target, is_show=False):
        target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

        h, w = target.shape
        scale_factor = cls.max_len / max(h, w)
        target = cv2.resize(target, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # 计算待检测的图像的SIFT向量
        kp2, des2 = sift.detectAndCompute(target, None)
        # 与每一个模板进行匹配
        best_score = 0
        best_dst = None
        for idx, (kp1, des1, template) in enumerate(cls.lst_tmp_vec):
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
            if sum_score > best_score:
                best_score = sum_score
                best_dst = dst.reshape(4, 2)
        return best_score, best_dst / scale_factor

# class UtilSift:
#     def __init__(self):
#         # 初始化模板向量
#         self.max_len = 1000  # 最长边转为该值
#         self.dir_tmp = "pic_tmps/"
#         self.lst_tmp_vec = self.get_lst_tmp_vec(self.dir_tmp)
#
#     def get_lst_tmp_vec(self, dir_tmp):
#         lst_tmp_vec = []
#         lst_url_tmp = [dir_tmp + i for i in os.listdir(dir_tmp)]
#         for url_tmp in lst_url_tmp:
#             template = cv2.imread(url_tmp, 0)  # queryImage
#             kp1, des1 = sift.detectAndCompute(template, None)
#             lst_tmp_vec.append([kp1, des1, template])
#         return lst_tmp_vec
#
#     def calc_good_match(self, target, is_show=False):
#         target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
#
#         h, w = target.shape
#         self.scale_factor = self.max_len / max(h, w)
#         target = cv2.resize(target, (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
#
#         # 计算待检测的图像的SIFT向量
#         kp2, des2 = sift.detectAndCompute(target, None)
#         # 与每一个模板进行匹配
#         self.best_score = 0
#         self.best_dst = None
#         for idx, (kp1, des1, template) in enumerate(self.lst_tmp_vec):
#             matches = flann.knnMatch(des1, des2, k=2)
#             # store all the good matches as per Lowe's ratio test.
#             good = []
#             # 舍弃大于0.7的匹配
#             for m, n in matches:
#                 if m.distance < 1 * n.distance:
#                     good.append(m)
#             if len(good) >= MIN_MATCH_COUNT:
#                 # 获取关键点的坐标
#                 src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#                 dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#                 # 计算变换矩阵和MASK
#                 # 计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
#                 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#                 matchesMask = mask.ravel().tolist()  # 先将mask变成一维，再将矩阵转化为列表
#                 h, w = template.shape
#                 # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
#                 pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#                 dst = cv2.perspectiveTransform(pts, M)
#                 sum_score = np.sum(matchesMask)
#                 if is_show:
#                     target2 = copy.deepcopy(target)
#                     cv2.polylines(target2, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
#                     draw_params = dict(matchColor=(0, 255, 0),
#                                        singlePointColor=None,
#                                        matchesMask=matchesMask,
#                                        flags=2)
#                     result = cv2.drawMatches(template, kp1, target2, kp2, good, None, **draw_params)
#                     cv2.imwrite(f"res{idx}_{sum_score}.png", result)
#             else:
#                 print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
#                 continue
#             if sum_score > self.best_score:
#                 self.best_score = sum_score
#                 self.best_dst = dst.reshape(4, 2)
#         return self.best_score, self.best_dst / self.scale_factor
#
