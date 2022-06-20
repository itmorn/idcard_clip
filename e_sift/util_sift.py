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


def calc_score_rect(dst):
    p1 = dst[0]  # 左上 (x,y)
    p2 = dst[1]  # 左下
    p3 = dst[2]  # 右下
    p4 = dst[3]  # 右上


    len_left = np.sum((p1 - p2) ** 2) ** 0.5
    len_right = np.sum((p3 - p4) ** 2) ** 0.5

    len_up = np.sum((p1 - p4) ** 2) ** 0.5
    len_down = np.sum((p2 - p3) ** 2) ** 0.5

    if np.min([len_left,len_right,len_up,len_down])<50:
        return 1000

    score = np.abs(len_left - len_right) / len_right + np.abs(len_up - len_down) / len_down

    return score


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
        best_score_rect = np.inf
        best_dst = None
        for idx, (kp1, des1, template) in enumerate(cls.lst_tmp_vec):
            matches = flann.knnMatch(des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            # 舍弃大于0.7的匹配
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
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
                dst = cv2.perspectiveTransform(pts, M).reshape(4, 2)
                num_match_point = np.sum(matchesMask)
                if is_show:
                    target2 = copy.deepcopy(target)
                    cv2.polylines(target2, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=None,
                                       matchesMask=matchesMask,
                                       flags=2)
                    result = cv2.drawMatches(template, kp1, target2, kp2, good, None, **draw_params)
                    cv2.imwrite(f"res{idx}_{num_match_point}.png", result)
            else:
                print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
                continue

            # 如果没有足够多的匹配点，则停止
            if num_match_point < 10:
                print(num_match_point)
                continue

            # 矩形分数 越小越是矩形
            score_rect = calc_score_rect(dst)

            if score_rect < best_score_rect:
                best_score_rect = score_rect
                best_dst = dst

        if best_score_rect > 0.2: #如果矩形分数很高 则认为匹配出错 放弃
            print(best_score_rect)
            return

        if best_dst is None: # 没有一个模板匹配到  也是匹配出错 放弃
            return

        return best_dst / scale_factor

