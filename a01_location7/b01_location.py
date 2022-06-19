# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
import time
import os
from multiprocessing import Process, Manager, Lock
import os
import math
from time import sleep
import warnings
import random
from a01_location7 import cfg, utils_calc
from scipy import ndimage


# warnings.filterwarnings('error')


class StepA:
    def __init__(self, url_img, img_test_rgb):
        self.url_img = url_img
        self.img_name = os.path.basename(url_img)

        self.img_test_rgb = img_test_rgb
        self.h, self.w, _ = self.img_test_rgb.shape

        self.scale = cfg.factor_stepA_scale / min(self.h, self.w)

        self.img_test_rgb = cv2.resize(self.img_test_rgb, (int(self.w * self.scale), int(self.h * self.scale)))
        self.img_test = cv2.cvtColor(self.img_test_rgb, cv2.COLOR_BGR2GRAY)

        self.img_test_rgb0 = self.img_test_rgb.copy()
        self.img_test0 = self.img_test.copy()

        self.h, self.w = self.img_test.shape

        self.error_rect_best = np.inf  # 矩形程度误差
        self.error_var_angle_best = np.inf  # 角度的方差
        self.success_detect = False
        self.num_match_points = 0

        self.flag_find_border = False

    def run(self):
        self.kp_test, self.des_test = sift.detectAndCompute(self.img_test, None)

        for tmp_info in lst_tmp_info:
            url_imgtmp = tmp_info.get("url_imgtmp")
            img_tmp = tmp_info.get("img_tmp")
            kp_tmp = tmp_info.get("kp_tmp")
            des_tmp = tmp_info.get("des_tmp")

            self.match_tmp(url_imgtmp=url_imgtmp, img_tmp=img_tmp, des_tmp=des_tmp, kp_tmp=kp_tmp)

        if self.error_rect_best == cfg.score_out_of_pic:
            print("照片缺失")
            self.save_pic(cfg.dir_incomplete, self.img_match_best)
            return

        if self.num_match_points < cfg.num_match_points:
            print("照片模糊")
            self.save_pic(cfg.dir_vague, self.img_match_best)
            return

        if self.error_rect_best > cfg.th_error_rect or self.error_var_angle_best > cfg.th_var_error:
            print("照片畸变")
            self.save_pic(cfg.dir_distortion, self.img_match_best)
            return

        # 透视变换
        self.perspective_transform()

        # 移除左右边框
        self.remove_border()

        # 重新缩放图片
        self.img_test_res = cv2.resize(self.img_test_res, (cfg.w_output, cfg.h_output))

        self.save_pic(cfg.dir_output, self.img_test_res,need_border=True)



    def match_tmp(self, url_imgtmp, img_tmp, des_tmp, kp_tmp):
        matches = flann.knnMatch(des_tmp, self.des_test, k=2)
        good = []
        # 舍弃大于0.7的匹配
        for m, n in matches:
            if m.distance < 1.2 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            # 获取关键点的坐标
            pts_tmp = np.float32([kp_tmp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_test = np.float32([self.kp_test[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 计算变换矩阵和MASK
            M, mask = cv2.findHomography(pts_tmp, pts_test, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = img_tmp.shape
            # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img_test_rgb_draw = self.img_test_rgb.copy()
            cv2.polylines(img_test_rgb_draw, [np.int32(dst)], True, (255, 0, 0), 10, cv2.LINE_AA)
            # self.enough_matches_cur = True
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
            # self.enough_matches_cur = False
            return
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img_result = cv2.drawMatches(img_tmp, kp_tmp, img_test_rgb_draw, self.kp_test, good, None, **draw_params)
        self.img_result_cur = img_result

        # 框体为矩形的程度，
        arr_corner = dst.reshape([dst.shape[0], dst.shape[-1]])
        error_rect = self.calc_score_square(arr_corner)

        # 文字成比例程度
        kp_tmp2 = pts_tmp[np.array(matchesMask) == 1]
        kp_test2 = pts_test[np.array(matchesMask) == 1]

        kp_tmp2 = kp_tmp2.reshape([kp_tmp2.shape[0], kp_tmp2.shape[-1]])
        kp_test2 = kp_test2.reshape([kp_test2.shape[0], kp_test2.shape[-1]])

        var_error, angle_mean_rot = self.calc_char_prop_var_error(kp_tmp2, kp_test2)

        if error_rect < self.error_rect_best:
            self.num_match_points = len(kp_tmp2)
            self.url_imgtmp = url_imgtmp
            self.error_var_angle_best = var_error
            self.error_rect_best = error_rect
            self.img_match_best = img_result
            self.arr_corner_best = arr_corner
            self.angle_mean_rot_best = 360 - angle_mean_rot
            self.kp_tmp_good = kp_tmp2
            self.kp_test_good = kp_test2

    def calc_score_square(self, points):
        # 框体为矩形的程度
        error_rect = 0
        # 如果识别出的框体的四个顶点在照片之外较远，直接返回0分
        for x, y in points:
            if x < -cfg.extend_pix or x > self.w + cfg.extend_pix or \
                    y < -cfg.extend_pix or y > self.h + cfg.extend_pix:
                return cfg.score_out_of_pic
        ## 四个角的sin值的累乘
        for i in range(len(points)):
            p1 = points[i].copy()
            p2 = points[(i + 1) % len(points)].copy()
            p3 = points[(i + 2) % len(points)].copy()

            p1[-1] = -p1[-1]
            p2[-1] = -p2[-1]
            p3[-1] = -p3[-1]

            v1 = np.float64(p2 - p1)
            v2 = np.float64(p2 - p3)

            len_v1 = np.sum(v1 ** 2) ** 0.5
            len_v2 = np.sum(v2 ** 2) ** 0.5
            # 如果边的长度过小，就认为不是正确的框，直接0分
            if len_v1 < 200 or len_v2 < 200:
                error_rect = 999
                break

            cosx = np.dot(v1, v2) / (len_v1 * len_v2)
            error_rect += abs(np.rad2deg(np.arccos(np.clip(cosx, -1, 1))) - 90)

        return error_rect

    def _get_lst_two_point_idx(self, arr2d_kp_tmp):
        # 计算模板图像上两个点的索引
        lst_two_point_idx = []
        random.seed(666)
        for i in range(cfg.num_choice_point):
            res = random.choices(range(0, len(arr2d_kp_tmp)), k=2)
            if res[0] != res[1]:
                lst_two_point_idx.append(res)
        return lst_two_point_idx

    def _get_var_error(self, arr2d_kp_tmp, arr2d_kp_test, lst_two_point_idx):
        # 计算任意两个点形成的向量的夹角是否变化很小,可能偶尔有几个匹配错误的点，可以过滤掉
        lst_angle = []
        for two_point_idx in lst_two_point_idx:
            idx_p1, idx_p2 = two_point_idx
            p10 = arr2d_kp_tmp[idx_p1]
            p20 = arr2d_kp_tmp[idx_p2]
            v1 = p10 - p20
            v1[-1] = -v1[-1]

            p1 = arr2d_kp_test[idx_p1]
            p2 = arr2d_kp_test[idx_p2]
            v2 = p1 - p2
            v2[-1] = -v2[-1]

            len_v1 = np.sum(v1 ** 2) ** 0.5
            len_v2 = np.sum(v2 ** 2) ** 0.5
            if len_v1 < 10 or len_v2 < 10:  # 两个点不能太近，否则算出来的角度误差太大
                continue

            angle = utils_calc.GetClockAngle(v2, v1)

            lst_angle.append(angle)

        # #可能偶尔有几个匹配错误的点（太近），可以过滤掉
        arr_angle = np.array(lst_angle)

        var_error, angle_mean_rot = self._calc_error_and_rot(arr_angle)

        # 如果证件照是正的，可能存在1°和364°的情况，这需要把较小的度数+360，然后算方差。
        # 与之前的方差比较，如果方差更小；就算出角度和误差
        # 如果（355，360）存在数据，【0，5）存在数据
        var_error2 = np.inf
        if sum((355 < arr_angle) & (arr_angle < 360)) > 0 and sum((0 <= arr_angle) & (arr_angle < 5)) > 0:
            arr_angle2 = arr_angle.copy()
            arr_angle2[(0 <= arr_angle2) & (arr_angle2 < 180)] += 360
            var_error2, angle_mean_rot2 = self._calc_error_and_rot(arr_angle2)
            if angle_mean_rot2 >= 360:  # 如果旋转度数为365度，其实就是转5度
                angle_mean_rot2 -= 360
        if var_error < var_error2:
            return var_error, angle_mean_rot
        else:
            return var_error2, angle_mean_rot2

    def _calc_error_and_rot(self, arr_angle):
        # 移除均值，算方差，方差越大，越不成比例
        arr_abs_error = np.c_[np.abs(arr_angle - arr_angle.mean()), arr_angle]
        arr_abs_error = arr_abs_error[arr_abs_error[:, 0].argsort()]
        abs_error = arr_abs_error[:-cfg.num_filter]
        var_error = np.var(abs_error[:, 1])
        angle_mean_rot = abs_error[:, 1].mean()
        return var_error, angle_mean_rot

    def calc_char_prop_var_error(self, arr2d_kp_tmp, arr2d_kp_test):
        # 文字成比例程度

        # 计算模板图像上两个点的索引
        lst_two_point_idx = self._get_lst_two_point_idx(arr2d_kp_tmp)

        # 计算任意两个点形成的向量的夹角是否变化很小,可能偶尔有几个匹配错误的点，可以过滤掉
        var_error, angle_mean_rot = self._get_var_error(arr2d_kp_tmp, arr2d_kp_test, lst_two_point_idx)

        return var_error, angle_mean_rot

    def cut_pic(self):
        # 粗略裁剪出身份证（多带一些边框，未来再裁剪）
        self.arr_corner_best = self.arr_corner_best.reshape(
            [self.arr_corner_best.shape[0], self.arr_corner_best.shape[-1]])

        self.arr_corner0 = self.arr_corner_best  # / self.scale
        # 这个点的顺序是随机的，所以直接拿左上角和右下角
        x_min = min([i[0] for i in self.arr_corner0])
        x_max = max([i[0] for i in self.arr_corner0])

        y_min = min([i[1] for i in self.arr_corner0])
        y_max = max([i[1] for i in self.arr_corner0])

        # 多带一些边框
        len_x = x_max - x_min
        len_y = y_max - y_min

        h, w = self.img_test0.shape

        x_min_cut = int(max(0, x_min - cfg.cut_extend * len_x))
        x_max_cut = int(min(w, x_max + cfg.cut_extend * len_x))
        y_min_cut = int(max(0, y_min - cfg.cut_extend * len_y))
        y_max_cut = int(min(h, y_max + cfg.cut_extend * len_y))

        self.img_test0 = self.img_test0[y_min_cut:y_max_cut, x_min_cut:x_max_cut]
        self.img_test_rgb0 = self.img_test_rgb0[y_min_cut:y_max_cut, x_min_cut:x_max_cut]

    def calc_arr_corner_best(self):
        # 粗略裁剪出身份证（多带一些边框，未来再裁剪）
        self.arr_corner_best = self.arr_corner_best.reshape(
            [self.arr_corner_best.shape[0], self.arr_corner_best.shape[-1]])

        # 点的顺序 左上，左下，右下，右上
        for idx in range(len(self.arr_corner_best)):
            if self.arr_corner_best[idx][0] < 0:
                self.arr_corner_best[idx][0] = 0
            if self.arr_corner_best[idx][0] > self.w:
                self.arr_corner_best[idx][0] = self.w
            if self.arr_corner_best[idx][1] < 0:
                self.arr_corner_best[idx][1] = 0
            if self.arr_corner_best[idx][1] > self.h:
                self.arr_corner_best[idx][1] = self.h

        # self.arr_corner_best = np.int32(self.arr_corner_best)

    def check_card(self):
        # 对边相差不大，长宽比适宜身份证
        p1, p2, p3, p4 = self.arr_corner_best
        # 默认已经是方形了
        line_up = p4[0] - p1[0]
        line_left = p2[1] - p1[1]

        # pass

    def perspective_transform(self):

        pts = self.arr_corner_best

        p1, p2, p3, p4 = self.arr_corner_best
        # len_line_up = abs(p4[0] - p1[0])
        # len_line_down = abs(p3[0] - p2[0])
        # len_line_left = p2[1] - p1[1]
        # len_line_right = p3[1] - p4[1]
        # scale = ((len_line_up + len_line_down) / 2) / ((len_line_left + len_line_right) / 2)
        h = cfg.h_output
        w = cfg.w_output
        pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])

        # 先得确定透视变换的系数：

        M = cv2.getPerspectiveTransform(pts, pts1)

        # 对原图进行这个变换：

        self.img_test_res = cv2.warpPerspective(self.img_test_rgb, M, (w, h))

    def save_pic(self, dir_name, img_save, need_ratio_w_h=False, need_border=False):
        lst2d = []
        if need_border:
            lst2d.append(["border",  "%06d" % self.border_pix])
        lst2d.append(["rect", str(self.error_rect_best)[:5]])
        lst2d.append(["var", str(self.error_var_angle_best)[:5]])
        lst2d.append(["points", str(self.num_match_points)])
        if need_ratio_w_h:
            p1, p2, p3, p4 = self.arr_corner_best
            # 默认已经是方形了
            line_up = p4[0] - p1[0]
            line_left = p2[1] - p1[1]
            ratio_w_h = line_up / line_left
            lst2d.append(["ratiowh", str(ratio_w_h)[:5]])
        lst2d.append(["name", self.img_name])

        s = dir_name
        if self.flag_find_border:
            s += "a"
        for i in lst2d:
            s += "-".join(i) + "_"
        s = s[:-1]
        # img_save = cv2.resize(img_save,(160,100))
        cv2.imwrite(s, img_save)

    def remove_border(self):
        # 移除左右边框
        # canny(): 边缘检测
        # img1 = cv2.GaussianBlur(cv2.cvtColor(self.img_test_res, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        # canny = cv2.Canny(img1, 50, 90)
        # sobelx = cv2.Sobel(cv2.cvtColor(self.img_test_res, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
        # sobelx = cv2.convertScaleAbs(sobelx)  # 也可以用 sobelx = np.absolute(gradX)

        img1 = cv2.GaussianBlur(cv2.cvtColor(self.img_test_res, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        canny = cv2.Canny(img1, 50, 150)


        img_expand = cv2.dilate(canny, kernel=kernel)

        # cv2.imwrite("/data/zhaoyichen/filename.png", canny.astype("uint8") * 255)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 膨胀
        # dst = cv2.dilate(canny, kernel=kernel) == 255

        # 去除人像位置 上124  下 269

        right = 490  # 右边从多少像素开始判断边框
        # 将头像覆盖
        img_expand[124:269, 400:520] = 0

        # 上下覆盖几个像素
        img_expand[:20, :] = 0
        img_expand[-20:, :] = 0

        arr_count_pix = np.sum(img_expand==255, axis=0)[right:]  # 取出右侧片段，并统计个数

        # 如果没有3列以上的0，说明边界不清晰，就放弃截取
        is_border_clear = sum(arr_count_pix == 0)>3
        if is_border_clear:
            self.border_pix = arr_count_pix.max()
            if self.border_pix>100:#如果纵向超过140个像素，就认为找到边框，向左边找到全0的列
                a = arr_count_pix[:arr_count_pix.argmax()]
                has_all_zero = sum(a==0)>0
                if has_all_zero:
                    right+=len(a)#游标右边界变为检测到的最大边界
                    step_move = 20 # 从最大边界最多向左移动X个像素，找不到就算了，就截取到这
                    num_move = 0
                    for i in a[::-1]:
                        num_move+=1
                        if i==0 or num_move> step_move:
                            break
                        right-=1
                # self.img_test_res = img_expand[:, :right]
                self.img_test_res = self.img_test_res[:, :right]
        else:
            self.border_pix = 999
            # self.img_test_res = img_expand
        print()


def f(lst_url_img):
    for url_img in lst_url_img:
        with lock:
            mydict["num"] += 1
            print(mydict["num"])

        img_test_rgb = cv2.imread(url_img, 1)
        h, w, _ = img_test_rgb.shape

        if min(h, w) < cfg.min_h_or_w:  # 舍弃尺寸较小的照片
            os.system(f"cp {url_img} {cfg.dir_small_input}")
            print(cfg.dir_small_input)
            continue

        stepA = StepA(url_img, img_test_rgb)
        stepA.run()


if __name__ == '__main__':
    os.system(f"rm -f {cfg.dir_small_input}*")
    os.system(f"rm -f {cfg.dir_vague}*")
    os.system(f"rm -f {cfg.dir_distortion}*")
    os.system(f"rm -f {cfg.dir_incomplete}*")
    os.system(f"rm -f {cfg.dir_output}*")

    if not os.path.exists(cfg.dir_small_input): os.mkdir(cfg.dir_small_input)
    if not os.path.exists(cfg.dir_vague): os.mkdir(cfg.dir_vague)
    if not os.path.exists(cfg.dir_distortion): os.mkdir(cfg.dir_distortion)
    if not os.path.exists(cfg.dir_incomplete): os.mkdir(cfg.dir_incomplete)
    if not os.path.exists(cfg.dir_output): os.mkdir(cfg.dir_output)

    MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
    # 创建设置FLANN匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Initiate SIFT detector创建sift检测器
    sift = cv2.xfeatures2d.SIFT_create()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 膨胀

    dir_tmp = cfg.dir_tmp
    lst_url_imgtmp = [dir_tmp + i for i in os.listdir(dir_tmp)]
    lst_url_imgtmp.sort()

    lst_tmp_info = []
    for url_imgtmp in lst_url_imgtmp:
        img_tmp = cv2.imread(url_imgtmp, 0)
        kp_tmp, des_tmp = sift.detectAndCompute(img_tmp, None)
        lst_tmp_info.append({"url_imgtmp": url_imgtmp, "img_tmp": img_tmp, "kp_tmp": kp_tmp, "des_tmp": des_tmp})

    # dir_p = "E:/data_static/20210720shenfenzheng_raw/"
    # dir_input = "/data/zhaoyichen/data/idcard/" #第一批 1920
    # dir_input = "/data/zhaoyichen/data/idcard2/" #第二批 2880
    dir_input = "/data/zhaoyichen/data/ec-gpaas-idcard-check/idcard4/" #第1/2批合集 4816
    lines = [dir_input + i for i in os.listdir(dir_input)]
    # lines = [dir_input + "C0456601101732109000002.jpg"]  # 1399495
    random.seed(666)
    random.shuffle(lines)
    # lines = lines[:1000] 
    core_num = 100
    # core_num = 1
    manager = Manager()
    lock = Lock()
    mydict = manager.dict({"num": 0})
    print(mydict)

    core_data_len = math.ceil(len(lines) / core_num)

    lst_p = []
    for i in range(core_num):
        p = Process(target=f,
                    args=(lines[i * core_data_len:i * core_data_len + core_data_len],))
        p.start()
        lst_p.append(p)
    [p.join() for p in lst_p]
    print(mydict)
    # os.system(f"tar -zcf res.tar.gz {cfg.dir_output}")
    os.system(f"tar -zcf res.tar.gz {cfg.dir_output} {cfg.dir_small_input} {cfg.dir_vague} {cfg.dir_distortion} {cfg.dir_incomplete}")
