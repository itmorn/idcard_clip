"""
@Auth: itmorn
@Date: 2022/6/20-9:58
@Email: 12567148@qq.com
"""
import os
import shutil

import cv2
import numpy as np

from e_sift.util_sift import UtilSift

from e_sift import util_rotate
from multiprocessing import Process, Manager,Lock
import os
import math
from time import sleep


def cut_card(url_img):
    target = cv2.imread(url_img, 1)

    # 输入待检测图像，计算最佳匹配 ，逆时针的坐标序列 （x,y）
    best_dst = UtilSift.calc_good_match(target, is_show=is_show)
    if best_dst is None:
        return

        # 计算旋转角度照片原图进行旋转，然后裁剪
    rotate_angle = util_rotate.calc_rotate_angle(best_dst)

    # 旋转图像
    image_rotate, M = util_rotate.rotate_bound(target, rotate_angle)

    # 旋转坐标点
    target_point = util_rotate.rotate(best_dst, M)

    # 裁剪
    target_point = np.array(target_point)

    x_min, x_max = int(target_point[:, 0].min()), int(target_point[:, 0].max())
    y_min, y_max = int(target_point[:, 1].min()), int(target_point[:, 1].max())

    # 获取身份证框体的长度
    length_SIFT = x_max - x_min
    length_extend = int(0.2 * length_SIFT)
    # 对坐标进行扩展，为了把身份证的边缘囊括进来
    x_min = max(x_min - length_extend, 0)
    x_max = x_max + length_extend
    y_min = max(y_min - length_extend, 0)
    y_max = y_max + length_extend

    image_crop = image_rotate[y_min:y_max + 1, x_min:x_max + 1, :]
    return image_crop

def f(mydict,lst_file):
    for url_img in lst_file:
        # if "1395733.jpg" not in url_img:
        #     continue
        with lock:
            mydict["num"]+=1
            print(mydict["num"])

        print(url_img)
        image_crop = cut_card(url_img=url_img)
        if image_crop is None:
            shutil.copy(url_img, dir_out_SIFT_fail)
        else:
            # image_crop = cv2.resize(image_crop, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(dir_out + os.path.basename(url_img), image_crop)

if __name__ == '__main__':
    dir_in = "../img_idcard/"
    dir_in = "/data01/zhaoyichen/data/ec-gpaas-idcard-check/idcard/"

    dir_out = "img_idcard_crop/"
    dir_out_SIFT_fail = "img_idcard_SIFT_fail/"

    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)

    if os.path.exists(dir_out_SIFT_fail):
        shutil.rmtree(dir_out_SIFT_fail)
    os.makedirs(dir_out_SIFT_fail)

    lst_file = [dir_in + i for i in os.listdir(dir_in) if ".jpg" in i]
    is_show = False#True#

    manager = Manager()
    lock = Lock()
    mydict = manager.dict({"num": 0})
    print(mydict)

    core_num = 48
    core_data_len = math.ceil(len(lst_file) / core_num)

    lst_p = []
    for i in range(core_num):
        p = Process(target=f, args=(mydict, lst_file[i * core_data_len:i * core_data_len + core_data_len]))
        p.start()
        lst_p.append(p)
    [p.join() for p in lst_p]
    print(mydict)

    os.system(f"tar -zcf res.tar.gz {dir_out} {dir_out_SIFT_fail}")




# image_crop = cut_card(url_img='../img_idcard/1254139.jpg')
# image_crop = cut_card(url_img='../img_idcard/1254139.jpg')
# cv2.imwrite("4.png", image_crop)
