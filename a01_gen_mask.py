"""
@Auth: itmorn
@Date: 2022/6/15-21:55
@Email: 12567148@qq.com

移除红色，生成用于标注的模板
"""
import os
import cv2
import numpy as np


def remove_red(dir_pic_ori):
    lst_name = [dir_pic_ori + i for i in os.listdir(dir_pic_ori)]
    for name in lst_name:
        print(name)
        img = cv2.imread(name)
        img[:, :, -1] = np.clip(img[:, :, -1], 1, 255)
        cv2.imwrite(f"img_idcard_mask/{os.path.basename(name)}",img)
        pass


if __name__ == '__main__':
    remove_red(dir_pic_ori="img_idcard/")
