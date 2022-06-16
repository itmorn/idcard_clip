"""
@Auth: itmorn
@Date: 2022/6/16-11:13
@Email: 12567148@qq.com
"""
import json
import os
import cv2
import numpy as np


def calc_corner(url_jsn):
    jsn = json.loads(open(url_jsn).read())
    shapes = jsn['shapes']
    # 标注点的顺序，上左 上右  顺时针 每条边上2个点确定一条直线  一共8个点
    # 产生的四条线的顺序为  上 右 下 左
    lst_line = [] # 拟合4条直线
    for i in range(4):
        p1 = np.array(shapes[i*2]['points'][0])
        p2 = np.array(shapes[i*2+1]['points'][0])
        ris = p2-p1
        k = ris[1] / ris[0]
        b = p1[1]-k*p1[0]
        lst_line.append([k,b])

    # 四个角的顺序为  左上 右上 右下  左下
    lst_corner = []
    def calc_cross_by_two_line(line1,line2):
        k1, b1 = line1
        k2, b2 = line2
        point_x = (b2 - b1) / (k1 - k2)
        point_y = k1 * point_x + b1
        return [point_x, point_y]


    lst_corner.append(calc_cross_by_two_line(lst_line[0],lst_line[3]))
    lst_corner.append(calc_cross_by_two_line(lst_line[0],lst_line[1]))
    lst_corner.append(calc_cross_by_two_line(lst_line[1],lst_line[2]))
    lst_corner.append(calc_cross_by_two_line(lst_line[2],lst_line[3]))
    jsn["lst_corner"] = lst_corner

    s = json.dumps(jsn,indent=4)
    f = open(url_jsn,"w")
    f.write(s+"\n")
    f.close()


if __name__ == '__main__':
    # 获取图像列表
    dir_img_idcard = "../img_idcard/"
    lst_url_jsn = [dir_img_idcard + i for i in os.listdir(dir_img_idcard) if i.endswith(".json")]
    lst_url_jsn.sort()

    for url_jsn in lst_url_jsn:
        # if "1352471" not in img_url_mask:
        #     continue
        # 计算四个角的坐标
        calc_corner(url_jsn)
