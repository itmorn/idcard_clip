"""
整张大图中寻找人脸比较容易，但是直接寻找边缘比较难，同时人脸检测模型也可以粗略的过滤掉质量不好的图像
可以通过人脸检测 对图像进行一下裁剪和调整方向，为有利于后续的图像标注

人脸检测裁剪+旋转（可选）
模板匹配裁剪+旋转（可选）
DNN计算证件四个顶点坐标
透视变换



"""
import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# cv2读取图像
img = cv2.imread("../img_idcard/1254139.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])

        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 2, color=(0, 255, 0))
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx + 1), None, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imwrite("filename.png", img)
# cv2.namedWindow("img", 2)
# cv2.imshow("img", img)
# cv2.waitKey(0)