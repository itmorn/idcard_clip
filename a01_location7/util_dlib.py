import numpy as np
import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()
os.chdir(os.path.dirname(os.path.dirname(__file__)))
# predictor5 = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
# predictor68 = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# predictor5 = dlib.shape_predictor("../models/shape_predictor_5_face_landmarks.dat")
# predictor68 = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")


def get_lst_detected_face(img_gray):
    lst_detected_face = detector(img_gray, 0)
    return lst_detected_face


def get_face_keypoint(img,detected_face,predictor):
    landmarks = [[p.x, p.y] for p in predictor(img, detected_face).parts()]
    return landmarks


if __name__ == '__main__':
    # cv2读取图像
    img = cv2.imread("../reflection000000_C0602308106352109000039.png")

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lst_detected_face = get_lst_detected_face(img_gray)

    if lst_detected_face:
        face_keypoint = get_face_keypoint(img,lst_detected_face[0],predictor5)
        print(face_keypoint)
        # # 人脸数rects
        # for i in range(len(lst_detected_face)):
        #     landmarks = np.matrix([[p.x, p.y] for p in predictor5(img, lst_detected_face[i]).parts()])
        #     for idx, point in enumerate(landmarks):
        #         # 68点的坐标
        #         pos = (point[0, 0], point[0, 1])
        #
        #         # 利用cv2.circle给每个特征点画一个圈，共68个
        #         cv2.circle(img, pos, 2, color=(0, 255, 0))
        #         # 利用cv2.putText输出1-68
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cv2.putText(img, str(idx + 1), None, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.imwrite("/data/zhaoyichen/filename.png", img)
        # cv2.namedWindow("img", 2)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)