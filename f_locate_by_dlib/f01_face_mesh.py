import os
import shutil
import time

import cv2
import imutils
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# For static images:
# IMAGE_FILES = ["../img_idcard/1254139.jpg"]
dir_in = "../img_idcard/"
IMAGE_FILES = [dir_in + i for i in os.listdir(dir_in) if ".jpg" in i]

dir_out = "../img_idcard_face_mesh/"
dir_out_SIFT_fail = "../img_idcard_SIFT_fail/"

if os.path.exists(dir_out):
    shutil.rmtree(dir_out)
os.makedirs(dir_out)

if os.path.exists(dir_out_SIFT_fail):
    shutil.rmtree(dir_out_SIFT_fail)
os.makedirs(dir_out_SIFT_fail)


with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.9) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        print(file)
        image = cv2.imread(file)

        h, w,c = image.shape
        scale_factor = 1000 / max(h, w)
        image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        # image = imutils.rotate(image, 150)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = time.time()
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(img)
        print(time.time() - a)
        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            shutil.copy(file, dir_out_SIFT_fail)
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        # cv2.imwrite(str(idx) + '.png', annotated_image)

        cv2.imwrite(dir_out + os.path.basename(file), annotated_image)


