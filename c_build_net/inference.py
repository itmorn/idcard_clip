"""
@Auth: itmorn
@Date: 2022/6/17-10:57
@Email: 12567148@qq.com
"""
import copy
import json

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import cv2
import lenet2

import torch.nn.functional as F

transform1 = transforms.Compose([
    # transforms.Resize(60),
    # transforms.CenterCrop(60),
    transforms.ToTensor()
])

class_names = ['#s001#', '#s002#', '#s003#']


# model = models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 62)
# # model = model.to(device)
# model = torch.nn.DataParallel(model).to(device)

PATH = '15.pth'
model = lenet2.LeNet()
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)
# model.eval()
lst_pics = []
dir_img_idcard = "../img_idcard/"
lst_url_jsn = [dir_img_idcard + i for i in os.listdir(dir_img_idcard) if i.endswith(".json")]
lst_url_jsn.sort()
os.system("rm -f res_pic/*")
for url_jsn in lst_url_jsn:
    jsn = json.loads(open(url_jsn).read())
    imagePath = jsn['imagePath']
    print(imagePath)
    shapes = jsn['shapes']
    if len(shapes) != 4:
        print(imagePath)
        raise Exception("len(shapes)!=4")

    lst_corner = [i['points'][0] for i in jsn['shapes']]
    img = cv2.imread(dir_img_idcard + imagePath)
    height, width = img.shape[:-1]
    height_new = 600
    width_new = 800

    # 对坐标点进行缩放矫正
    width_factor = width_new / width
    height_factor = height_new / height

    img = np.array(img, dtype=np.float32)
    img = cv2.resize(img, (width_new, height_new))

    img0 = copy.deepcopy(img)

    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
    img = torch.tensor(img,requires_grad=False)

    with torch.set_grad_enabled(False):
        inputs = img.to(device)
        outputs = model(inputs)
        out_arr = model(inputs).cpu().numpy()[0]
        for i in range(4):
            x = out_arr[i*2]*800
            y = out_arr[i*2+1]*600
            img0[int(y), int(x), :] = 0
            img0[int(y), int(x), -1] = 255
            pass

    cv2.imwrite(f"res_pic/{imagePath}", img0.astype("uint8"))

#         lst_pics.append(img1)
#
# inputs = torch.cat(lst_pics).to(device)
