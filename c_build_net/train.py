"""
@Auth: itmorn
@Date: 2022/6/17-10:56
@Email: 12567148@qq.com
"""
import json
import os

from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"#
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import cv2
import random
import lenet2

dir_img_idcard = "../img_idcard/"
lst_url_jsn = [dir_img_idcard + i for i in os.listdir(dir_img_idcard) if i.endswith(".json")]
lst_url_jsn.sort()

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_hat, label):
        # y_hat0 = torch.sigmoid(y_hat[0])
        # loss0 = torch.mean((y_hat0-target_resize)**2)*100
        # y_hat = torch.sigmoid(y_hat)
        loss = torch.mean((y_hat-label)**2)

        return loss

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_num = 0
        # Each epoch has a training and validation phase
        for phase in ['train']: #, 'val'
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            batch_size = 30

            num_train_batch = 10
            while num_train_batch>0:
                num_train_batch-=1
                lst_url_jsn_batch = np.random.choice(a=lst_url_jsn, size=batch_size, replace=False).tolist()
                lst_input = []
                lst_label = []
                for url_jsn in lst_url_jsn_batch:
                    jsn = json.loads(open(url_jsn).read())
                    imagePath = jsn['imagePath']
                    shapes = jsn['shapes']
                    if len(shapes) != 4:
                        print(imagePath)
                        raise Exception("len(shapes)!=4")
                    lst_corner = [i['points'][0] for i in jsn['shapes']]

                    # if "1369049.jpg"!=imagePath:
                    #     continue
                    # print(imagePath)
                    img = cv2.imread(dir_img_idcard+imagePath)
                    height, width = img.shape[:-1]
                    height_new = 600
                    width_new = 800

                    # 对坐标点进行缩放矫正
                    width_factor = width_new / width
                    height_factor = height_new / height


                    img = np.array(img,dtype=np.float32)
                    img = cv2.resize(img,(width_new,height_new))

                    img = img / 255.0
                    img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
                    img = torch.tensor(img)

                    batch_num += 1

                    inputs = img.to(device)
                    lst_input.append(inputs)

                    # ts_target = torch.zeros_like(inputs[0,0,:,:]).to(device)
                    for i in range(len(lst_corner)):
                        lst_corner[i][0]*=width_factor
                        lst_corner[i][0]/=800
                        lst_corner[i][1]*=height_factor
                        lst_corner[i][1] /= 600
                    label = torch.tensor(lst_corner).reshape(-1,8).to(device)
                    lst_label.append(label)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = torch.concat(lst_input)
                    outputs = model(inputs)

                    label = torch.concat(lst_label)
                    # _, preds = torch.max(outputs, 1)
                    # outputs = outputs.squeeze()
                    loss = criterion(outputs, label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss = loss.item() * inputs.size(0)
                print(running_loss)

        PATH = str(epoch) + '.pth'
        torch.save(model.state_dict(), PATH)


model_ft = lenet2.LeNet()

# PATH = '15.pth'
# model_ft = lenet2.LeNet()
# model_ft.load_state_dict(torch.load(PATH, map_location=device))

# model_ft = models.resnet18(pretrained=False)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, len(class_names))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

criterion = My_loss()
# criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters())

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25000)