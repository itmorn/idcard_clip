"""
@Auth: itmorn
@Date: 2022/6/17-10:53
@Email: 12567148@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.in_channels = 3
        self.conv1 = self.__make_layer(out_channels = 24, num=1)
        self.conv2 = self.__make_layer(out_channels = 24, num=1)
        self.conv3 = self.__make_layer(out_channels = 24, num=1)
        self.conv4 = self.__make_layer(out_channels = 24, num=1)
        self.conv5 = self.__make_layer(out_channels = 24, num=1)

        # self.conv_trans1 = self.__make_layer_transpose(in_channels=self.in_channels,out_channels=self.in_channels//2,kernel_size=4, stride=2, padding=1, num=1)
        # self.conv_trans2 = self.__make_layer_transpose(in_channels=self.in_channels,out_channels=self.in_channels//2,kernel_size=4, stride=2, padding=1, num=1)
        # self.conv_trans3 = self.__make_layer_transpose(in_channels=self.in_channels,out_channels=self.in_channels//2,kernel_size=4, stride=2, padding=1, num=1)
        # self.conv_trans4 = self.__make_layer_transpose(in_channels=self.in_channels,out_channels=1,kernel_size=4, stride=2, padding=1, num=1)


        self.fc1 = nn.Linear(24*18*25, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, 8)

    def __make_layer(self, out_channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1))  # same padding
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def __make_layer_transpose(self, in_channels,out_channels,kernel_size=4, stride=2, padding=1, num=1):
        layers = []
        for i in range(num):
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))  # same padding
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3(out)
        out = F.max_pool2d(out, 2)
        out = self.conv4(out)
        out = F.max_pool2d(out, 2)
        out = self.conv5(out)
        out = F.max_pool2d(out, 2)
        # out = self.conv_trans1(out)
        # out = self.conv_trans2(out)
        # out = self.conv_trans3(out)
        # out = self.conv_trans4(out)
        # out = F.max_pool2d(out, 2)
        # out = self.conv6(out)
        # out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out