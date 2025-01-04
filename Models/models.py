# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:06:57 2024

@author: 18307
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2DModel(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN2DModel, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第四层卷积 + BatchNorm + 池化
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=5)

        # 全连接层
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x