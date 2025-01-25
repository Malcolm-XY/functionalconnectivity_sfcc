# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:06:57 2024

@author: 18307
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# %% adaptive
class CNN_2layers_adaptive_avgpool_2(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 avgpool layers: p1 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_avgpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_2layers_adaptive_maxpool_2(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 maxpool layers: p1 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_maxpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_2layers_adaptive_avgpool_3(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 avgpool layers: p1 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_avgpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_2layers_adaptive_maxpool_3(nn.Module):
    """
    this model is identified as:
    2 convolution layers: c1 = c2 = 3, 1
    1 maxpool layers: p1 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_2layers_adaptive_maxpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_3layers_adaptive_avgpool_3(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 avgpool layers: p1 = p2 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_avgpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_3layers_adaptive_maxpool_3(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 maxpool layers: p1 = p2 = 3, 3
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_maxpool_3, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class CNN_3layers_adaptive_avgpool_2(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 avgpool layers: p1 = p2 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_avgpool_2, self).__init__()

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
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_3layers_adaptive_maxpool_2(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    2 maxpool layers: p1 = p2 = 2, 2
    1 global maxpool
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_adaptive_maxpool_2, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1)) 

        # 全连接层
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# %% fixed
class CNN_3layers_avgpool(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    3 avgpool layers: p1 = p2 = p3 = 3, 3
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_avgpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(in_features=3*3*128, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class CNN_3layers_maxpool(nn.Module):
    """
    this model is identified as:
    3 convolution layers: c1 = c2 = c3 = 3, 1
    3 maxpool layers: p1 = p2 = p3 = 3, 3
    """
    def __init__(self, channels=3, num_classes=3):
        super(CNN_3layers_maxpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 全连接层
        self.fc1 = nn.Linear(in_features=3*3*128, out_features=128)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# %% 4 layers
class CNN_4layers_avgpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN_4layers_avgpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=3)

        # 第四层卷积 + BatchNorm + 池化
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.AvgPool2d(kernel_size=3, stride=3)

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
    
class CNN_4layers_maxpool(nn.Module):
    def __init__(self, channels=3, num_classes=3):
        super(CNN_4layers_maxpool, self).__init__()

        # 第一层卷积 + BatchNorm + 池化
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第二层卷积 + BatchNorm + 池化
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第三层卷积 + BatchNorm + 池化
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)

        # 第四层卷积 + BatchNorm + 池化
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3)

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
