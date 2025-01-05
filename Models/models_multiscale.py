# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:24:22 2024

@author: usouu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN, self).__init__()

        # 分支1：小尺度特征
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支2：中尺度特征
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支3：大尺度特征
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 融合特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 分支1
        branch1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        branch1 = self.branch1_pool1(branch1)
        branch1 = F.relu(self.branch1_bn2(self.branch1_conv2(branch1)))
        branch1 = self.branch1_pool2(branch1)

        # 分支2
        branch2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        branch2 = self.branch2_pool1(branch2)
        branch2 = F.relu(self.branch2_bn2(self.branch2_conv2(branch2)))
        branch2 = self.branch2_pool2(branch2)

        # 分支3
        branch3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        branch3 = self.branch3_pool1(branch3)
        branch3 = F.relu(self.branch3_bn2(self.branch3_conv2(branch3)))
        branch3 = self.branch3_pool2(branch3)

        # 融合分支
        branch1 = self.global_pool(branch1).view(x.size(0), -1)
        branch2 = self.global_pool(branch2).view(x.size(0), -1)
        branch3 = self.global_pool(branch3).view(x.size(0), -1)

        # 拼接特征
        combined = torch.cat([branch1, branch2, branch3], dim=1)

        # 全连接层
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)

        return x

class SimpleMultiScaleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(SimpleMultiScaleCNN, self).__init__()
        
        # 第一条分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=7, padding=3),  # padding 计算为 (kernel_size - 1) // 2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出固定为 1x1
        )
        
        # 第二条分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=9, padding=4),  # padding 计算为 (kernel_size - 1) // 2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出固定为 1x1
        )
        
        # 第三条分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=5, padding=2),  # padding 计算为 (kernel_size - 1) // 2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出固定为 1x1
        )
        
        # 合并和输出层
        self.concat_norm = nn.BatchNorm1d(48)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(48, 8)
        self.fc2 = nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 分支输出
        out1 = self.branch1(x).view(x.size(0), -1)  # 展平
        out2 = self.branch2(x).view(x.size(0), -1)  # 展平
        out3 = self.branch3(x).view(x.size(0), -1)  # 展平
        
        # 合并分支
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.concat_norm(out)
        out = self.dropout(out)
        
        # 全连接层和分类
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out