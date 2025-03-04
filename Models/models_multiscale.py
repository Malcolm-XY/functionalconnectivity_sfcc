# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:24:22 2024

@author: usouu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN_2layers_avgpool(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_2layers_avgpool, self).__init__()

        # Branch 1: Small-scale features
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # Branch 2: Medium-scale features
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # Branch 3: Large-scale features
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.AvgPool2d(kernel_size=3, stride=3)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.AvgPool2d(kernel_size=3, stride=3)

        # Fusion features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 3, 128)  # Concatenate three branches
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        # Branch 1
        x1 = self.branch1_pool1(F.relu(self.branch1_bn1(self.branch1_conv1(x))))
        x1 = self.branch1_pool2(F.relu(self.branch1_bn2(self.branch1_conv2(x1))))
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)

        # Branch 2
        x2 = self.branch2_pool1(F.relu(self.branch2_bn1(self.branch2_conv1(x))))
        x2 = self.branch2_pool2(F.relu(self.branch2_bn2(self.branch2_conv2(x2))))
        x2 = self.global_pool(x2)
        x2 = torch.flatten(x2, 1)

        # Branch 3
        x3 = self.branch3_pool1(F.relu(self.branch3_bn1(self.branch3_conv1(x))))
        x3 = self.branch3_pool2(F.relu(self.branch3_bn2(self.branch3_conv2(x3))))
        x3 = self.global_pool(x3)
        x3 = torch.flatten(x3, 1)

        # Fusion
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class MultiScaleCNN_2layers_maxpool(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_2layers_avgpool, self).__init__()

        # Branch 1: Small-scale features
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Branch 2: Medium-scale features
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Branch 3: Large-scale features
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Fusion features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 3, 128)  # Concatenate three branches
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        # Branch 1
        x1 = self.branch1_pool1(F.relu(self.branch1_bn1(self.branch1_conv1(x))))
        x1 = self.branch1_pool2(F.relu(self.branch1_bn2(self.branch1_conv2(x1))))
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)

        # Branch 2
        x2 = self.branch2_pool1(F.relu(self.branch2_bn1(self.branch2_conv1(x))))
        x2 = self.branch2_pool2(F.relu(self.branch2_bn2(self.branch2_conv2(x2))))
        x2 = self.global_pool(x2)
        x2 = torch.flatten(x2, 1)

        # Branch 3
        x3 = self.branch3_pool1(F.relu(self.branch3_bn1(self.branch3_conv1(x))))
        x3 = self.branch3_pool2(F.relu(self.branch3_bn2(self.branch3_conv2(x3))))
        x3 = self.global_pool(x3)
        x3 = torch.flatten(x3, 1)

        # Fusion
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class MultiScaleCNN_4layers_avgpool(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_4layers_avgpool, self).__init__()

        # 分支1：小尺度特征
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.branch1_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.branch1_bn3 = nn.BatchNorm2d(128)
        self.branch1_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.branch1_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch1_bn4 = nn.BatchNorm2d(256)
        self.branch1_pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 分支2：中尺度特征
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.branch2_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.branch2_bn3 = nn.BatchNorm2d(128)
        self.branch2_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.branch2_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch2_bn4 = nn.BatchNorm2d(256)
        self.branch2_pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 分支3：大尺度特征
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.branch3_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.branch3_bn3 = nn.BatchNorm2d(128)
        self.branch3_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.branch3_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch3_bn4 = nn.BatchNorm2d(256)
        self.branch3_pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 融合特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256 * 3, 512)  # Concatenate three branches
        self.dropout1 = nn.Dropout(p=0.25)        

        self.fc2 = nn.Linear(512, num_classes)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        # 分支1
        x1 = self.branch1_pool1(F.relu(self.branch1_bn1(self.branch1_conv1(x))))
        x1 = self.branch1_pool2(F.relu(self.branch1_bn2(self.branch1_conv2(x1))))
        x1 = self.branch1_pool3(F.relu(self.branch1_bn3(self.branch1_conv3(x1))))
        x1 = self.branch1_pool4(F.relu(self.branch1_bn4(self.branch1_conv4(x1))))
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)

        # 分支2
        x2 = self.branch2_pool1(F.relu(self.branch2_bn1(self.branch2_conv1(x))))
        x2 = self.branch2_pool2(F.relu(self.branch2_bn2(self.branch2_conv2(x2))))
        x2 = self.branch2_pool3(F.relu(self.branch2_bn3(self.branch2_conv3(x2))))
        x2 = self.branch2_pool4(F.relu(self.branch2_bn4(self.branch2_conv4(x2))))
        x2 = self.global_pool(x2)
        x2 = torch.flatten(x2, 1)

        # 分支3
        x3 = self.branch3_pool1(F.relu(self.branch3_bn1(self.branch3_conv1(x))))
        x3 = self.branch3_pool2(F.relu(self.branch3_bn2(self.branch3_conv2(x3))))
        x3 = self.branch3_pool3(F.relu(self.branch3_bn3(self.branch3_conv3(x3))))
        x3 = self.branch3_pool4(F.relu(self.branch3_bn4(self.branch3_conv4(x3))))
        x3 = self.global_pool(x3)
        x3 = torch.flatten(x3, 1)

        # 融合分支
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class MultiScaleCNN_4layers_maxpool(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_4layers_maxpool, self).__init__()

        # 分支1：小尺度特征
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.branch1_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.branch1_bn3 = nn.BatchNorm2d(128)
        self.branch1_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch1_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch1_bn4 = nn.BatchNorm2d(256)
        self.branch1_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支2：中尺度特征
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.branch2_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.branch2_bn3 = nn.BatchNorm2d(128)
        self.branch2_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch2_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch2_bn4 = nn.BatchNorm2d(256)
        self.branch2_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支3：大尺度特征
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.branch3_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.branch3_bn3 = nn.BatchNorm2d(128)
        self.branch3_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch3_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.branch3_bn4 = nn.BatchNorm2d(256)
        self.branch3_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 融合特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256 * 3, 512)  # Concatenate three branches
        self.dropout1 = nn.Dropout(p=0.25)        

        self.fc2 = nn.Linear(512, num_classes)  # Corrected dimension
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x):
        # 分支1
        x1 = self.branch1_pool1(F.relu(self.branch1_bn1(self.branch1_conv1(x))))
        x1 = self.branch1_pool2(F.relu(self.branch1_bn2(self.branch1_conv2(x1))))
        x1 = self.branch1_pool3(F.relu(self.branch1_bn3(self.branch1_conv3(x1))))
        x1 = self.branch1_pool4(F.relu(self.branch1_bn4(self.branch1_conv4(x1))))
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)

        # 分支2
        x2 = self.branch2_pool1(F.relu(self.branch2_bn1(self.branch2_conv1(x))))
        x2 = self.branch2_pool2(F.relu(self.branch2_bn2(self.branch2_conv2(x2))))
        x2 = self.branch2_pool3(F.relu(self.branch2_bn3(self.branch2_conv3(x2))))
        x2 = self.branch2_pool4(F.relu(self.branch2_bn4(self.branch2_conv4(x2))))
        x2 = self.global_pool(x2)
        x2 = torch.flatten(x2, 1)

        # 分支3
        x3 = self.branch3_pool1(F.relu(self.branch3_bn1(self.branch3_conv1(x))))
        x3 = self.branch3_pool2(F.relu(self.branch3_bn2(self.branch3_conv2(x3))))
        x3 = self.branch3_pool3(F.relu(self.branch3_bn3(self.branch3_conv3(x3))))
        x3 = self.branch3_pool4(F.relu(self.branch3_bn4(self.branch3_conv4(x3))))
        x3 = self.global_pool(x3)
        x3 = torch.flatten(x3, 1)

        # 融合分支
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class MultiScaleCNN_1layers_avgpool(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_1layers_avgpool, self).__init__()
        
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
    
class MultiScaleCNN_1layers_maxpool(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_1layers_maxpool, self).__init__()
        
        # 第一条分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 第二条分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 第三条分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        
        # 合并和输出层
        self.concat_norm = nn.BatchNorm1d(48)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(48, 8)
        self.fc2 = nn.Linear(8, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 分支输出
        out1 = self.branch1(x).view(x.size(0), -1)
        out2 = self.branch2(x).view(x.size(0), -1)
        out3 = self.branch3(x).view(x.size(0), -1)
        
        # 合并分支
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.concat_norm(out)
        out = self.dropout(out)
        
        # 全连接层和分类
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
