# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:07:14 2025

@author: 18307
"""

import numpy as np

from utils import load_cmdata2d
from utils import draw_projection

feature = 'PCC'

# 初始化存储结果的列表
cmdata_averages_dict = []

# 用于累积频段的所有数据
all_alpha_values = []
all_beta_values = []
all_gamma_values = []

# 遍历 subject 和 experiment
for subject in range(1, 3):  # 假设 subjects 是整数
    for experiment in range(1, 4):  # 假设 experiments 是整数
        print(f'sub: {subject} ex: {experiment}')
        try:
            # 加载数据
            cmdata_alpha = load_cmdata2d(feature, 'alpha', f'sub{subject}ex{experiment}')
            cmdata_beta = load_cmdata2d(feature, 'beta', f'sub{subject}ex{experiment}')
            cmdata_gamma = load_cmdata2d(feature, 'gamma', f'sub{subject}ex{experiment}')
            
            # 计算平均值
            cmdata_alpha_averaged = np.mean(cmdata_alpha, axis=0)
            cmdata_beta_averaged = np.mean(cmdata_beta, axis=0)
            cmdata_gamma_averaged = np.mean(cmdata_gamma, axis=0)
            
            # 累积数据
            all_alpha_values.append(cmdata_alpha_averaged)
            all_beta_values.append(cmdata_beta_averaged)
            all_gamma_values.append(cmdata_gamma_averaged)
            
            # # 可视化
            # draw_projection(cmdata_alpha_averaged)
            # draw_projection(cmdata_beta_averaged)
            # draw_projection(cmdata_gamma_averaged)
            
            # 合并同 subject 同 experiment 的数据
            cmdata_averages_dict.append({
                "subject": subject,
                "experiment": experiment,
                "averages": {
                    "alpha": cmdata_alpha_averaged,
                    "beta": cmdata_beta_averaged,
                    "gamma": cmdata_gamma_averaged
                }
            })
        except Exception as e:
            print(f"Error processing sub {subject} ex {experiment}: {e}")

# 计算整个数据集的全局平均值
global_alpha_average = np.mean(all_alpha_values, axis=0)
global_beta_average = np.mean(all_beta_values, axis=0)
global_gamma_average = np.mean(all_gamma_values, axis=0)

# 输出结果
draw_projection(global_alpha_average)
draw_projection(global_beta_average)
draw_projection(global_gamma_average)

