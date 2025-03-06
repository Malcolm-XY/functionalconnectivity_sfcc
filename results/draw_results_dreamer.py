# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:40:09 2025

@author: 18307
"""
import numpy as np
import matplotlib.pyplot as plt

# 数据集和实验结果
datasets = ["DREAMER PCC", "DREAMER PLV"]
metrics = ["Arousal", "Dominance", "Valence"]
methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]

# PCC 数据
pcc_data = [
    [0.7863, 0.7600, 0.7198, 0.7563],  # Arousal
    [0.7121, 0.6836, 0.6157, 0.6967],  # Dominance
    [0.8261, 0.8107, 0.7567, 0.7819]   # Valence
]

# PLV 数据
plv_data = [
    [0.8019, 0.7569, 0.6289, 0.6719],  # Arousal
    [0.7077, 0.6663, 0.5828, 0.6731],  # Dominance
    [0.8168, 0.7890, 0.7087, 0.7535]   # Valence
]

# 颜色方案
colors = {
    "SFCC": "#1f77b4",   # 蓝色，突出显示本研究方法
    "CM": "#7f7f7f",     # 灰色，传统方法
    "MCSR-CM": "#ff7f0e", # 橙色，竞争方法1
    "VCSR-CM": "#d62728"  # 红色，竞争方法2
}

# 绘制柱形图，将同一方法的三个维度数据放在一起
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 两张图分别表示 PCC 和 PLV
bar_width = 0.2
x = np.arange(len(metrics))  # 3个维度

for i, (dataset, data) in enumerate(zip(datasets, [pcc_data, plv_data])):
    ax = axes[i]
    for j, method in enumerate(methods):
        method_data = [data[k][j] for k in range(len(metrics))]  # 取出该方法在三个维度上的数据
        ax.bar(x + j * bar_width, method_data, width=bar_width, label=method, color=colors[method])

    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.5, 0.9)
    ax.set_title(f"Accuracy Comparison Across Methods on {dataset}")
    ax.set_ylabel("Accuracy")
    ax.legend()

plt.tight_layout()
plt.show()
