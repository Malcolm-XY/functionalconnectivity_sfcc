# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:24:09 2025

@author: 18307
"""

import matplotlib.pyplot as plt

def plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name, _min=0.8, _max=None):
    """
    绘制 Accuracy 和 F1 Score 的对比柱状图，并自动添加显著性标记
    
    参数：
    - methods: list[str] 方法名称
    - accuracy: list[float] 各方法的准确率
    - f1_score: list[float] 各方法的 F1 Score
    - significance_results: dict {method_name: p-value string} 显著性标记
    - feature_name: str 特征名称
    - _min: float y 轴最小值（默认 0.8）
    - _max: float or None y 轴最大值（如果为 None，则自动调整）
    """

    # 颜色方案
    color_map = {
        "SFCC": "#1f77b4",    # 红色
        "CM": "#7f7f7f",      # 灰色
        "MCSR-CM": "#ff7f0e", # 橙色
        "VCSR-CM": "#d62728"  # 蓝色
    }
    colors = [color_map[method] for method in methods]

    # 生成图像
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 计算 y 轴最大值
    acc_max = _max if _max is not None else max(accuracy) + 0.03
    f1_max = _max if _max is not None else max(f1_score) + 0.03

    # 绘制 Accuracy 柱状图
    axs[0].bar(methods, accuracy, color=colors, alpha=0.7)
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title(f"Accuracy Comparison Across Methods ({feature_name})")
    axs[0].set_ylim(_min, acc_max)  # 自动调整 y 轴

    # 绘制 F1 Score 柱状图
    axs[1].bar(methods, f1_score, color=colors, alpha=0.7)
    axs[1].set_ylabel("F1 Score")
    axs[1].set_title(f"F1 Score Comparison Across Methods ({feature_name})")
    axs[1].set_ylim(_min, f1_max)  # 自动调整 y 轴

    # 标注数值
    for ax, values in zip(axs, [accuracy, f1_score]):
        for bar, value in zip(ax.patches, values):
            ax.annotate(f'{value:.4f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3),  # 偏移量
                        textcoords="offset points",
                        ha='center', va='bottom')

    # 添加显著性横线
    def add_significance(ax, x1, x2, y, p_text):
        """ 在两个柱子间添加显著性标记 """
        ax.plot([x1, x1, x2, x2], [y, y + 0.00125, y + 0.00125, y], lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y + 0.00125, p_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 在 SFCC 和其他方法间绘制显著性
    sfcc_index = methods.index("SFCC")
    y_offset = 0.0075  # 纵向偏移，避免覆盖数据标签
    y_offset_grad = 0.0075  # 每个标记的递增偏移量

    for i, method in enumerate(methods):
        if method != "SFCC":
            offset = y_offset + (i - 1) * y_offset_grad  # 逐步增加偏移量
            add_significance(axs[0], sfcc_index, i, max(accuracy) + offset, "**")
            add_significance(axs[1], sfcc_index, i, max(f1_score) + offset, "**")

    # 修改 caption
    caption_text = f"Comparison of Accuracy and F1 Score Across Methods Using {feature_name}"
    fig.text(0.5, -0.02, caption_text, ha='center', va='top', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

# %% SEED dataset
# PCC
methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
accuracy = [0.9029, 0.8606, 0.8389, 0.8587]
f1_score = [0.8995, 0.8567, 0.8335, 0.8553]

significance_results = {
    "CM": "p<0.01",
    "MCSR-CM": "p<0.01",
    "VCSR-CM": "p<0.01"
}

plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='PCC')

# PLV
methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
accuracy = [0.9005, 0.8491, 0.8947, 0.8349]  # 直接使用 avg 数据
f1_score = [0.8969, 0.8452, 0.8905, 0.8312]

significance_results = {
    "CM": "p<0.01",
    "MCSR-CM": "p<0.01",
    "VCSR-CM": "p<0.01"
}

plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='PLV')

# %% DREAMER dataset; PCC
# # PCC; Arousal
# methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
# accuracy = [0.7863, 0.7600, 0.7198, 0.7563]
# f1_score = [0.7696, 0.7472, 0.7070, 0.7400]

# significance_results = {
#     "CM": "p<0.01",
#     "MCSR-CM": "p<0.01",
#     "VCSR-CM": "p<0.01"
# }

# plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='Feature: PCC; Dimension: Arousal', _min=0.7)

# # PCC; Dominance
# methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
# accuracy = [0.7121, 0.6836, 0.6157, 0.6967]
# f1_score = [0.7202, 0.6916, 0.6210, 0.7059]

# significance_results = {
#     "CM": "p<0.01",
#     "MCSR-CM": "p<0.01",
#     "VCSR-CM": "p<0.01"
# }

# plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='Feature: PCC; Dimension: Arousal', _min=0.6)

# # PCC; Valence
# methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
# accuracy = [0.8261, 0.8107, 0.7567, 0.7819]
# f1_score = [0.8072, 0.7908, 0.7404, 0.7652]

# significance_results = {
#     "CM": "p<0.01",
#     "MCSR-CM": "p<0.01",
#     "VCSR-CM": "p<0.01"
# }

# plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='Feature: PCC; Dimension: Arousal', _min=0.7)

# # %% DREAMER dataset; PLV
# # PLV; Arousal
# methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
# accuracy = [0.8019, 0.7569, 0.6289, 0.6719]
# f1_score = [0.7920, 0.7474, 0.6160, 0.6617]

# significance_results = {
#     "CM": "p<0.01",
#     "MCSR-CM": "p<0.01",
#     "VCSR-CM": "p<0.01"
# }

# plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='Feature: PLV; Dimension: Arousal', _min=0.6, _max=0.85)

# # PLV; Dominance
# methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
# accuracy = [0.7077, 0.6663, 0.5828, 0.6731]
# f1_score = [0.7166, 0.6747, 0.5879, 0.6805]

# significance_results = {
#     "CM": "p<0.01",
#     "MCSR-CM": "p<0.01",
#     "VCSR-CM": "p<0.01"
# }

# plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='Feature: PLV; Dimension: Dominance', _min=0.55)

# # PLV; Arousal
# methods = ["SFCC", "CM", "MCSR-CM", "VCSR-CM"]
# accuracy = [0.8168, 0.7890, 0.7087, 0.7535]
# f1_score = [0.8015, 0.7734, 0.6882, 0.7368]

# significance_results = {
#     "CM": "p<0.01",
#     "MCSR-CM": "p<0.01",
#     "VCSR-CM": "p<0.01"
# }

# plot_comparison_chart(methods, accuracy, f1_score, significance_results, feature_name='Feature: PLV; Dimension: Valence', _min=0.6)
