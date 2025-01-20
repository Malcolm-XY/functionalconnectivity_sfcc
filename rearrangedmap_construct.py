# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:53:48 2025

@author: 18307
"""

import os
import pandas
import numpy as np

import utils

def generate_rearrangedcm(cm_data, method, imshow=False):
    """
    Generate a rearranged confusion matrix based on the specified method.

    Parameters:
        cm_data (array-like): The original confusion matrix data.
        method (str): The rearrangement method ('MX' or 'VC').
        imshow (bool): If True, display the rearranged matrix as a projection.

    Returns:
        sorted_matrix (array-like): The rearranged confusion matrix.

    Raises:
        ValueError: If an invalid rearrangement method is specified.
    """
    if method == 'MX':
        # Implement logic for 'MX' rearrangement
        rearranged_indices = get_rearrangedindex(method)
        sorted_matrix = reshape_and_sort(cm_data, rearranged_indices)
    elif method == 'VC':
        # Implement logic for 'VC' rearrangement
        rearranged_indices = get_rearrangedindex(method)
        sorted_matrix = reshape_and_sort(cm_data, rearranged_indices)
    else:
        raise ValueError(f"Invalid rearranged method: {method}")

    if imshow:
        utils.draw_projection(sorted_matrix[0])

    return sorted_matrix
    
def get_rearrangedindex(method):
    path_current = os.getcwd()
    
    if method == 'MX':
        path_txt = os.path.join(path_current, 'rearrangement', 'MXindex.txt')
    elif method =='VC':
        path_txt = os.path.join(path_current, 'rearrangement', 'VCindex.txt')
    else: raise ValueError('Invalid rearranged method!')
    
    index = read_rearrangedindex(path_txt) -1
    return index

def read_rearrangedindex(path_txt):
    index = pandas.read_csv(path_txt, sep='\t', header=None).to_numpy().flatten()
    return index

def reshape_and_sort(matrix, custom_index):
    """
    将输入矩阵从 samples x channels x n x n 转换为 samples x channels x n^2，
    使用自定义索引对 n^2 维度重新排序后恢复到 samples x channels x n x n。

    参数:
        matrix: numpy.ndarray, 形状为 (samples, channels, n, n)
        custom_index: numpy.ndarray, 自定义排序索引，形状为 (n^2,)

    返回:
        reshaped_matrix: numpy.ndarray, 重新排序后的矩阵，形状为 (samples, channels, n, n)
    """
    samples, channels, n, _ = matrix.shape
    n_squared = n * n

    # 检查自定义索引的合法性
    if custom_index.shape[0] != n_squared:
        raise ValueError("自定义索引的长度必须为 n^2")
    if set(custom_index) != set(range(n_squared)):
        raise ValueError("自定义索引必须包含 0 到 n^2-1 的所有值")

    # Step 1: 将矩阵展平到 samples x channels x n^2
    flattened = matrix.reshape(samples, channels, n_squared)

    # Step 2: 使用自定义索引对 n^2 维度排序
    sorted_flattened = flattened[:, :, custom_index]

    # Step 3: 恢复到原始形状 samples x channels x n x n
    reshaped_matrix = sorted_flattened.reshape(samples, channels, n, n)

    return reshaped_matrix

if __name__ == '__main__':
    cm_data = utils.load_cmdata2d('PCC', 'joint', 'sub1ex1')
    utils.draw_projection(cm_data[0])

    rearranged_MX_cm = generate_rearrangedcm(cm_data, 'MX', imshow=True)
    rearranged_VC_cm = generate_rearrangedcm(cm_data, 'VC', imshow=True)