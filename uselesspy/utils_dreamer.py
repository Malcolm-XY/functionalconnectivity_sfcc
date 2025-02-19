# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan 13 15:49:11 2025

# @author: 18307
# """

# import os
# import h5py

# import numpy as np
# import pandas as pd

# import scipy.io
# import scipy.ndimage

# # %% read mat
# def read_dreamer():
#     path_current = os.getcwd()
#     path_parent = os.path.dirname(path_current)
#     path_data = os.path.join(path_parent, 'data', 'DREAMER', 'DREAMER.mat')

#     # get mat
#     dreamer = read_mat(path_data)
#     dreamer = dreamer["DREAMER"]
#     dreamer_data = dreamer["Data"]
    
#     # get eeg
#     eeg_dic = []
#     for trial_data in dreamer_data:
#         eeg_list = trial_data["EEG"]["stimuli"]
        
#         eeg_flattened = np.vstack(eeg_list)
    
#         eeg_dic.append(eeg_flattened)
    
#     # get electrodes
#     electrode_list = dreamer["EEG_Electrodes"]
    
#     return dreamer, eeg_dic, electrode_list

# def read_mat(path_file, simplify=True):
#     """
#     读取 MATLAB 的 .mat 文件。
#     - 自动支持 HDF5 格式和非 HDF5 格式。
#     - 可选简化数据结构。
    
#     参数：
#     - path_file (str): .mat 文件路径。
#     - simplify (bool): 是否简化数据结构（默认 True）。
    
#     返回：
#     - dict: 包含 .mat 文件数据的字典。
#     """
#     # 确保文件存在
#     if not os.path.exists(path_file):
#         raise FileNotFoundError(f"File not found: {path_file}")

#     try:
#         # 尝试以 HDF5 格式读取文件
#         with h5py.File(path_file, 'r') as f:
#             print("HDF5 format detected.")
#             data = {key: simplify_mat_structure(f[key]) for key in f.keys()} if simplify else f
#             return data

#     except OSError:
#         # 如果不是 HDF5 格式，尝试使用 scipy.io.loadmat
#         print("Not an HDF5 format.")
#         mat_data = scipy.io.loadmat(path_file, squeeze_me=simplify, struct_as_record=not simplify)
#         if simplify:
#             mat_data = {key: simplify_mat_structure(value) for key, value in mat_data.items() if key[0] != '_'}
#         return mat_data

# def simplify_mat_structure(data):
#     """
#     递归解析和简化 MATLAB 数据结构。
#     - 将结构体转换为字典。
#     - 将 Cell 数组转换为列表。
#     - 移除 NumPy 数组中的多余维度。
#     """
#     if isinstance(data, h5py.Dataset):  # 处理 HDF5 数据集
#         return data[()]  # 转换为 NumPy 数组或标量

#     elif isinstance(data, h5py.Group):  # 处理 HDF5 文件组
#         return {key: simplify_mat_structure(data[key]) for key in data.keys()}

#     elif isinstance(data, scipy.io.matlab.mat_struct):  # 处理 MATLAB 结构体
#         return {field: simplify_mat_structure(getattr(data, field)) for field in data._fieldnames}

#     elif isinstance(data, np.ndarray):  # 处理 NumPy 数组
#         if data.dtype == 'object':  # 递归解析对象数组
#             return [simplify_mat_structure(item) for item in data]
#         return data.squeeze()  # 移除多余维度

#     else:  # 其他类型直接返回
#         return data

# # %% labels
# def read_labels_dreamer():
#     # 获取当前路径的父目录
#     path_current = os.getcwd()
#     path_parent = os.path.dirname(path_current)

#     # 构建 labels.txt 的路径
#     path_labels = os.path.join(path_parent, 'data', 'DREAMER', 'labels', 'labels.txt')

#     # 读取数据，假设是以空格或 Tab 分隔
#     df = pd.read_csv(path_labels, sep=r'\s+', engine='python')  # 自动识别空格/TAB 分隔

#     # 转换为字典，每个 key 关联到一个 NumPy 数组
#     labels_dict = {col: df[col].to_numpy() for col in df.columns}

#     print('Labels Reading Done')

#     return labels_dict  # 返回 {'arousal': np.array([...]), 'dominance': np.array([...]), 'valence': np.array([...])}

# def generate_labels(samplingrate=128):
#     path_current = os.getcwd()
#     path_parent = os.path.dirname(path_current)
#     path_data = os.path.join(path_parent, 'data', 'DREAMER', 'DREAMER.mat')

#     mat_data = read_mat(path_data)
    
#     # %% labels
#     score_arousal = 0
#     score_dominance = 0
#     score_valence = 0
#     index = 0
#     eeg_all = []
#     for data in mat_data['DREAMER']['Data']:
#         index += 1
#         score_arousal += data['ScoreArousal']
#         score_dominance += data['ScoreDominance']
#         score_valence += data['ScoreValence']
#         eeg_all.append(data['EEG']['stimuli'])
        
#     labels = [1, 3, 5]
#     score_arousal_labels = normalize_to_labels(score_arousal, labels)
#     score_dominance_labels = normalize_to_labels(score_dominance, labels)
#     score_valence_labels = normalize_to_labels(score_valence, labels)
    
#     # %% data
#     eeg_sample = eeg_all[0]
#     labels_arousal = []
#     labels_dominance = []
#     labels_valence = []
#     for eeg_trial in range(0,len(eeg_sample)):     
#         label_container = np.ones(len(eeg_sample[eeg_trial]))
        
#         label_arousal = label_container * score_arousal_labels[eeg_trial]
#         label_dominance = label_container * score_dominance_labels[eeg_trial]
#         label_valence = label_container * score_valence_labels[eeg_trial]
        
#         labels_arousal = np.concatenate((labels_arousal, label_arousal))
#         labels_dominance = np.concatenate((labels_dominance, label_dominance))
#         labels_valence = np.concatenate((labels_valence, label_valence))
        
#     labels_arousal = labels_arousal[::samplingrate]
#     labels_dominance = labels_dominance[::samplingrate]
#     labels_valence = labels_valence[::samplingrate]
    
#     return labels_arousal, labels_dominance, labels_valence

# def normalize_to_labels(array, labels):
#     """
#     Normalize an array to discrete labels.
    
#     Parameters:
#         array (np.ndarray): The input array.
#         labels (list): The target labels to map to (e.g., [1, 3, 5]).
    
#     Returns:
#         np.ndarray: The normalized array mapped to discrete labels.
#     """
#     # Step 1: Normalize array to [0, 1]
#     array_min = np.min(array)
#     array_max = np.max(array)
#     normalized = (array - array_min) / (array_max - array_min)
    
#     # Step 2: Map to discrete labels
#     bins = np.linspace(0, 1, len(labels))
#     discrete_labels = np.digitize(normalized, bins, right=True)
    
#     # Map indices to corresponding labels
#     return np.array([labels[i - 1] for i in discrete_labels])

# # %% interpolation
# def interpolate_matrices(data, scale_factor):
#     """
#     对形如 samples x channels x w x h 的数据进行插值，使每个 w x h 矩阵放缩
    
#     参数:
#     - data: numpy.ndarray, 形状为 (samples, channels, w, h)
#     - scale_factor: float 或 (float, float)，插值的缩放因子
    
#     返回:
#     - new_data: numpy.ndarray, 形状不变 (samples, channels, w, h)
#     """
#     samples, channels, w, h = data.shape
#     new_w, new_h = int(w * scale_factor[0]), int(h * scale_factor[1])
    
#     # 目标尺寸
#     output_shape = (samples, channels, new_w, new_h)
#     new_data = np.zeros(output_shape, dtype=data.dtype)

#     # 对每个 w x h 矩阵进行插值
#     for i in range(samples):
#         for j in range(channels):
#             new_data[i, j] = scipy.ndimage.zoom(data[i, j], zoom=scale_factor, order=3)  # 采用三次插值
    
#     return new_data

# # %% visualization
# import matplotlib.pyplot as plt
# def draw_projection(sample_projection):
#     if sample_projection.ndim == 2:
#         # Visualize the 2D matrix
#         plt.imshow(sample_projection, cmap='viridis')  # Use 'viridis' colormap
#         plt.colorbar()  # Add a colorbar
#         plt.title("Matrix Visualization using imshow")
#         plt.xlabel("X-axis")
#         plt.ylabel("Y-axis")
#         plt.show()
#     elif sample_projection.ndim == 3 and sample_projection.shape[0] == 3:
#         # Visualize each channel of the 3D projection
#         for i, channel_projection in enumerate(sample_projection):
#             plt.imshow(channel_projection, cmap='viridis')  # Use 'viridis' colormap
#             plt.colorbar()  # Add a colorbar
#             plt.title(f"Matrix Visualization of Channel {i + 1}")
#             plt.xlabel("X-axis")
#             plt.ylabel("Y-axis")
#             plt.show()
#     else:
#         raise ValueError("Input projection sample should be a 2D array or a 3D array with 3 channels.")

# # %% usage
# if __name__ == '__main__':
#     # eeg and info from original dataset
#     dreamer, dreamer_eeg, electrode_list = read_dreamer()
    
#     # # compute labels
#     # labels_arousal, labels_dominance, labels_valence = generate_labels()
    
#     # # read labels.txt
#     # labels = read_labels_dreamer()