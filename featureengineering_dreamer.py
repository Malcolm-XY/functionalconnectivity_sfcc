# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:25:38 2025

@author: usouu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import hilbert

import mne
from mne_connectivity import spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle

# %% Filter EEG
def filter_eeg(eeg, freq=128, verbose=False):
    info = mne.create_info(ch_names=['Ch' + str(i) for i in range(eeg.shape[0])], sfreq=freq, ch_types='eeg')
    mneeeg = mne.io.RawArray(eeg, info)
    
    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 63),
    }
    
    band_filtered_eeg = {}

    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_eeg = mneeeg.copy().filter(l_freq=low_freq, h_freq=high_freq, method="fir", phase="zero-double")
        band_filtered_eeg[band] = filtered_eeg
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")

    return band_filtered_eeg

def filter_eeg_and_save(subject, verbose=True):
    _, eeg, _ = utils_common.load_dreamer()
    eeg = eeg[subject].transpose()
    
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'Filtered_EEG')
    os.makedirs(path_folder, exist_ok=True)
    
    # 调用 filter_eeg 函数
    filtered_eeg_dict = filter_eeg(eeg, verbose=verbose)
    
    # 保存每个频段的数据
    for band, filtered_eeg in filtered_eeg_dict.items():
        path_file = os.path.join(path_folder, f"sub{subject+1}_{band}_eeg.fif")
        filtered_eeg.save(path_file, overwrite=True)
        if verbose:
            print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict
    
def filter_eeg_and_save_circle(_range=range(0,23)):
    for subhject in _range:
        print(subhject)
        filter_eeg_and_save(subhject)
            
def read_filtered_eegdata(identifier, freq_band="Joint"):
    """
    Read filtered EEG data for the specified experiment and frequency band.

    Parameters:
    experiment (str): Name of the experiment (e.g., subject or session).
    freq_band (str): Frequency band to load ("alpha", "beta", "gamma", "delta", "theta", or "joint").
                     Default is "Joint".

    Returns:
    mne.io.Raw | dict: Returns the MNE Raw object for a single band or a dictionary of Raw objects for "joint".

    Raises:
    ValueError: If the specified frequency band is not valid.
    FileNotFoundError: If the expected file does not exist.
    """
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'Filtered_EEG')

    try:
        if freq_band in ["alpha", "beta", "gamma", "delta", "theta"]:
            path_file = os.path.join(path_folder, f"{identifier}_{freq_band.capitalize()}_eeg.fif")
            filtered_eeg = mne.io.read_raw_fif(path_file, preload=True)
            return filtered_eeg

        elif freq_band.lower() == "joint":
            filtered_eeg = {}
            for band in ["Alpha", "Beta", "Gamma", "Delta", "Theta"]:
                path_file = os.path.join(path_folder, f"{identifier}_{band}_eeg.fif")
                filtered_eeg[band.lower()] = mne.io.read_raw_fif(path_file, preload=True)
            return filtered_eeg

        else:
            raise ValueError(f"Invalid frequency band: {freq_band}. Choose from 'alpha', 'beta', 'gamma', 'delta', 'theta', or 'joint'.")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found for '{identifier}' and frequency band '{freq_band}'. Check the path and file existence.")

# %% Feature Engineering
def load_cms_dreamer(experiment, feature='PCC', band='joint', imshow=True):
    # 获取当前路径及父路径
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    
    # 根据方法选择对应文件夹
    if feature == 'PCC':
        path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'functional connectivity', 'PCC')
    elif feature == 'PLV':
        path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'functional connectivity', 'PLV')
    else:
        raise ValueError(f"Unsupported feature: {feature}")
    
    # 拼接数据文件路径
    path_file = os.path.join(path_folder, f'{experiment}.npy')
    
    # 加载数据
    cms_load = np.load(path_file, allow_pickle=True).item()
    
    # 从加载的数据中获取各频段的列表（列表中每个元素为形状 wxw 的数组）
    cms_alpha = cms_load["alpha"]
    cms_beta = cms_load["beta"]
    cms_gamma = cms_load["gamma"]
    
    # 将列表转换为 NumPy 数组，形状为 (n_samples, w, w)
    cms_alpha = np.array(cms_alpha)
    cms_beta = np.array(cms_beta)
    cms_gamma = np.array(cms_gamma)     
        
    # 根据 freq_band 参数返回相应的数据
    if band == "alpha":
        if imshow: utils_common.draw_projection(np.mean(cms_alpha, axis=0))
        return cms_alpha
    elif band == "beta":
        if imshow: utils_common.draw_projection(np.mean(cms_beta, axis=0))
        return cms_beta
    elif band == "gamma":
        if imshow: utils_common.draw_projection(np.mean(cms_gamma, axis=0))
        return cms_gamma
    elif band == "joint":
        joint = np.stack([cms_alpha, cms_beta, cms_gamma], axis=1)
        if imshow: utils_common.draw_projection(np.mean(joint, axis=0))
        return joint
    else:
        raise ValueError(f"Unknown freq_band parameter: {band}")

def compute_cms_and_save_circle(method, _range=range(0,23)):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    if method == "pcc":
        path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'functional connectivity', 'PCC')
    elif method == "plv":
        path_folder = os.path.join(path_parent, 'data', 'DREAMER', 'functional connectivity', 'PLV')
    
    for subject in _range:
        cms = compute_synchronization(subject + 1, method=method, freq_band="joint")
        
        path_file = os.path.join(path_folder, f"sub{subject+1}.npy")
        np.save(path_file, np.array(cms))

def compute_synchronization(subject, method, freq_band="joint", 
                                     samplingrate=128, window=1, overlap=0, 
                                     verbose=True, visualization=True):
    """
    Compute temporal synchronization metrics (PCC or PLV) for EEG data.
    
    Parameters:
        subject: Subject data identifier.
        method (str): Synchronization method, "pcc" for Pearson correlation, "plv" for Phase Locking Value.
        freq_band (str): Frequency band to analyze.
        samplingrate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays computed matrices.
    
    Returns:
        dict: Dictionary containing computed matrices for different frequency bands (if applicable).
    """
    filtered_eeg = read_filtered_eegdata(subject, freq_band=freq_band)
    
    compute_function = compute_corr_matrices if method == "pcc" else compute_plv_matrices
    
    if freq_band == "joint":
        eeg_alpha = filtered_eeg["alpha"].get_data()
        eeg_beta = filtered_eeg["beta"].get_data()
        eeg_gamma = filtered_eeg["gamma"].get_data()
        
        cms_alpha = compute_function(eeg_alpha, samplingrate, window=window, overlap=overlap, verbose=verbose)
        cms_beta = compute_function(eeg_beta, samplingrate, window=window, overlap=overlap, verbose=verbose)
        cms_gamma = compute_function(eeg_gamma, samplingrate, window=window, overlap=overlap, verbose=verbose)
        
        cms = {
            "alpha": cms_alpha,
            "beta": cms_beta,
            "gamma": cms_gamma
        }
    else:
        eeg = filtered_eeg.get_data()
        cms = compute_function(eeg, samplingrate, window=window, overlap=overlap, verbose=verbose)
    
    return cms

def compute_corr_matrices(eeg_data, samplingrate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute correlation matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        samplingrate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays correlation matrices.
    
    Returns:
        list of numpy.ndarray: List of correlation matrices for each window.
    """
    # Compute step size based on overlap
    step = int(samplingrate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(samplingrate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length] 
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    # Compute correlation matrices
    corr_matrices = []
    for idx, segment in enumerate(split_segments):
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments
        
        # Compute Pearson correlation
        corr_matrix = np.corrcoef(segment)
        corr_matrices.append(corr_matrix)

        if verbose:
            print(f"Computed correlation matrix {idx + 1}/{len(split_segments)}")

    # Optional: Visualization of correlation matrices
    if visualization and corr_matrices:
        avg_corr_matrix = np.mean(corr_matrices, axis=0)
        utils_common.draw_projection(avg_corr_matrix)

    return corr_matrices

def compute_plv_matrices(eeg_data, samplingrate, window=1, overlap=0, verbose=True, visualization=True):
    """
    Compute Phase Locking Value (PLV) matrices for EEG data using a sliding window approach.
    
    Parameters:
        eeg_data (numpy.ndarray): EEG data with shape (channels, time_samples).
        samplingrate (int): Sampling rate of the EEG data in Hz.
        window (float): Window size in seconds for segmenting EEG data.
        overlap (float): Overlap fraction between consecutive windows (0 to 1).
        verbose (bool): If True, prints progress.
        visualization (bool): If True, displays PLV matrices.
    
    Returns:
        list of numpy.ndarray: List of PLV matrices for each window.
    """
    step = int(samplingrate * window * (1 - overlap))  # Step size for moving window
    segment_length = int(samplingrate * window)

    # Split EEG data into overlapping windows
    split_segments = [
        eeg_data[:, i:i + segment_length] 
        for i in range(0, eeg_data.shape[1] - segment_length + 1, step)
    ]

    plv_matrices = []
    for idx, segment in enumerate(split_segments):
        if segment.shape[1] < segment_length:
            continue  # Skip incomplete segments
        
        # Compute Hilbert transform to obtain instantaneous phase
        analytic_signal = hilbert(segment, axis=1)
        phase_data = np.angle(analytic_signal)  # Extract phase information
        
        # Compute PLV matrix
        num_channels = phase_data.shape[0]
        plv_matrix = np.zeros((num_channels, num_channels))
        
        for ch1 in range(num_channels):
            for ch2 in range(num_channels):
                phase_diff = phase_data[ch1, :] - phase_data[ch2, :]
                plv_matrix[ch1, ch2] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        plv_matrices.append(plv_matrix)

        if verbose:
            print(f"Computed PLV matrix {idx + 1}/{len(split_segments)}")
    
    # Optional visualization
    if visualization and plv_matrices:
        avg_plv_matrix = np.mean(plv_matrices, axis=0)
        utils_common.draw_projection(avg_plv_matrix)
    
    return plv_matrices

# %% Label Engineering
def read_labels_dreamer():
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_labels = os.path.join(path_parent, 'data', 'DREAMER', 'labels', 'labels.txt')
    df = pd.read_csv(path_labels, sep=r'\s+', engine='python')
    return {col: df[col].to_numpy() for col in df.columns}

def generate_labels(samplingrate=128):
    path_current = os.getcwd()
    path_parent = os.path.dirname(path_current)
    path_data = os.path.join(path_parent, 'data', 'DREAMER', 'DREAMER.mat')

    mat_data = utils_common.read_mat(path_data)
    
    # %% labels
    score_arousal = 0
    score_dominance = 0
    score_valence = 0
    index = 0
    eeg_all = []
    for data in mat_data['DREAMER']['Data']:
        index += 1
        score_arousal += data['ScoreArousal']
        score_dominance += data['ScoreDominance']
        score_valence += data['ScoreValence']
        eeg_all.append(data['EEG']['stimuli'])
        
    labels = [1, 3, 5]
    score_arousal_labels = normalize_to_labels(score_arousal, labels)
    score_dominance_labels = normalize_to_labels(score_dominance, labels)
    score_valence_labels = normalize_to_labels(score_valence, labels)
    
    # %% data
    eeg_sample = eeg_all[0]
    labels_arousal = []
    labels_dominance = []
    labels_valence = []
    for eeg_trial in range(0,len(eeg_sample)):     
        label_container = np.ones(len(eeg_sample[eeg_trial]))
        
        label_arousal = label_container * score_arousal_labels[eeg_trial]
        label_dominance = label_container * score_dominance_labels[eeg_trial]
        label_valence = label_container * score_valence_labels[eeg_trial]
        
        labels_arousal = np.concatenate((labels_arousal, label_arousal))
        labels_dominance = np.concatenate((labels_dominance, label_dominance))
        labels_valence = np.concatenate((labels_valence, label_valence))
        
    labels_arousal = labels_arousal[::samplingrate]
    labels_dominance = labels_dominance[::samplingrate]
    labels_valence = labels_valence[::samplingrate]
    
    return labels_arousal, labels_dominance, labels_valence

def normalize_to_labels(array, labels):
    """
    Normalize an array to discrete labels.
    
    Parameters:
        array (np.ndarray): The input array.
        labels (list): The target labels to map to (e.g., [1, 3, 5]).
    
    Returns:
        np.ndarray: The normalized array mapped to discrete labels.
    """
    # Step 1: Normalize array to [0, 1]
    array_min = np.min(array)
    array_max = np.max(array)
    normalized = (array - array_min) / (array_max - array_min)
    
    # Step 2: Map to discrete labels
    bins = np.linspace(0, 1, len(labels))
    discrete_labels = np.digitize(normalized, bins, right=True)
    
    # Map indices to corresponding labels
    return np.array([labels[i - 1] for i in discrete_labels])

# %% Spectral Connectivity; Not Compeleted
def compute_spectral_connectivity(subject, experiment, method, freq_band, 
                                  window=1, overlap=0, freq_density=1, verbose=True):
    # %% asign eeg
    try:
        _, eeg, _ = utils_common.load_dreamer()
        eeg = eeg[subject]
    except FileNotFoundError as e:
        raise ValueError(f"Experiment file not found: {experiment}") from e
    
    epochs = mne.make_fixed_length_epochs(eeg, duration=window, overlap=0)
    
    # parameters
    fmin, fmax = 2, 50
    freqs = np.linspace(fmin, fmax, int((fmax-fmin)/freq_density))
    
    if freq_band:
        freq_band_map = {
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 50),
            "theta": (2, 4),
            "delta": (4, 8),
        }
        if freq_band in freq_band_map:
            fmin, fmax = freq_band_map[freq_band]
        else:
            raise ValueError(f"Invalid freq_band '{freq_band}'. Choose from {list(freq_band_map.keys())}.")
  
    # %% compute spectral connectivity
    con = spectral_connectivity_time(epochs, freqs=freqs, method=method,
        fmin=fmin, fmax=fmax, faverage=True, verbose=True)
    
    conn_matrix = con.get_data()
    conn_matrix = np.mean(conn_matrix, axis=(2))
    n_channels = int(np.sqrt(conn_matrix.shape[1]))
    conn_matrix = conn_matrix.reshape((-1, n_channels, n_channels))
    
    # %% visualization; matplot
    if verbose:
        # get labels
        labels = eeg.ch_names
        
        # get conn matrix
        conn_matrix_avg = np.mean(conn_matrix, axis=(0))
        
        # plot heatmap
        plt.figure(figsize=(18, 15))
        plt.imshow(conn_matrix_avg, cmap='viridis', origin='lower')
        plt.colorbar(label='Connectivity Strength')
        plt.title('Functional Connectivity Matrix')
        plt.xlabel('Channels')
        plt.ylabel('Channels')

        plt.xticks(range(n_channels), labels, rotation=90)
        plt.yticks(range(n_channels), labels)

        plt.show()
        
        # %% visualization; mne
        plot_connectivity_circle(conn_matrix_avg, node_names=labels, 
                                 n_lines=62, title= 'Top Functional Connections',
                                 facecolor='white', textcolor='black')

    return conn_matrix

# %% interpolation
import scipy.interpolate
def interpolate_matrices(data, scale_factor=(1.0, 1.0), method='nearest'):
    """
    对形如 samples x channels x w x h 的数据进行插值，使每个 w x h 矩阵放缩

    参数:
    - data: numpy.ndarray, 形状为 (samples, channels, w, h)
    - scale_factor: float 或 (float, float)，插值的缩放因子
    - method: str，插值方法，可选：
        - 'nearest' (最近邻)
        - 'linear' (双线性)
        - 'cubic' (三次插值)

    返回:
    - new_data: numpy.ndarray, 插值后的数据，形状 (samples, channels, new_w, new_h)
    """
    samples, channels, w, h = data.shape
    new_w, new_h = int(w * scale_factor[0]), int(h * scale_factor[1])

    # 目标尺寸
    output_shape = (samples, channels, new_w, new_h)
    new_data = np.zeros(output_shape, dtype=data.dtype)

    # 原始网格点 (w, h)
    x_old = np.linspace(0, 1, w)
    y_old = np.linspace(0, 1, h)
    xx_old, yy_old = np.meshgrid(x_old, y_old, indexing='ij')

    # 目标网格点 (new_w, new_h)
    x_new = np.linspace(0, 1, new_w)
    y_new = np.linspace(0, 1, new_h)
    xx_new, yy_new = np.meshgrid(x_new, y_new, indexing='ij')

    # 插值
    for i in range(samples):
        for j in range(channels):
            old_points = np.column_stack([xx_old.ravel(), yy_old.ravel()])  # 原始点坐标
            new_points = np.column_stack([xx_new.ravel(), yy_new.ravel()])  # 目标点坐标
            values = data[i, j].ravel()  # 原始像素值

            # griddata 进行插值
            interpolated = scipy.interpolate.griddata(old_points, values, new_points, method=method)
            new_data[i, j] = interpolated.reshape(new_w, new_h)

    return new_data

# %% usage
if __name__ == "__main__":
    import utils_common
    
    # %% Read and Filter Sample EEG
    # dreamer, eeg, electrodes = utils_common.load_dreamer()
    # eeg_sample = eeg[0].transpose()
    
    # # filter
    # calculated_filtered_eeg_dict = filter_eeg(eeg_sample, verbose=True)
    
    # # filter and save
    # # filter_eeg_and_save_circle()
    
    # # read filtered
    # loaded_filtered_eeg = read_filtered_eegdata('sub1', freq_band="joint")
    
    # %% Feature Engineeing; Compute Connectivity Matrices
    # correlation matrices
    # cms_pcc = compute_synchronization(subject_sample, method="PCC", freq_band="joint")
    # cms_plv = compute_synchronization(subject_sample, method="PLV", freq_band="joint")
    
    # compute correlation matrices and save
    # compute_cms_and_save_circle(method="pcc")
    # compute_cms_and_save_circle(method="plv")
    
    # %% Load Saved Connectivity Matrices
    subject_sample = 'sub1'
    
    cms_pcc = load_cms_dreamer(experiment=subject_sample, feature='PCC', band='joint')
    cms_plv = load_cms_dreamer(experiment=subject_sample, feature='PLV', band='joint')
    
    cms_pcc_2 = utils_common.load_cms(dataset='DREAMER', experiment='sub1', feature='PCC', band='joint')
    cms_pcc_2 = utils_common.load_cms(dataset='DREAMER', experiment='sub1', feature='PCC', band='joint')