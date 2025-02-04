# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:53:48 2025

@author: 18307
"""

import os
import pandas
import numpy as np

# %% rearrangement by using orders
def generate_rearrangedcm(cm_data, method, order="SEED", padding=True, imshow=True):
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
    rearranged_indices = get_rearrangedindex(method, order=order)
    sorted_matrix = reshape_and_sort(cm_data, rearranged_indices)

    if padding:
        sorted_matrix = global_padding(sorted_matrix)
        
    if imshow:
        utils.draw_projection(np.mean(sorted_matrix, axis=0))

    return sorted_matrix
    
def get_rearrangedindex(method, order="SEED"):
    path_current = os.getcwd()
    
    if method == 'MX':
        if order=="SEED": 
            path_txt = os.path.join(path_current, 'rearrangement', 'MXindex.txt')
        elif order == "DREAMER":
            path_txt = os.path.join(path_current, 'rearrangement', 'MXindex_dreamer.txt')
    elif method =='VC':
        if order == "SEED":
            path_txt = os.path.join(path_current, 'rearrangement', 'VCindex.txt')
        elif order == "DREAMER":
            path_txt = os.path.join(path_current, 'rearrangement', 'VCindex_dreamer.txt')
    else: raise ValueError('Invalid rearranged method!')

    if not os.path.exists(path_txt):
        raise FileNotFoundError(f"Index file not found: {path_txt}")

    index = pandas.read_csv(path_txt, sep='\t', header=None).to_numpy().flatten()
    return index

def reshape_and_sort(matrix, index):
    rearranged_matrix = matrix[:, :, index, :]
    rearranged_matrix = rearranged_matrix[:, :, :, index]
    return rearranged_matrix

# %% compute order of rearrangement
def compute_sorted_global_avg(num_subjects=15, num_experiments=3, single=False):
    """
    Computes and sorts the global average of channel-wise connectivity data 
    across subjects and experiments for different frequency bands (gamma, beta, alpha).
    """
    # num_subjects = 15  # Total number of subjects
    # num_experiments = 3  # Total number of experiments per subject
    
    if single:
        start_subject = num_subjects
    else: start_subject = 1
    
    joint_averages = []  # Stores the joint average connectivity data for each experiment
    
    # Iterate through each subject and experiment
    for subject_id in range(start_subject, num_subjects + 1):
        for experiment_id in range(1, num_experiments + 1):
            # Load connectivity data for different frequency bands
            gamma_data = utils.load_cmdata2d('PCC', 'gamma', f'sub{subject_id}ex{experiment_id}')
            beta_data = utils.load_cmdata2d('PCC', 'beta', f'sub{subject_id}ex{experiment_id}')
            alpha_data = utils.load_cmdata2d('PCC', 'alpha', f'sub{subject_id}ex{experiment_id}')
            
            # Compute the mean connectivity for each frequency band
            gamma_avg = np.mean(gamma_data, axis=0)
            beta_avg = np.mean(beta_data, axis=0)
            alpha_avg = np.mean(alpha_data, axis=0)
            
            # Compute the joint average of all frequency bands
            joint_avg = (gamma_avg + beta_avg + alpha_avg) / 3
            joint_averages.append(joint_avg)
    
    # Compute the global average across all subjects and experiments
    global_joint_avg = np.mean(joint_averages, axis=0)
    
    # Visualize the projection of the global average
    utils.draw_projection(global_joint_avg)
    
    # Compute the global average across channels and sort the values
    global_channel_avg = np.mean(global_joint_avg, axis=0)
    sorted_order = np.argsort(global_channel_avg)[::-1]

    return sorted_order

# %% tools; padding
def global_padding(matrix, width=81, verbose=True):
    """
    Pads a 2D or 4D matrix to the specified width while keeping the original data centered.

    Parameters:
        matrix (np.ndarray): The input matrix to be padded.
        width (int): The target width/height for padding.
        verbose (bool): If True, prints the original and padded shapes.

    Returns:
        np.ndarray: The padded matrix with the specified width.
    """
    if len(matrix.shape) == 2:
        width_input, _ = matrix.shape
        total_padding = width - width_input
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before

        padded_matrix = np.pad(
            matrix,
            pad_width=((pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )

    elif len(matrix.shape) == 4:
        _, _, width_input, _ = matrix.shape
        total_padding = width - width_input
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before

        padded_matrix = np.pad(
            matrix,
            pad_width=((0, 0), (0, 0), (pad_before, pad_after), (pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )

    else:
        raise ValueError("Input matrix must be either 2D or 4D.")

    if verbose:
        print("Original shape:", matrix.shape)
        print("Padded shape:", padded_matrix.shape)

    return padded_matrix

# %% usage
if __name__ == '__main__':
    import utils
    import featureengineering_dreamer
    
    # compute MX order
    mx_index = compute_sorted_global_avg(num_subjects=15)
    
    # SEED
    feature_sample, freq_sample, experiment_sample = 'PCC', 'joint',  'sub1ex1'
    cm_data = utils.load_cmdata2d(feature_sample, freq_sample, experiment_sample)

    rearranged_MX_cm = generate_rearrangedcm(cm_data, 'MX', imshow=True)
    rearranged_VC_cm = generate_rearrangedcm(cm_data, 'VC', imshow=True)
    
    # DREAMER
    cm_data_d = featureengineering_dreamer.read_cms(1, feature="PCC", freq_band="joint", imshow=True)
    rearranged_MX_cm_d = generate_rearrangedcm(cm_data_d, 'MX', order="DREAMER", padding=True, imshow = True)