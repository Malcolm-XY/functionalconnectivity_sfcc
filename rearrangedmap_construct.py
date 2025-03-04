# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:53:48 2025

@author: 18307
"""

import os
from math import gamma

import pandas
import numpy as np

import utils_common
import utils_visualization

# %% rearrangement by using orders
def generate_rearranged_fcs(fcs, method, electrode_order='SEED', padding=True, imshow=True):
    """
    Generate a rearranged confusion matrix based on the specified method.

    This function dynamically detects available frequency bands instead of assuming fixed keys.

    Parameters:
        fcs (dict): Functional connectivity matrices. Shape: {band: (samples, ch, ch)}
        method (str): The rearrangement method ('MX' or 'VC').
        electrode_order (str): The dataset order ('SEED' or 'DREAMER').
        padding (bool): If True, applies global padding to the sorted fcs.
        imshow (bool): If True, displays the rearranged matrix projection.

    Returns:
        sorted_fcs (np.ndarray): The rearranged connectivity matrices. Shape: (samples, bands, ch, ch)

    Raises:
        ValueError: If an invalid rearrangement method is specified.
    """
    # Normalize parameters
    electrode_order = electrode_order.upper()

    # Read indices as the basis of rearrangement
    rearranged_indices = read_rearranged_index(method, electrode_order=electrode_order)

    # Dynamically sort all available bands
    sorted_bands = [reshape_and_sort(fcs[band], rearranged_indices) for band in fcs.keys()]

    # Stack the bands along the second axis
    sorted_fcs = np.stack(sorted_bands, axis=1)

    print(f'The shape of sorted fcs is {sorted_fcs.shape}')

    if padding:
        sorted_fcs = global_padding(sorted_fcs, 81)

    if imshow:
        import utils_visualization
        utils_visualization.draw_projection(np.mean(sorted_fcs, axis=0))

    return sorted_fcs

def read_rearranged_index(method, electrode_order='SEED'):
    # Normalize parameters
    electrode_order = electrode_order.upper()
    path_current = os.getcwd()

    path_txt = ""
    if method == 'MX':
        if electrode_order == 'SEED':
            path_txt = os.path.join(path_current, 'rearrangement', 'MXindex.txt')
        elif electrode_order == "DREAMER":
            path_txt = os.path.join(path_current, 'rearrangement', 'MXindex_dreamer.txt')
    elif method == 'VC':
        if electrode_order == 'SEED':
            path_txt = os.path.join(path_current, 'rearrangement', 'VCindex.txt')
        elif electrode_order == "DREAMER":
            path_txt = os.path.join(path_current, 'rearrangement', 'VCindex_dreamer.txt')
    else:
        raise ValueError('Invalid rearrangement method!')

    if not os.path.exists(path_txt):
        raise FileNotFoundError(f"Index file not found: {path_txt}")

    index = pandas.read_csv(path_txt, sep='\t', header=None).to_numpy().flatten()
    return index

def reshape_and_sort(matrix, index):
    """
    matrix: samples x ch x ch
    """
    rearranged_matrix = matrix[:, index, :]
    rearranged_matrix = rearranged_matrix[:, :, index]
    return rearranged_matrix

# %% compute order of rearrangement
def compute_sorted_global_avg(feature='pcc', range_subject=range(1, 2), range_experiment=range(1, 2)):
    """
    Computes and sorts the global average of channel-wise connectivity data
    across subjects and experiments for different frequency bands.

    Automatically adapts to the available frequency bands instead of assuming specific keys.

    Parameters:
        feature (str): The connectivity feature to use (default: 'pcc').
        range_subject (range): The range of subjects to include.
        range_experiment (range): The range of experiments to include.

    Returns:
        np.ndarray: Indices of channels sorted by global average connectivity in descending order.
    """

    joint_averages = []  # Stores the joint average connectivity data for each experiment

    # Iterate through each subject and experiment
    for subject_id in range_subject:
        for experiment_id in range_experiment:
            # Load connectivity data for different frequency bands
            identifier = f'sub{subject_id}ex{experiment_id}'
            fcs = utils_feature_loading.read_fcs('seed', identifier, feature)

            # Dynamically get available frequency bands
            frequency_bands = list(fcs.keys())

            # Compute the mean connectivity for each frequency band
            band_averages = [np.mean(fcs[band], axis=0) for band in frequency_bands]

            # Compute the joint average of all frequency bands
            joint_avg = np.mean(band_averages, axis=0)
            joint_averages.append(joint_avg)

    # Compute the global average across all subjects and experiments
    global_joint_avg = np.mean(joint_averages, axis=0)

    # Visualize the projection of the global average
    utils_visualization.draw_projection(global_joint_avg)

    # Compute the global average across channels and sort the values
    global_channel_avg = np.mean(global_joint_avg, axis=0)
    sorted_order = np.argsort(global_channel_avg)[::-1]

    return sorted_order

# %% tools; padding
def global_padding(matrix, width=81, verbose=True):
    """
    Pads a 2D, 3D or 4D matrix to the specified width while keeping the original data centered.
    For shape of: width x height, samples x width x height, samples x channels x width x height.

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

    elif len(matrix.shape) == 3:
        _, width_input, _ = matrix.shape
        total_padding = width - width_input
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before

        padded_matrix = np.pad(
            matrix,
            pad_width=((0, 0), (pad_before, pad_after), (pad_before, pad_after)),
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
        raise ValueError("Input matrix must be either 2D, 3D or 4D.")

    if verbose:
        print("Original shape:", matrix.shape)
        print("Padded shape:", padded_matrix.shape)

    return padded_matrix

# %% usage
if __name__ == '__main__':
    # %% Generate MX order
    mx_index = compute_sorted_global_avg(feature='pcc')
    
    # %% SEED
    # import for feature reading
    import utils_feature_loading
    fcs_sample_seed = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')

    rearranged_MX_cm = generate_rearranged_fcs(fcs_sample_seed, 'MX', imshow=True)
    rearranged_VC_cm = generate_rearranged_fcs(fcs_sample_seed, 'VC', imshow=True)
    
    # %% DREAMER
    fcs_sample_dreamer = utils_feature_loading.read_fcs('dreamer', 'sub1', 'pcc')

    rearranged_MX_cm_dreamer = generate_rearranged_fcs(fcs_sample_dreamer, 'MX', 'dreamer', imshow=True)
    rearranged_VC_cm_dreamer = generate_rearranged_fcs(fcs_sample_dreamer, 'VC', 'dreamer', imshow=True)