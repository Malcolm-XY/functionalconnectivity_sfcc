import os
import numpy
import pandas

import utils_common

# %% intergrated
def generate_sfcc(cm_data, dataset, imshow=False):
    """
    Common function to generate SFCC (Sparse Functional Connectivity Convolution) from CM data.

    Args:
        cm_data (numpy.ndarray): Input connectivity matrix data. Can be 3D (samples, size, size) 
                                 or 4D (samples, channels, size, size).
        dataset (str): The dataset name ("SEED" or "DREAMER").
        imshow (bool, optional): Whether to display the projection of the first sample. Default is False.

    Returns:
        numpy.ndarray: The generated SFCC data.
    """
    # Get the spatial map (smap) and channel order
    smap, order = read_distribution(dataset)

    # Step 1: Generate lmap and covmap
    lmap, covmap = generate_lmap_and_covmap(smap)

    # Step 2: Generate connectivity matrix and vector
    CM, CV = generate_connectivity_matrix(order)

    # Step 3: Generate numerical covmap
    covmap_num = generate_covmap_num(covmap, CV)

    if cm_data.ndim == 3:
        # Case: 3D input data (samples, size, size)
        sfcc = cm2sfcc(cm_data, covmap_num)

    elif cm_data.ndim == 4:
        # Case: 4D input data (samples, channels, size, size)
        sfcc = numpy.stack([cm2sfcc(cm_data[:, ch, :, :], covmap_num) for ch in range(cm_data.shape[1])], axis=1)

    else:
        raise ValueError("Input cm_data must be 3D or 4D.")

    if imshow:
        utils_common.draw_projection(sfcc[0])

    return sfcc

def read_distribution(dataset):
    """
    Reads and returns the sensor mapping and channel order distribution for the specified dataset.

    Parameters:
        dataset (str): The name of the dataset. Supported values: "SEED", "DREAMER".

    Returns:
        tuple: (smap (numpy.ndarray), order (numpy.ndarray))
               - smap: Sensor mapping array.
               - order: Channel order array.
    """
    if dataset not in {"SEED", "DREAMER"}:
        raise ValueError("Unsupported dataset. Choose either 'SEED' or 'DREAMER'.")

    path_current = os.getcwd()
    
    # Define file paths based on dataset
    mapping_files = {
        "SEED": ("smap.txt", "biosemi62_64_channels_original_distribution.txt"),
        "DREAMER": ("smap_dreamer.txt", "biosemi62_14_channels_original_distribution.txt")
    }

    smap_file, order_file = mapping_files[dataset]
    
    path_smap = os.path.join(path_current, "mapping", smap_file)
    path_order = os.path.join(path_current, "mapping", order_file)

    # Read files
    smap = pandas.read_csv(path_smap, sep="\t", header=None).values
    order = pandas.read_csv(path_order, sep="\t")["channel"].values

    return smap, order
   
# %% steps
def generate_lmap_and_covmap(smap):
    """
    根据原始电极图(smap)生成扩展电极图(lmap)和卷积映射(covmap)。
    
    Args:
        smap (ndarray): 原始电极图。
    
    Returns:
        lmap (ndarray): 扩展电极图。
        covmap (ndarray): 卷积映射。
    """
    size_smap = smap.shape[0]
    size_lmap = size_smap ** 2

    # 生成扩展的电极图 lmap
    lmap = numpy.empty((size_lmap, size_lmap), dtype=object)
    for n in range(size_lmap):
        for m in range(size_lmap):
            lmap[m, n] = smap[m // size_smap, n // size_smap]
    
    # 生成卷积映射 covmap
    covmap = numpy.empty((size_lmap, size_lmap), dtype=object)
    for n in range(size_lmap):
        for m in range(size_lmap):
            row = n // size_smap  # smap 行索引
            col = m // size_smap  # smap 列索引
            covmap[n, m] = f"{smap[row, col]}*{smap[n % size_smap, m % size_smap]}"
    
    return lmap, covmap

def generate_connectivity_matrix(electrode):
    """
    根据电极索引生成连接矩阵和连接向量。
    
    Args:
        electrode (list): 电极名称列表。
    
    Returns:
        CM (list): 电极连接矩阵。
        CV (list): 展平的连接向量。
    """
    ch = len(electrode)
    CM = [[f"{electrode[n]}*{electrode[m]}" for m in range(ch)] for n in range(ch)]
    CV = numpy.array(CM).flatten()
    return CM, CV 

def generate_covmap_num(covmap, CV):
    """
    根据连接向量(CV)生成数值卷积图(covmap_num)。
    
    Args:
        covmap (ndarray): 卷积映射。
        CV (ndarray): 展平的连接向量。
    
    Returns:
        covmap_num (ndarray): 数值卷积图。
    """
    size = covmap.shape[0]
    covmap_num = numpy.zeros((size, size), dtype=int)
    
    for i in range(size):
        for j in range(size):
            try:
                a = numpy.where(CV == covmap[i, j])[0][0] + 1
                covmap_num[i, j] = a
            except IndexError:
                covmap_num[i, j] = 0
    
    return covmap_num

def cm2sfcc(cm_data, covmap_num, imshow=False):
    """
    根据 cm 和 covmap_num 生成sfcc图。
    
    Args:
        cm (ndarray): 连通性矩阵。
        covmap_num (ndarray): 数值卷积图。
    
    Returns:
        sfcc (ndarray): sfcc图。
    """
    
    samples, size1_cm, size2_cm = cm_data.shape    
    size1_covm, size2_covm = covmap_num.shape
    
    cm_flatten = cm_data.reshape(samples, -1)    
    sfcc_temp = numpy.zeros((samples, size1_covm, size2_covm))
    
    for k in range(samples):
        for i in range(size1_covm):
            for j in range(size2_covm):
                tempnum = covmap_num[i, j]
                
                if tempnum == 0:
                    sfcc_temp[k, i, j] = 0
                else:
                    sfcc_temp[k, i, j] = cm_flatten[k, tempnum - 1]
    
    if imshow:
        utils_common.draw_projection(numpy.mean(sfcc_temp, axis=0))
    
    return sfcc_temp

# %% Example Usage
if __name__ == "__main__":
    # %% example
    # cms = numpy.random.rand(100, 4, 4) 
    # # step 0: distribution
    # smap = numpy.array([["ch1", "ch2"], ["ch3", "ch4"]])
    # order = ["ch1", "ch2", "ch3", "ch4"]
    
    # # step 1-4
    # lmap, covmap = generate_lmap_and_covmap(smap)
    # CM, CV = generate_connectivity_matrix(order)
    # covmap_num = generate_covmap_num(covmap, CV)
    # sfcc = cm2sfcc(cms, covmap_num, imshow=True)
    
    # utils_common.draw_projection(sfcc)
    
    # # %% seed sample
    # dataset, sample, feature, band = 'SEED', 'sub1ex1', 'PCC', 'gamma'
    # cms = utils_common.load_cms(dataset=dataset, experiment=sample, feature=feature, band=band)
    
    # # steo 0: read distribution
    # smap, order = read_distribution(dataset='SEED')
    
    # # step 1-4
    # lmap, covmap = generate_lmap_and_covmap(smap)
    # CM, CV = generate_connectivity_matrix(order)
    # covmap_num = generate_covmap_num(covmap, CV)
    # sfcc = cm2sfcc(cms, covmap_num, imshow=True)

    # # integration
    # cms_joint = utils_common.load_cms(dataset=dataset, experiment=sample, feature=feature, band='joint')
    # sfcc_joint = generate_sfcc(cms_joint, dataset="SEED", imshow=True)

    # %% dreamer sample
    dataset, sample, feature, band = 'DREAMER', 'sub1', 'PCC', 'gamma'
    # cms_d = utils_common.load_cms(dataset=dataset, experiment=sample, feature=feature, band=band)
    
    # # steo 0: read distribution
    # smap_d, order_d = read_distribution(dataset='DREAMER')
    
    # # step 1-4
    # lmap_d, covmap_d = generate_lmap_and_covmap(smap_d)
    # CM_d, CV_d = generate_connectivity_matrix(order_d)
    # covmap_num_d = generate_covmap_num(covmap_d, CV_d)
    # sfcc_d = cm2sfcc(cms_d, covmap_num_d, imshow=True)
    
    # integration
    cms_joint_d = utils_common.load_cms(dataset=dataset, experiment=sample, feature=feature, band='joint')
    sfcc_joint_d = generate_sfcc(cms_joint_d, dataset="DREAMER", imshow=True)
    
    import numpy as np
    import featureengineering_dreamer
    fcs = featureengineering_dreamer.interpolate_matrices(sfcc_joint_d)
    utils_common.draw_projection(np.mean(fcs, axis=(0)))