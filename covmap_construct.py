import os
import numpy
import pandas

import utils

def generate_sfcc(cm_data, imshow=False):
    """
    Generate SFCC (Sparse Functional Connectivity Convolution) from CM data.

    Args:
        cm_data (numpy.ndarray): Input connectivity matrix data. Can be 3D (samples, size, size) 
                                 or 4D (samples, channels, size, size).

    Returns:
        numpy.ndarray: The generated SFCC data.
    """
    # Get the spatial map (smap) and channel order from a utility function
    smap, order = get_distribution()
    
    # Step 1: Generate lmap and covmap
    lmap, covmap = generate_lmap_and_covmap(smap)
    
    # Step 2: Generate connectivity matrix and vector
    CM, CV = generate_connectivity_matrix(order)
    
    # Step 3: Generate numerical covmap
    covmap_num = generate_covmap_num(covmap, CV)
    
    if cm_data.ndim == 3:
        # Case: 3D input data (samples, size, size)
        samples, size, _ = cm_data.shape
        
        # Generate SFCC
        sfcc = cm2sfcc(cm_data, covmap_num)
    
    elif cm_data.ndim == 4:
        # Case: 4D input data (samples, channels, size, size)
        samples, channels, size, _ = cm_data.shape
        
        # Generate SFCC for each channel
        sfcc_ch0 = cm2sfcc(cm_data[:, 0, :, :], covmap_num)
        sfcc_ch1 = cm2sfcc(cm_data[:, 1, :, :], covmap_num)
        sfcc_ch2 = cm2sfcc(cm_data[:, 2, :, :], covmap_num)
        
        # Stack the SFCC results across channels
        sfcc = numpy.stack([sfcc_ch0, sfcc_ch1, sfcc_ch2], axis=1)
    else:
        raise ValueError("Input cm_data must be 3D or 4D.")

    if imshow: utils.draw_projection(sfcc[0])

    return sfcc

def get_distribution():
    # path
    path_current = os.getcwd()
   
    # path_smap
    path_smap = os.path.join(path_current, 'mapping', 'smap.txt')
    # path_channel order
    path_order = os.path.join(path_current, 'mapping', 'biosemi62_64_channels_original_distribution.txt')
    
    # read smap, channel order from .txt
    smap, order = pandas.read_csv(path_smap, sep='\t', header=None), pandas.read_csv(path_order, sep='\t')
    smap, order = numpy.array(smap), numpy.array(order['channel'])
    
    return smap, order

def build_sfcc(smap, order, cm_data):
    # 文件1：生成 lmap 和 covmap
    lmap, covmap = generate_lmap_and_covmap(smap)
    # 文件2：生成连接矩阵和向量
    CM, CV = generate_connectivity_matrix(order)
    # 文件3：生成数值卷积图
    covmap_num = generate_covmap_num(covmap, CV)
    # 文件4：生成sfcc
    sfcc = cm2sfcc(cm_data, covmap_num)
    
    return sfcc
    
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

def cm2sfcc(cm_data, covmap_num):
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
    
    return sfcc_temp

# # %% Example Usage
# # # 示例变量
# # smap = numpy.array([["ch1", "ch2"], ["ch3", "ch4"]])
# # order = ["ch1", "ch2", "ch3", "ch4"]
# # cm_data = numpy.random.rand(100, 4, 4)  # 示例 cm 数据

# # smap, order = get_distribution()
# # cm_data = utils.load_cmdata2d('PCC', 'gamma', 'sub1ex1')

# # # 文件1：生成 lmap 和 covmap
# # lmap, covmap = generate_lmap_and_covmap(smap)

# # # 文件2：生成连接矩阵和向量
# # CM, CV = generate_connectivity_matrix(order)

# # # 文件3：生成数值卷积图
# # covmap_num = generate_covmap_num(covmap, CV)

# # # 文件4：生成灰度图
# # sfcc = cm2sfcc(cm_data, covmap_num)
# # draw_projection(sfcc[0])

# cm_data = utils.load_cmdata2d('PCC', 'joint', 'sub1ex1')
# utils.draw_projection(cm_data[0])
# sfcc = generate_sfcc(cm_data, imshow=True)
