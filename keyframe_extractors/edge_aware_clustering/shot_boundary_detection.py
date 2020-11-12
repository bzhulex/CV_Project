# Authors: Garrison Ramey, Alex Karwashki, Bzhulex
# Date: 2020-11-02

# The following methods are based on the paper by Priya and Dominic

import numpy as np
from os import listdir
import math

def rgb2gray(img):
    """
    Converts an RGB image to a grayscale image
    
    Parameters
    ----------
    ndarray of an RGB image of shape (N x M x 3)

    Returns
    -------
    ndarray of the corresponding grayscale image of shape (N x M)
    
    """
    
    if(img.ndim != 3 or img.shape[-1] != 3):
        print("Invalid image! Please provide an RGB image of the shape (N x M x 3) instead.".format(img.ndim))
        return None
    
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

####################################
# 3.1 - Feature Extraction Methods #
####################################

def block_mean(block):
    """
    Gets the mean for a block of a grayscale image

    Parameters
    ----------
    block: array of shape (N, )
        A single block that exists in the grayscale image
    
    Returns
    -------
    mean: float
        The average grayscale value for the given block
    """
    return np.mean(block)


def edge_classifier_value(block):
    """
    Gets the edge classifier value for a single block of a grayscale image

    Parameters
    ----------
    block: array of shape (N, )
        A single block that exists in the grayscale image
    
    Returns
    -------
    ecv: float
        The edge classifier value
    """
    max_intensity_value = np.max(block)
    median = np.median(block)
    ecv = max_intensity_value - median
    
    return ecv 

def get_block_edge_pattern(block):
    """
    Gets the edge pattern for a single block of a grayscale image

    Parameters
    ----------
    block: array of shape (4, 4)
        A single block that exists in the grayscale image
    
    Returns
    -------
    EP: array of type uint8 and shape (N, )
        The edge pattern for the block

    Notes
    -----
    The paper said that 16 was used for the value of Thr at the 5th line from the bottom of the 
    3rd paragraph on page 109
    """
    Thr = 16
    b_mean = block_mean(block)
    ecv = edge_classifier_value(block)
    if ecv < Thr:
        return np.zeros(block.shape[0])

    edge_pattern = np.zeros(block.shape[0])
    edge_indices = np.where(block > b_mean)[0]
    edge_pattern[edge_indices] = 1
    
    return edge_pattern

####################################
# 3.1.1 - Categorization of Blocks #
####################################

def get_q_value(block):
    """
    Finds the q value (number of pixels in a block higher than the block mean)
    """

    mean = block_mean(block)
    q_value = np.where(block > mean)[0].shape[0]
    
    return q_value

def get_q_prime_value(block):
    """
    Comutes q prime value (number of edge pixels in a nonedge block)

    Parameters
    ----------
    edge_pattern: array of shape (N, )
        The edge pattern for the non-edge block

    Returns
    -------
    q_prime: float
        The number of edge pixels in the nonedge block
    """
    edge_pattern = get_block_edge_pattern(block)

    q_prime_value = np.where(edge_pattern == 1)[0].shape[0]

    return q_prime_value

def lower_mean(block):
    """
    Computes the lower mean for a single block in a grayscale image

    Parameters
    ----------
    block: array of shape (N, )
        A single block that exists in the grayscale image
    
    Returns
    -------
    lower_mean: float
        The lower mean for the block
    """

    mean = block_mean(block)
    q = get_q_value(block)
    summation_indices = np.where(block < mean)[0]
    summation = np.sum(block[summation_indices])

    lower_mean = (1 / (16 - q)) * summation 

    return lower_mean 

def higher_mean(block):
    """
    Computes the higher mean for a single block in a grayscale image

    Parameters
    ----------
    block: array of shape (N, )
        A single block that exists in the grayscale image
    
    Returns
    -------
    higher_mean: float
        The higher mean for the block
    """

    mean = block_mean(block)
    q = get_q_value(block)
    summation_indices = np.where(block >= mean)[0]
    summation = np.sum(block[summation_indices])

    higher_mean = (1 / q) * summation

    return higher_mean

def is_edge_block(edge_pattern):
    """
    Determines whether a block is an edge or nonedge block

    Parameters
    ----------
    edge_pattern: array of shape (N, )
        The edge pattern for a block

    Returns
    -------
    is_edge_block: boolean

    Notes
    -------
    8 is used because the frames we are using can be split perfectly; therefore, each block will contain 16 pixels.
    If over 8 pixels are a 1, then it is an edge block
    """
    num_edge_pixels = np.nonzero(edge_pattern)[0].shape[0]
    
    is_edge_block = num_edge_pixels > 8
    
    return is_edge_block


###########################
# 3.2 Similarity Measures #
###########################

def diff_both_edge_blocks(block_1, block_2):
    """
    Computes the difference between two edge blocks
    pass
    """
    # Get block means
    m1 = block_mean(block_1)
    m2 = block_mean(block_2)
    # Get q values for each block
    q1 = get_q_value(block_1)
    q2 = get_q_value(block_2)
    # Get lower_mean for each block
    lm1 = lower_mean(block_1)
    lm2 = lower_mean(block_2)
    # Get higher_mean for each block
    hm1 = higher_mean(block_1)
    hm2 = higher_mean(block_2)

    # "t" stands for term
    t1 = np.abs((q1 - q2) * (hm1 - hm2))
    t2 = np.abs(lm1 - lm2) + np.abs(hm1 - hm2)
    
    diff = t1 * t2

    return diff


def diff_both_nonedge_blocks(block_1, block_2):
    """
    Computes the difference between two nonedge blocks
    """
    # Get block means
    m1 = block_mean(block_1)
    m2 = block_mean(block_2)
    # Get q values for each block
    q1 = get_q_prime_value(block_1)
    q2 = get_q_prime_value(block_2)

    # "t" stands for term
    t1 = np.abs(q1 - q2)
    t2 = np.abs(m1 - m2)

    diff = 2 * t1 * t2 

    return diff

def diff_edge_nonedge(block_1, block_2):
    """
    Computes the difference between an edge block and a non-edge block

    Parameters
    ----------
    block_1: array of type uint8 and shape (N, )
        the edge block
    block_2: array of shape (N, )
        the non-edge block

    Returns
    -------
    diff: float
        a similarity measure
    """
    # Get the non-edge block mean
    m2 = block_mean(block_2)
    # Get q value for block 1 and q prime value for block 2
    q1 = get_q_value(block_1)
    q2 = get_q_prime_value(block_2)
    # Get lower and higher mean for block 1
    lm1 = lower_mean(block_1)
    hm1 = higher_mean(block_1)

    # "t" stands for term
    t1 = np.abs(q1 - q2)
    t2 = np.abs(lm1 + hm1 - (2 * m2))

    diff = t1 * t2 

    return diff

def diff_nonedge_edge(block_1, block_2):
    """
    Computes the difference between an edge block and a non-edge block

    Parameters
    ----------
    block_1: array of type uint8 and shape (N, )
        the non-edge block
    block_2: array of shape (N, )
        the edge block

    Returns
    -------
    diff: float
        a similarity measure
    """
    # Get mean of non-edge block
    m1 = block_mean(block_1)
    # Get q values
    q1 = get_q_prime_value(block_1)
    q2 = get_q_value(block_2)
    # Get lower and higher mean for block 2
    lm2 = lower_mean(block_2)
    hm2 = higher_mean(block_2)

    # "t" stands for term
    t1 = np.abs(q1 - q2)
    t2 = np.abs((2 * m1) - (lm2 + hm2))

    diff = t1 * t2

    return diff

def frame_blocks_perfect(frame_1, frame_2):
    """
    Splits each frame into (4 x 4) blocks. Each dimension for each frame must be perfectly divisible by 4.
    """
    # Both frames will have same shape
    rows, cols = frame_1.shape

    if rows < 4 or cols < 4:
        raise ValueError("Frame must have a larger shape than (4, 4)")

    num_blocks_down = int(rows / 4)
    num_blocks_across = int(cols / 4)

    # This array's shape[0] should be divisible by 4 with remainder 0
    perfect_blocks_1 = np.zeros(shape=(1, 4))
    perfect_blocks_2 = np.zeros(shape=(1, 4))
    # Split array into groups of 4 rows
    lower_row = 0
    higher_row = 4
    step = 4
    # make sure to delete the first row of zeros after loop
    while higher_row <= rows:
        row_group_1 = frame_1[np.arange(lower_row, higher_row),:]
        row_group_2 = frame_2[np.arange(lower_row, higher_row),:]
        lower_col = 0
        higher_col = 4
        while higher_col <= cols:
            block_1 = row_group_1[:,np.arange(lower_col, higher_col)]
            block_2 = row_group_2[:,np.arange(lower_col, higher_col)]
            perfect_blocks_1 = np.vstack((perfect_blocks_1, block_1))
            perfect_blocks_2 = np.vstack((perfect_blocks_2, block_2))
            lower_col += step
            higher_col += step
        lower_row += step
        higher_row += step
    
    # Delete the leading zeros placeholder
    perfect_blocks_1 = np.delete(perfect_blocks_1, 0, axis=0)
    perfect_blocks_2 = np.delete(perfect_blocks_2, 0, axis=0)

    return perfect_blocks_1, perfect_blocks_2


def frame_blocks_extra(frame_1, frame_2):
    """
    Splits each frame into (4 x 4) blocks

    Notes
    ------
    If there were extra rows, then I considered all cols and made
    blocks of size num_extra_rows x 4. If there were any remaining columns, I discarded them.
    If there were extra cols, then I considered all rows up to, but not including, the start of
    the extra rows and made blocks of size 4 x num_extra_cols. 

    I split the frame into 3 regions to handle frames whose size is not perfectly divisble by 16 (4 x 4)
    """
    # Both frames will have same shape
    rows, cols = frame_1.shape

    if rows < 4 or cols < 4:
        raise ValueError("Frame must have a larger shape than (4, 4)")
    
    num_blocks_across = int(cols / 4)
    extra_width = cols % 4

    num_blocks_down = int(rows / 4)
    extra_height = rows % 4
    # There should be num_blocks_across * num_blocks_down perfect blocks. The vars above are for testing

    # Consider all blocks that fit perfectly first (Region 1)
    # This array's shape[0] should be divisible by 4 with remainder 0
    perfect_blocks_1 = np.zeros(shape=(1, 4))
    perfect_blocks_2 = np.zeros(shape=(1, 4))
    # Split array into groups of 4 rows
    lower_row = 0
    higher_row = 4
    step = 4
    # make sure to delete the first row of zeros after loop
    while higher_row <= rows:
        row_group_1 = frame_1[np.arange(lower_row, higher_row),:]
        row_group_2 = frame_2[np.arange(lower_row, higher_row),:]
        lower_col = 0
        higher_col = 4
        while higher_col <= cols:
            block_1 = row_group_1[:,np.arange(lower_col, higher_col)]
            block_2 = row_group_2[:,np.arange(lower_col, higher_col)]
            perfect_blocks_1 = np.vstack((perfect_blocks_1, block_1))
            perfect_blocks_2 = np.vstack((perfect_blocks_2, block_2))
            lower_col += step
            higher_col += step
        lower_row += step
        higher_row += step
    
    # Delete the leading zeros placeholder
    perfect_blocks_1 = np.delete(perfect_blocks, 0, axis=0)
    perfect_blocks_2 = np.delete(perfect_blocks, 0, axis=0)

    # Get the blocks of the extra rows (Region 2)
    start_index = higher_row - step
    extra_row_blocks_1 = None
    extra_row_blocks_2 = None
    lower_col = 0
    higher_col = 4
    if start_index < rows:
        extra_row_blocks_1 = np.zeros(shape=(rows - start_index, 4))
        extra_row_blocks_2 = np.zeros(shape=(rows - start_index, 4))
        extra_rows_group_1 = np.delete(frame_1, np.arange(0, start_index), axis=0)
        extra_rows_group_2 = np.delete(frame_2, np.arange(0, start_index), axis=0)
        while higher_col <= cols:
            block_1 = extra_rows_group_1[:,np.arange(lower_col, higher_col)]
            block_2 = extra_rows_group_2[:,np.arange(lower_col, higher_col)]
            extra_rows_blocks_1 = np.vstack((extra_rows_blocks_1, block_1))
            extra_rows_blocks_2 = np.vstack((extra_rows_blocks_2, block_2))
            lower_col += step 
            higher_col += step
        
        # Delete the leading zeros placeholders
        extra_row_blocks_1 = np.delete(extra_row_blocks_1, 0, axis=0)
        extra_row_blocks_2 = np.delete(extra_row_blocks_2, 0, axis=0)

    # Get the blocks of the extra columsn (Region 3)
    start_index = higher_col - step
    extra_cols_blocks_1 = None 
    extra_cols_blocks_2 = None
    if start_index < cols:
        extra_cols_blocks_1 = np.zeros(shape=(4, cols - start_index))
        extra_cols_blocks_2 = np.zeros(shape=(4, cols - start_index))
        extra_cols_group_1 = np.delete(frame_1, np.arange(0, start_index), axis=1)
        extra_cols_group_2 = np.delete(frame_2, np.arange(0, start_index), axis=1)
        lower_row = 0
        higher_row = 4
        while higher_row <= rows:
            block_1 = extra_cols_group_1[np.arange(lower_row, higher_row),:]
            block_2 = extra_cols_group_2[np.arange(lower_row, higher_row),:]
            extra_cols_blocks_1 = np.vstack((extra_cols_blocks_1, block_1))
            extra_cols_blocks_2 = np.vstack((extra_cols_blocks_2, block_2))

        # Delete the leading zeros placeholders
        extra_row_blocks_1 = np.delete(extra_row_blocks_1, 0, axis=0)
        extra_row_blocks_2 = np.delete(extra_row_blocks_2, 0, axis=0)

    return perfect_blocks_1, perfect_blocks_2, extra_rows_blocks_1, extra_rows_blocks_2, extra_cols_blocks_1, extra_cols_blocks_2

def get_diff_values(frame1, frame2):
    """
    Gets the difference/continuity value for all corresponding blocks between 2 consecutive frames

    Returns
    ---------
    diff_values: array of type float and shape (N, )
        All difference values between corresponding blocks of 2 consecutive frames

    Notes
    -------
    Should return an array of shape(14400, ) for the frames we are using. 
    We should expect a lot of zeros in this array since consecutive frames are likely to be similar.
    """
    diff_values = np.array([])
    # pb1 and pb2 have shape (N, 4) where N is the total number of pixels divided by 4 
    pb1, pb2 = frame_blocks_perfect(frame1, frame2)
    start = 0
    step = 4
    while start < pb1.shape[0]:
        # will be a 4 x 4 block so must reshape to (16, )
        diff_value = None
        upper = start + step
        block_1 = pb1[range(start, upper),:].reshape(-1)
        block_2 = pb2[range(start, upper),:].reshape(-1)
        ep1 = get_block_edge_pattern(block_1)
        ep2 = get_block_edge_pattern(block_2)
        b1_is_edge = is_edge_block(ep1)
        b2_is_edge = is_edge_block(ep2)
        if b1_is_edge and b2_is_edge:
            diff_value = diff_both_edge_blocks(block_1, block_2)
        elif b1_is_edge and (not b2_is_edge):
            diff_value = diff_edge_nonedge(block_1, block_2)
        elif b2_is_edge and (not b1_is_edge):
            diff_value = diff_nonedge_edge(block_1, block_2)
        else:
            diff_value = diff_both_nonedge_blocks(block_2, block_2)
        # Append diff value to array of diff values
        diff_values = np.hstack((diff_values, diff_value))
        # Increment start by 4 to consider next pair of corresponding blocks
        start += 4

    return diff_values
        

def frame_continuity_value(diff_values):
    """
    Computes the continuity value which represents ALL corresponding blocks in consecutive frames

    Parameters
    ----------
    diff_values: Array of shape (N, )
        The array that contains all the corresponding difference/similarity scores between 2 consecutive frames
    
    Returns
    -------
    continuity_value: float
        The continuity value between 2 consecutive frames
    """
    return np.sum(diff_values)

def full_sequence_diff_vals(frame_dir):
    """
    Computes the difference/continuity values between ALL consecutive frames in the video
    """
    frame_names = listdir(frame_dir)
    full_sequence_diffs = np.array([])
    for i in range(len(frame_name) - 1):
        f1_name = frame_names[i]
        f2_name = frame_names[i + 1]
        f1 = np.load(join(frame_dir, f1_name))
        f2 = np.load(join(frame_dir, f2_name))
        f1_gray = rgb2gray(f1)
        f2_gray = rgb2gray(f2)
        diff_values = get_diff_values(f1_gray, f2_gray)
        cont_val = frame_continuity_value(diff_values)
        full_sequence_diffs = np.hstack((full_sequence_diffs, cont_val))
    
    return full_sequence_diffs

    
###########################
# 3.3 Least Squares       #
###########################

def least_squared_error(diff_values):
    g = diff_values * np.transpose(diff_values)
    m = np.zeros((2, len(diff_values)))
    m[:0] = diff_values
    m[:1] = diff_values

    m_zero = np.zeros((2, 2))

    h_0 = np.concatenate((g, np.transpose(m)), axis=1)
    h_1 = np.concatenate((m, m_zero), axis=1)
    h = np.concatenate((h_0, h_1), axis=0)

    b = h * diff_values

    f = None
    total = 0
    for i in range(1, len(diff_values) + 1):
        f = b[m + len(diff_values)]
        total += np.sum(diff_values)

###########################
# 3.4 Shot Frame Clustering#
###########################

def shot_frame_clustering(diff_values):
    """
    diff_values here is for the entire sequence of frames. has shape (N-1, ) where N is total number of frames
    """
    delta = 5
    d = 1
    i = 1
    clusters = []
    clusters.append([diff_values[0]])
    i += 1
    cont_val = frame_continuity_value(diff_values)
    while i < len(diff_values):
        if diff_values[i] < 5:
            clusters[d] = np.append(clusters[d], i)
        elif len(clusters[d]) == 1:
            clusters[d - 1] = np.append(clusters[d - 1], i)
        elif len(clusters[d]) > 1 and len(clusters[d]) < 5:
            l = clusters[d - i][len(clusters[d - 1]) - 1]
            f = clusters[d][0]
            s = clusters[d][1]

            if (diff_values[f] - diff_values[l]) - (diff_values[f] - diff_values[s]) < 0.5*delta:
                clusters[d - 1] = np.append(clusters[d], clusters[d - 1])
                clusters[d - 1] = np.flatten(clusters[d - 1])
                clusters[d] = np.delete(clusters, d)
            else:
                d += 1
        i += 1
    
    return clusters
    




