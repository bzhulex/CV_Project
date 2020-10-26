# Author: Garrison Ramey
# Date: 2020-10-25
# Time: 10:47 EDT

import numpy as np

###################
# Methods for PCC #
###################

def channel_threshold(channel, alpha=1):
    """
    Computes the threshold for a single channel in a given frame

    input: 
    - channel: ndarray of shape (N, ) representing a red, blue, or green color channel of a frame
    - alpha: float between 0 and 1, a hyperparameter

    output: 
    - threshold: float representing the color threshold for that frame
    """
    mu = np.mean(channel)
    variance = np.var(channel)
    # Given on page 271 of the paper
    threshold = mu + (alpha * variance)

    return threshold

def channel_correlation_coefficient(channel_1, channel_2):
    """
    Computes the pearson correlation coefficeint between two given frames

    input:
    - channel_1: ndarray of shape(N, ) ; the r/g/b channel corresponding to the first frame
    - channel_2: ndarray of shape(N, ) ; the r/g/b channel corresponding to the second frame
    Both channels should be for the same color

    output:
    - pcc: the pearson correlation coefficient between the two frames for a single color channel
    """

    # Stack the channels then get covariance 
    combined_channels = np.vstack((channel_1, channel_2))

    # Set ddof to zero to make sure the value it returns for one dimension is the same as the variance.
    cov_matrix = np.cov(combined_channels, ddof=0)
    var_1 = cov_matrix[0][0]
    cov = cov_matrix[0][1]
    var_2 = cov_matrix[1][1]

    # Equation 1 given on page 271 in the paper
    coeff = (1 / (var_1 * var_2)) * cov

    return coeff 

############################
# Methods for Color Moment #
############################

def rgb2gray(img):
    """
    Converts an RGB image to a grayscale image
    
    Input: 
    - ndarray of an RGB image of shape (N x M x 3)

    Output:
    - ndarray of the corresponding grayscale image of shape (N x M)
    
    """
    
    if(img.ndim != 3 or img.shape[-1] != 3):
        print("Invalid image! Please provide an RGB image of the shape (N x M x 3) instead.".format(img.ndim))
        return None
    
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def grayscale_histogram(grayscale_img):
    """
    Computes the grayscale histogram for a given grayscale image

    input:
    - grayscale_img: ndarray of type uint8 and shape (N, M)

    output:
    - grayscale_hist: ndarray of shape (256, )
    The ouput histogram gives the number of pixels that have the grayscale
    value at the index value. For example, if hist[20] were equal to 10, that
    would mean that 10 pixels in the image have a graycale value equal to 20.
    """

    hist, _ = np.histogram(grayscale_img, bins=256)

    return hist

def color_moment_mean(grayscale_hist):
    """
    computes the color moment mean as given in equation 2 on page 272 in the paper

    input: 
    - grayscale_hist: ndarray of shape (256, )

    output:
    - cm_mean: float value for the color momement mean
    """
    # sum of grayscale_hist gives total number of pixels, which is N
    N = np.sum(grayscale_hist)
    color_levels = np.arange(256)
    summation_GC = np.sum(color_levels * grayscale_hist)

    mean = (1 / N) * summation_GC

    return mean 

def color_moment_std_dev(grayscale_hist)
    """
    Computes the color moment standard deviation as given in equation 3 on page 272 in the paper

    input: 
    - grayscale_hist: ndarray of shape (256, )

    outpus:
    - cm_std: float value for the color moment standard deviation
    """
    # sum of grayscale_hist gives total number of pixels, which is N
    N = np.sum(grayscale_hist)
    color_levels = np.arange(256)

    cm_mean = color_moment_mean(grayscale_hist)
    Gm_diff_squared = np.square(color_levels - cm_mean)
    summation = Gm_diff_squared * grayscale_hist

    cm_std = np.sqrt((1 / (N - 1)) * summation)

    return cm_std

def color_moment_skewness(grayscale_hist):
    pass







    





