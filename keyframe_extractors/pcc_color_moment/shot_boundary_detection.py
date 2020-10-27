# Authors: Garrison Ramey
# Date: 2020-10-25
# Time: 10:47 EDT

import numpy as np

###################
# Methods for PCC #
###################

def channel_threshold(channel, alpha=1):
    """
    Computes the threshold for a single channel in a given frame by equation 6 on page 271 of the paper

    Parameters
    ----------
    channel: ndarray of shape (N, ) 
        represents a red, blue, or green color channel of a frame
    alpha: float 
        A hyperparameter between 0 and 1

    Returns
    -------
    threshold: float 
        represents the color threshold for a channel in frame

    Notes
    -----
    Given in equation 6 on page 273 in the paper
    """

    if len(channel.shape) != 1:
        raise ValueError("The channel has the incorrect shape. It must be (N, )")

    mu = np.mean(channel)
    variance = np.var(channel)
    # Equation 6 on page 271 of the paper
    threshold = mu + (alpha * variance)

    return threshold

def channel_correlation_coefficient(channel_1, channel_2):
    """
    Computes the pearson correlation coefficeint between two given frames

    Parameters
    ----------
    channel_1: ndarray of shape(N, )  
        the r/g/b channel corresponding to the first frame
    channel_2: ndarray of shape(N, ) 
        the r/g/b channel corresponding to the second frame

    Returns
    -------
    pcc: float
        the pearson correlation coefficient between the two frames for a single color channel

    Notes
    -----
    Both channels should be for the same color
    """
    if len(channel_1.shape) != 1 or len(channel_2.shape) != 1:
        raise ValueError("One of the channels has the incorrect shape. Both must be (N, )")

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

def grayscale_histogram(grayscale_img):
    """
    Computes the grayscale histogram for a given grayscale image

    Parameters
    ----------
    grayscale_img: ndarray of type uint8 and shape (N, M)

    Returns
    -------
    grayscale_hist: ndarray of shape (256, )

    Notes
    -----
    The ouput histogram gives the number of pixels that have the grayscale
    value at the index value. For example, if hist[20] were equal to 10, that
    would mean that 10 pixels in the image have a graycale value equal to 20.
    """

    if len(grayscale_img.shape) != 2:
        raise ValueError("The image has the incorrect shape. It must be (N, M)")

    hist, _ = np.histogram(grayscale_img, bins=256)

    return hist

def color_moment_mean(grayscale_hist):
    """
    computes the color moment mean

    Parameters
    ----------
    grayscale_hist: ndarray of shape (256, )

    Returns
    -------
    cm_mean: float 
        value for the color momement mean

    Notes
    -----
    Given in equation 2 on page 272 in the paper
    """

    if len(grayscale_hist) != 1 or grayscale_hist.shape[0] != 256:
        raise ValueError("The histogram has the incorrect shape. It must be (N, )")

    # sum of grayscale_hist gives total number of pixels, which is N
    N = np.sum(grayscale_hist)
    color_levels = np.arange(256)
    summation_GC = np.sum(color_levels * grayscale_hist)

    mean = (1 / N) * summation_GC

    return mean 

def color_moment_std(grayscale_hist)
    """
    Computes the color moment standard deviation

    Parameters
    ----------
    grayscale_hist: ndarray of shape (256, )
        a grayscale histogram of a single frames
    
    Returns
    -------
    cm_std: float 
        value for the color moment standard deviation
    
    Notes
    -----
    Given in equation 3 on page 272 in the paper
    """
    if len(grayscale_hist) != 1 or grayscale_hist.shape[0] != 256:
        raise ValueError("The histogram has the incorrect shape. It must be (N, )")

    # sum of grayscale_hist gives total number of pixels, which is N
    N = np.sum(grayscale_hist)
    color_levels = np.arange(256)

    cm_mean = color_moment_mean(grayscale_hist)
    Gm_diff_squared = np.square(color_levels - cm_mean)
    summation = np.sum(Gm_diff_squared * grayscale_hist)

    cm_std = np.sqrt((1 / (N - 1)) * summation)

    return cm_std

def color_moment_skewness(grayscale_hist):
    """
    Computes the color moment skewness for a frame

    Parameters
    ----------
    grayscale_hist: ndarray of shape (256, )
        a grayscale histogram of a single frame

    Returns
    -------
    skewness: float 
        value for the color moment skewness of a single frame

    Notes
    -----
    Given in equation 4 on page 272 in the paper
    """
    if len(grayscale_hist) != 1 or grayscale_hist.shape[0] != 256:
        raise ValueError("The histogram has the incorrect shape. It must be (N, )")
    
    N = np.sum(grayscale_hist)
    color_levels = np.arange(256)

    cm_mean = color_moment_mean(grayscale_hist)
    Gm_diff_cubed = np.power(color_levels - cm_mean, 3)
    summation = np.sum(Gm_diff_cubed * grayscale_hist)

    cm_std = color_moment_std(grayscale_hist)
    cm_std_cubed = cm_std ** 3
    skewness = (1 / ((N - 1) * cm_std_cubed)) * summation 

    return skewness

def color_moment_kurtosis(grayscale_hist):
    """
    Computes the color moment kurtosis for a frame

    Parameters
    ----------
    grayscale_hist: ndarray of shape (256, )
        a grayscale histogram of a single frame

    Returns
    -------
    kurtosis: float 
        value for the color moment kurtosis of a single frame
    
    Notes
    -----
    Given in equation 5 on page 272 in the paper
    """
    if len(grayscale_hist) != 1 or grayscale_hist.shape[0] != 256:
        raise ValueError("The histogram has the incorrect shape. It must be (N, )")

    N = np.sum(grayscale_hist)
    color_levels = np.arange(256)

    cm_mean = color_moment_mean(grayscale_hist)
    Gm_diff_quartic = np.power(color_levels - cm_mean, 4)
    summation = np.sum(Gm_diff_quartic * grayscale_hist)

    cm_std = color_moment_std(grayscale_hist)
    cm_std_quartic = cm_std ** 4
    kurtosis = (1 / ((N - 1) * cm_std_quartic)) * summation

    return kurtosis


def single_feature_mean_difference(cm_feature_arr):
    """
    Computes the mean difference array of a color moment feature space (mean, std, skewness, or kurtosis)

    Parameters
    ----------
    cm_feature_arr: array of shape (N, ) 
        An array containing the elements for a color moment feature

    Returns
    -------
    cm_difference: ndarray of shape (N - 1, )
        The mean difference array for the frames

    Notes
    -----
    The array returned from this method should be used to compute the color moment thresholds
    """

    if len(cm_feature_arr.shape) != 1:
        raise ValueError("The channel has the incorrect shape. It must be (N, )")
    
    length = cm_feature_arr.shape[0]

    # This will be used to take the difference using array broadcasting
    # take out the first element. will have shape (N - 1, ) now
    feature_arr_2 = np.delete(cm_feature_arr, 0, axis=0)

    # Take out the last element
    cm_feature_arr = np.delete(cm_feature_arr, length - 1, axis=0)

    # Take absolute value of the difference between each of the frames
    cm_difference = np.absolute(cm_feature_arr - feature_arr_2)

    return cm_difference

def color_moment_threshold(cm_difference_arr, alpha=1):
    """
    Computes the threshold for a single feature of the color moment.

    Parameters 
    ----------
    cm_difference_arr: (N, ) array
        The color moment difference array 
    alpha: float
        A hyperparameter between 0 and 1 that affects the value of the threshold
    
    Returns
    -------
    threshold: float
        The threshole for a single feature of the color moment

    Notes
    -----
    This works for mean, std, skewness, and kurtosis because they all have the same equations
    given on pages 273 to 274 at eq (9), (10), (11), (12) 
    """
    if len(cm_difference_arr.shape) != 1:
        raise ValueError("The difference array must have shape (N, ),")

    mu = np.mean(cm_difference_arr)

    threshold = alpha * mu 

    return threshold








    





