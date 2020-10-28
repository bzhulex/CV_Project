# Authors: Garrison Ramey
# Date: 2020-10-26
# Time: 10:46 PM EDT

import sys
from os.path import isdir, join
from os import listdir, chdir
from glob import glob
import numpy as np
import shot_boundary_detection as sbd
from argparse import ArgumentParser
from time import time
import keyframe_image_display as kid
import matplotlib.pyplot as plt
frames_dir = None


def handle_args(args):
    """
    Handles arguments

    Notes
    -----
    Needs to be updated to make sure all files in the frames_dir are the same type.
    """
    frames_dir = args.fd 
    save_dir = args.sd 
    if frames_dir == None or save_dir == None:
        print("You must specify the absolute path to the video frames (-fp)" \
              " and the absolute path to the directory you wish to save the keyframes (-sp)")
        sys.exit(0)
    # Check if provided paths exist on the system.
    frames_dir_exists = isdir(frames_dir)
    save_dir_exists = isdir(save_dir)
    if frames_dir_exists == False:
        print("The video frames directory provided does not exist on your system." \
              " Please provide a valid path.")
        sys.exit(0)
    if save_dir_exists == False:
        print("The save directory provided does not exist on your system." \
              " Please provide a valid path.")
        sys.exit(0)
    # Check that there are only frames in the directory and all have the same type
    frame_type = "npy"
    return frames_dir, save_dir, frame_type

def apply_PCC_thresholds(red_coeffs, green_coeffs, blue_coeffs):
    """
    Returns all indices in the coeff channel arrays that are below their respective thresholds
    """
    red_threshold = sbd.channel_coeff_threshold(red_coeffs)
    green_threshold = sbd.channel_coeff_threshold(green_coeffs)
    blue_threshold = sbd.channel_coeff_threshold(blue_coeffs)

    red_indices = sbd.coeff_indices_below_threshold(red_coeffs, red_threshold) + 1
    green_indices = sbd.coeff_indices_below_threshold(green_coeffs, green_threshold) + 1
    blue_indices = sbd.coeff_indices_below_threshold(blue_coeffs, blue_threshold) + 1

    all_indices_below_threshold = np.hstack((red_indices, green_indices, blue_indices))
    all_indices_below_threshold = np.unique(all_indices_below_threshold)

    return all_indices_below_threshold 

def apply_CM_thresholds(cm_means, cm_stds, cm_skews, cm_kurtoses):
    """
    Documentation needed

    Notes
    -----
    If the element at index 0 of cm_diff exceeds the threshold, that means there is 
    a difference between frames 1 and 2, and frame 2 would be marked as the shot boudary index.
    In this case frame 2 would correspond to index 1 of the original frame_files. So, the corresponding
    frames that exceed the threshold should be the index of the cm_diff array that exceeds thresh PLUS 1.
    """

    cm_means_diff = sbd.single_feature_mean_difference(cm_means)
    cm_stds_diff = sbd.single_feature_mean_difference(cm_stds)
    cm_skews_diff = sbd.single_feature_mean_difference(cm_skews)
    cm_kurtoses_diff = sbd.single_feature_mean_difference(cm_kurtoses)

    cm_means_threshold = sbd.color_moment_threshold(cm_means_diff)
    cm_stds_threshold = sbd.color_moment_threshold(cm_stds_diff)
    cm_skews_threshold = sbd.color_moment_threshold(cm_skews_diff)
    cm_kurtoses_threshold = sbd.color_moment_threshold(cm_kurtoses_diff)

    cm_means_indices = sbd.color_moment_indices_above_threshold(cm_means_diff, cm_means_threshold) + 1
    cm_stds_indices = sbd.color_moment_indices_above_threshold(cm_stds_diff, cm_stds_threshold) + 1
    cm_skews_indices = sbd.color_moment_indices_above_threshold(cm_skews_diff, cm_skews_threshold) + 1
    cm_kurtoses_indices = sbd.color_moment_indices_above_threshold(cm_kurtoses_diff, cm_kurtoses_threshold) + 1

    all_indices_above_threshold = np.hstack((cm_means_indices, cm_stds_indices, cm_skews_indices, cm_kurtoses_indices))
    all_indices_above_threshold = np.unique(all_indices_above_threshold)

    return all_indices_above_threshold


def get_keyframes_from_shots(all_keyframe_indices, cm_means, cm_stds):
    """
    Gets all keyframes from shot boundaries

    Notes
    -----
    The paper says to choose the frame with the highest cm_mean and cm_std as the keyframe.
    If one frame does not have the highest for both, then a strategy of dominance is used. 
    I did not implement the strategy of dominance, instead I am selecting the frames that 
    satisfy the highest mean and std. Therefore, two frames may be selected from a single shot.
    """
    keyframe_indices = np.array([])
    num_keyframes = len(all_keyframe_indices)
    left = 0
    for i in range(num_keyframes):
        right = all_keyframe_indices[i]
        kf_mean_index = np.argmax(cm_means[left:right]) + left
        kf_std_index = np.argmax(cm_stds[left:right]) + left
        keyframe_indices = np.hstack((keyframe_indices, kf_mean_index, kf_std_index))
        left = right

    return np.sort(np.unique(keyframe_indices))


def select_keyframes(frame_dir, save_dir, frame_type="npy"):
    """
    Selects and saves the keyframes found in a video

    Parametsrs
    ----------
    frame_dir: str
        The directory containing all frames of the video. 
        Frames must be of type npy, png, jpg, or jpeg
    save_dir: str
        The directory to save the selected keyframes
    frame_type: str
        The file type of the frames in frame_dir
    
    Returns
    -------
    keyframes: array of type str and shape (N, )
        Array containing the names of all the frames that were selected as keyframes

    Notes
    -----
    This method does not support any frame types other than .npy at the moment. Needs
    to be updated so that it can handle the images themselves.
    """
    # Change to frames directory
    chdir(frame_dir)
    # Get all files in frames directory
    all_frame_files = listdir(frame_dir)
    num_frames = len(all_frame_files)

    # These will be the arrays for the pearson color coefficients and color moments
    red_coeffs = np.array([])
    green_coeffs = np.array([])
    blue_coeffs = np.array([])
    cm_means = np.array([])
    cm_stds = np.array([])
    cm_skews = np.array([])
    cm_kurtoses = np.array([])

    for i in range(num_frames):
        # Load frame(s)
        frame1 = np.load(all_frame_files[i])
        if (i + 1) < num_frames:
            # PCC
            frame2 = np.load(all_frame_files[i + 1])

            # Get RGB channels for first frame
            red_channel_1 = frame1[:,:,0].copy().reshape(-1)
            green_channel_1 = frame1[:,:,1].copy().reshape(-1)
            blue_channel_1 = frame1[:,:,2].copy().reshape(-1)

            # Get RGB channesl for second frame
            red_channel_2 = frame2[:,:,0].copy().reshape(-1)
            green_channel_2 = frame2[:,:,1].copy().reshape(-1)
            blue_channel_2 = frame2[:,:,2].copy().reshape(-1)

            red_coeff = sbd.channel_correlation_coefficient(red_channel_1, red_channel_2)
            green_coeff = sbd.channel_correlation_coefficient(green_channel_1, green_channel_2)
            blue_coeff = sbd.channel_correlation_coefficient(blue_channel_1, blue_channel_2)

            red_coeffs = np.hstack((red_coeffs, red_coeff))
            green_coeffs = np.hstack((green_coeffs, green_coeff))
            blue_coeffs = np.hstack((blue_coeffs, blue_coeff))
        
        # Color Moment
        grayscale = sbd.rgb2gray(frame1)
        grayscale_hist = sbd.grayscale_histogram(grayscale)

        cm_mean = sbd.color_moment_mean(grayscale_hist)
        cm_std = sbd.color_moment_std(grayscale_hist)
        cm_skew = sbd.color_moment_skewness(grayscale_hist)
        cm_kurtosis = sbd.color_moment_kurtosis(grayscale_hist)

        cm_means = np.hstack((cm_means, cm_mean))
        cm_stds = np.hstack((cm_stds, cm_std))
        cm_skews = np.hstack((cm_skews, cm_skew))
        cm_kurtoses = np.hstack((cm_kurtoses, cm_kurtosis))
    
    PCC_shot_indices = apply_PCC_thresholds(red_coeffs, green_coeffs, blue_coeffs)
    CM_shot_indices = apply_CM_thresholds(cm_means, cm_stds, cm_skews, cm_kurtoses)


    all_shot_indices = np.intersect1d(PCC_shot_indices, CM_shot_indices)

    #kid.pick_images(all_shot_indices)


    #all_shot_indices = np.sort(np.unique(np.hstack((PCC_shot_indices, CM_shot_indices))))

    keyframe_indices = get_keyframes_from_shots(all_shot_indices, cm_means, cm_stds)

    # Change to save directory to save keyframes
    chdir(save_dir)
    for keyframe_index in keyframe_indices:
        kf_file_name = all_frame_files[int(keyframe_index)]
        # save the keyframe file to the save directory here
        kf = np.load(join(frame_dir, kf_file_name))
        with open(kf_file_name, "wb") as f:
            np.save(f, kf)

    histogram, bin_edges = np.histogram(all_shot_indices, bins=20)
    list_of_images = []
    #print(len(bin_edges))
    for i in range(len(bin_edges) - 1):
        list_of_images.append(int(bin_edges[i]))

    r = 4
    c = 5

    fig = plt.figure()
    count = 1

    for i in range(len(list_of_images)):
        img = np.load(join(frame_dir, all_frame_files[int(list_of_images[i])]))
        ax = fig.add_subplot(r, c, count)
        ax.axis('off')
        ax.imshow(img)
        count += 1


    plt.savefig("top_20_images.png")

    # with open("vid3_ chosen_keyframes.npy", "wb") as f:
    #     np.save("vid3_ chosen_keyframes.npy", list_of_images)


    plt.show()


    print(bin_edges)

if __name__ == "__main__":
    start = time()

    ap = ArgumentParser()
    ap.add_argument("-fd", help="absolute path to the directory containing the video frames")
    ap.add_argument("-sd", help="path to directory to save the selected keyframes")
    args = ap.parse_args()
    frames_dir, save_dir, frame_type = handle_args(args)

    select_keyframes(frame_dir=frames_dir, save_dir=save_dir, frame_type="npy")

    finish = time()

    print("Time Elapsed: {}".format(finish - start))
