# Authors: Garrison Ramey,
# Date: 2020-11-02

import sys
from os.path import isdir, join
from os import listdir, chdir
import numpy as np
from argparse import ArgumentParser
from time import time
import matplotlib.pyplot as plt

import shot_boundary_detection as sbd

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

########################
#  3.5 Keyframe Extraction  #
########################

def keyframes_from_clusters(clusters, epsilon, delta):
    """
    Selects the keyframes from a list of clusters

    Parameters
    -------------
    clusters: list of lists
        The cluster list that contains the indices of the frames that were clustered together
    epsilon: float
        A hyperparameter threshold to compare against the cluster variance
    delta: float
        A hyperparameter threshold used to determine subshot clusters
    
    Returns
    ---------
    keyframes: array of type int and size (N, )
        The keyframe indices

    Notes
    -------
    Line 8 on page 112 of the paper is difficult to interpret because the entire condition and assignment to perform if it is true  is on the same line.
    I believe it is saying if the absolute value of the difference is less than delta, then SC_{j} = continuity value of C_{c}. In other
    words, the subshot cluster should have the continuity value at "i" of the entire cluster added to it.

    I expect the subshot clusters to be large since we will be taking a keyframe from each of them as explained in step 12 of section 3.5
    """
    keyframe_indices = np.array([], dtype=np.uint32)
    for cluster in clusters:
        var = np.var(cluster)
        if var < epsilon:
            # Choose 1 frame from the shot/cluster
            c_values = np.array(cluster)
            centroid = np.mean(c_values)
            closest_index = np.argmin(np.abs(c_values - centroid))
            frame_index = c_values[closest_index]
            keyframe_indices = np.append(keyframe_indices, frame_index)
        else:
            # Create subshots of total shot/cluster and then choose keyframes from those subshots
            # our clusters start at index 0, the notation they use in th paper their clusters start at 1
            j, k = (0, 0)
            subshots = []
            subshot = []
            for i in range(len(cluster) - 1):
                t1 = np.abs(cluster[k] - cluster[i + 1])
                if t1 < delta:
                    subshot.append(cluster[i])
                else:
                    j += 1
                    k = i + 1
                    subshots.append(subshot)
                    subshot = []
            # Now that we have all subshots, we need to get a keyframe from each one
            for shot in subshots:
                shot_vals = np.array(shot)
                centroid = np.mean(shot_vals)
                closest_index = np.argmin(np.abs(shot_vals - centroid))
                frame_index = shot_vals[closest_index]
                keyframe_indices = np.append(keyframe_indices, frame_index)
    
    return keyframe_indices

def select_keyframe(frame_dir, save_dir, frame_type=None, epsilon=None, delta=None):
    """
    Finds the keyframe indices of a full video sequence

    Parameters
    -------------
    frame_dir: str
        The directory in which the numpy frames exist
    epsilon: float
        A hyperparameter threshold to compare against the cluster variance
    delta: float
        A hyperparameter threshold used to determine subshot clusters
    
    Returns
    ---------
    keyframe_indices: array of type int and size (N, )
        The keyframe indices
    """
    # sbd.full_sequence_diff_vals has been tested and should run without error.
    full_sequence_diffs = sbd.full_sequnce_diff_vals(frame_dir)
    clusters = sbd.shot_frame_clustering(full_sequence_diffs)
    keyframe_indices = keyframes_from_clusters(clusters, epsilon, delta)

    print("len keyframe indices: {}".format(len(keyframe_indices)))

    return keyframe_indices



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

    



            




    