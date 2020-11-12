# Authors: Garrison Ramey,
# Date: 2020-11-02

import numpy as np
import shot_boundary_detection as sbd

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



if __name__ == "__main__":
    pass
    



            




    