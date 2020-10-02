# Problem Statement

### As the amount of video content available and consumed on social media platforms has grown exponentially in the past decade, so has the necessity for classifying and summarizing videos with high accuracy. A key tool for real-time video processing is keyframe extraction. Keyframe extraction seeks to search for frames that best represent the content of each shot or best represent the scene in the video clip [1]. The main processing of keyframe extraction is shot segmentation. Shot segmentation involves detecting the transition between successive slots.
### The goal of this project is to survey and analyze two state-of-the-art keyframe extraction techniques on multiple public video datasets and compare their performance.

# Approach

### First, we will implement a detection-based keyframe extraction algorithm based on combination of two features: one is Pearson correlation coefficient (PCC) and other is color moments (CM). The linear transformation invariance property of PCC facilitates the proposed algorithm to work well under varying lighting conditions. On the other hand, the scale and rotation invariance properties of color moments are beneficial for representation of complex objects that may be present in different poses and orientation.

### The second method we will be using is non-sequential, meaning that instead of directly comparing consecutive frames to detect a keyframe, the frames will first be clustered into different shots (a collection of frames). This will be done by classifying blocks within a frame as edge or non-edge blocks. Then, additional measurements will be computed based on the classification of each block within a frame. Once all frames have been processed, we will cluster the frames according to certain threshold values to obtain shot boundaries for the image. Finally, keyframes will be extracted from each cluster of frames based on threshold values given by Priya and Domnic [3], and we may also experiment with our own threshold criteria to improve our results.


# Experiments and Results
### In order to compare the effectiveness of keyframe extraction by exploiting Pearson correlation coefficients and color moments versus edge-aware clustering, we will start by reading the implementation details given by Bommisetty et al. [2] and Priya and Dominic [3]. Without using any existing code, we will create our own implementations of the two techniques by following the pseudo-code given in each paper. If we find it necessary to compute fidelity measures, then we may utilize pre-existing shot boundary detection programs. 

### Because the evaluation of how well a keyframe represents a shot is subjective, we plan to use the Vimeo Creative Commons (V3C1) dataset that consists of 32 videos that have associated keyframe annotations. We will use the keyframe annotations in the dataset as the ground truth keyframes in order to compute the accuracy, recall, precision, and figure of merit (F-score) score for each method. 

### To set up our experiment, we will start by choosing two videos at random from all available videos. Then, we will choose one video as input for our keyframe extraction implementation that exploits Pearson correlation coefficients and color moments. Once we have the resulting keyframes, we will compute the compression ratio, accuracy, recall, precision, F-score, and then save each frame for visual comparison purposes. Next, we will test our edge-aware clustering implementation by following the same procedure. After the video processing is complete, we will have a total of four figures for visual comparison and four figures for quantitative comparison.

### We expect our experiments to show successful keyframe selection by producing visual representations that accurately summarize each video and good quantitative results. Because the methods proposed by Bommisetty et al. [2] and Priya and Dominic [3] achieved high scores for accuracy, precision, and recall, we expect to see similar results. However, because we will be implementing the techniques ourselves and using a different dataset for testing, we are highly uncertain about the outcomes, so our results may vary greatly from those achieved by Bommisetty et al. [2] and Priya and Dominic [3].


# Bibliography

### [1] S. Pandey, P. Dwivedy, S. Meena and A. Potnis, "A survey on key frame extraction methods of a MPEG video," 2017 International Conference on Computing, Communication and Automation (ICCCA), Greater Noida, 2017, pp. 1192-1196, doi: 10.1109/CCAA.2017.8229979.

### [2] Bommisetty, R. M., Prakash, O., & Khare, A. (2019). Keyframe extraction using Pearson correlation coefficient and color moments. Multimedia Systems, 1-33.

### [3] G.L. Priya, S. Domnic Shot based keyframe extraction for ecological video indexing and retrieval Ecological Inform., 23 (2014), pp. 107-117.


