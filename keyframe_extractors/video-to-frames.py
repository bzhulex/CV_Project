# Author: Garrion Ramey
# Date: 2020-10-12
# Time: 5:53 PM EDT

import cv2
import sys
import numpy as np 
import imageio
import time
from argparse import ArgumentParser
from os import getcwd as pwd


def handle_args(args):
    target_vid_path = args.vp 
    save_dir = args.sd
    if target_vid_path == None or save_dir == None:
        print("You must provide a path to the video (-vp) and a" \
              " destination directory (-sd) to save the frames.\n" \
              "An argument of \".\" will be interpreted as your current working directory.\n" \
              "Example 1: python vtf.py -vp ./videos/video1.mp4 -sd . \n" \
              "Example 2: python vtf.py -vp ./video1.mp4 -sd ./output-frames")
        sys.exit()
    if target_vid_path[0] == ".":
        target_vid_path = pwd()
    if save_dir[0] == ".":
        save_dir = pwd()
    
    return target_vid_path, save_dir


if __name__ == "__main__":
    start = time.time()
    # Handle arguments
    ap = ArgumentParser()
    ap.add_argument("-vp", help="absolute path to the target video")
    ap.add_argument("-sd", help="path to directory to save video frames")
    args = ap.parse_args()
    target_vid_path, save_dir = handle_args(args)


    cap = cv2.VideoCapture(target_vid_path)
    i = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_file = "{}/frame-{}.png".format(save_dir, i)
        cv2.imwrite(frame_file, frame)
        i += 1
    
    cap.release()

    finish = time.time()
    print("Time elapsed: {}".format(finish - start))

    
        
    

