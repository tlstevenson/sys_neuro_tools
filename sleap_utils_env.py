#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:48:46 2025

@author: alexandru
"""
#activate sleap before running
import os
#os.environ["QT_API"] = 'pyside2'
import sys
import sleap 
import json

    
def RunInference(vid_path, single_path, centroid_path, centered_path, write_path):
    '''Given a video, model location, and writing directory, runs inference and
    writes the analysis hdf5 file to the write_path
    
    Parameters 
    ---
    vid_path: location of the video
    centroid_path: location of the directory containing centroid model
    centered_path: location of the directory containing center model
    write_path: location to save the analysis file'''
    predictor = None
    if single_path != "NoFile":
        predictor = sleap.load_model([single_path], batch_size=16)
    else:
        predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    video = sleap.load_video(vid_path)
    print(video.shape, video.dtype)

    # Load frames
    #imgs = video
    #print(f"imgs.shape: {imgs.shape}")

    # Predict on the array.
    predictions = predictor.predict(video)
    predictions.export(write_path)

def RunInferenceDir(video_dir, single_path, centroid_path, centered_path, write_dir):
    """
    Iterates through a directory and creates inference files for all videos
    titled _r that haven't been renamed to _r_l (reformatted and labeled).
    
    Args:
    video_dir (str): The path to the directory containing video files.
    single_path (str): The path to the single instance model
    centroid_path (str): The path to the centroid model
    centered_instance_path (str): The path to the centered instance model
    """
    # Ensure the directory exists
    if not os.path.isdir(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return
    if not os.path.isdir(write_dir):
        print(f"Error: Directory '{write_dir}' not found.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(video_dir):
        print(filename)
        # Construct the full file path
        file_path = os.path.join(video_dir, filename)
        print(file_path)
        # Check if the file is a video (you can add more extensions if needed)
        if filename.endswith('.mp4'):
            # Label the video if its name ends with '_r'
            if filename.endswith('_r.mp4'):
                print("The file is a video that needs to be analyzed")
                #Name the labels file video_r_labels.hdf5 and run inference
                name_without_ext, ext = os.path.splitext(filename)
                write_path = os.path.join(write_dir, f"{name_without_ext}_labels.hdf5")
                print(write_path)
                #Return resulted in error
                try:
                    RunInference(file_path, single_path, centroid_path, centered_path, write_path)
                    print(f"Predicting on video: {filename} (ends with _r)")
                    
                    #Rename the video file to show that it was processed (_r_l)
                    new_path = os.path.join(video_dir, f"{name_without_ext}_l{ext}")
                    os.rename(file_path, new_path)
                    print(f"Renaming video file {file_path} to {new_path}")
                except:
                    print("Failed to label the video: {filename}")
            else:
                print(f"Skipping video: {filename} (doesn't end with '_r')")
                continue

sleap_settings_path = sys.argv[1]
with open(sleap_settings_path, "r") as file:
    sleap_settings = json.load(file)
    vid_dir = sleap_settings["vid_dir"]
    single_path = sleap_settings["single_path"]
    centroid_path = sleap_settings["centroid_path"]
    centered_path = sleap_settings["center_path"]
    write_dir = sleap_settings["write_dir"]
    #RunInference(vid_dir, single_path, centroid_path, centered_path, write_dir)
    RunInferenceDir(vid_dir, single_path, centroid_path, centered_path, write_dir)
"""vid_path = sys.argv[1]
single_path = sys.argv[2]
centroid_path = sys.argv[3]
centered_path = sys.argv[4]
write_path = sys.argv[5]
RunInference(vid_path, single_path, centroid_path, centered_path, write_path)"""