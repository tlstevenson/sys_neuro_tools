#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:48:46 2025

@author: alexandru
"""
#activate sleap before running
import sys
import sleap 
import json
    
#sys_neuro_tools
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
    imgs = video[:10]
    print(f"imgs.shape: {imgs.shape}")

    # Predict on the array.
    predictions = predictor.predict(imgs)
    predictions.export(write_path)

sleap_settings_path = sys.argv[1]
with open(sleap_settings_path, "r") as file:
    sleap_settings = json.load(file)
    vid_path = sleap_settings["vid_path"]
    single_path = sleap_settings["single_path"]
    centroid_path = sleap_settings["centroid_path"]
    centered_path = sleap_settings["center_path"]
    write_path = sleap_settings["write_dir"]
    RunInference(vid_path, single_path, centroid_path, centered_path, write_path)
"""vid_path = sys.argv[1]
single_path = sys.argv[2]
centroid_path = sys.argv[3]
centered_path = sys.argv[4]
write_path = sys.argv[5]
RunInference(vid_path, single_path, centroid_path, centered_path, write_path)"""