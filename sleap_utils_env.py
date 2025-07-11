#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:48:46 2025

@author: alexandru
"""
#activate sleap before running
import sys
import sleap 
    
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
    if single_path != "None":
        predictor = sleap.load_model([single_path], batch_size=16)
    else:
        predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    video = sleap.load_video(vid_path)
    print(video.shape, video.dtype)

    # Load frames
    imgs = video
    print(f"imgs.shape: {imgs.shape}")

    # Predict on the array.
    predictions = predictor.predict(imgs)
    predictions.export(write_path)

sleap_settings_path = sys.argv[1]
with open(sleap_settings_path, "r") as file:
    vid_path = file["vid_path"]
    single_path = file["single_path"]
    centroid_path = file["centroid_path"]
    centered_path = file["centered_path"]
    write_path = file["write_path"]
    RunInference(vid_path, single_path, centroid_path, centered_path, write_path)
"""vid_path = sys.argv[1]
single_path = sys.argv[2]
centroid_path = sys.argv[3]
centered_path = sys.argv[4]
write_path = sys.argv[5]
RunInference(vid_path, single_path, centroid_path, centered_path, write_path)"""