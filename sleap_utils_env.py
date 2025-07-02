#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:48:46 2025

@author: alexandru
"""
#activate sleap before running
import sleap 

#sys_neuro_tools
def RunInference(vid_path, centroid_path, centered_path, write_path):
    '''Given a video, model location, and writing directory, runs inference and
    writes the analysis hdf5 file to the write_path
    
    Parameters 
    ---
    vid_path: location of the video
    centroid_path: location of the directory containing centroid model
    centered_path: location of the directory containing center model
    write_path: location to save the analysis file'''
    predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    video = sleap.load_video(vid_path)
    print(video.shape, video.dtype)

    # Load frames
    imgs = video
    print(f"imgs.shape: {imgs.shape}")

    # Predict on the array.
    predictions = predictor.predict(imgs)
    predictions.export(write_path)
    
    