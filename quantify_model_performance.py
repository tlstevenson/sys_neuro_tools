#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:26:47 2025

@author: alex
"""
import sleap
import matplotlib.pyplot as plt
from matplotlib.patches import CirclePolygon
import numpy as np
import h5py
#Get the labels file
validation_label_path="/Users/alex/ValidationData.slp"
#Read the labels file and get labeled frames
validation_labels=sleap.Labels.load_file(validation_label_path).labeled_frames
#Check that none of the labeled frames were used to train (Optional but not really)
#Run inference on those labeled frames
def RunInference(vid_path, single_path, centroid_path, centered_path, write_path, vid_idxs):
    '''Given a video, model location, and writing directory, runs inference and
    writes the analysis hdf5 file to the write_path
    
    Parameters 
    ---
    vid_path: location of the video
    centroid_path: location of the directory containing centroid model
    centered_path: location of the directory containing center model
    write_path: location to save the analysis file
    vid_idxs: indices of the video to label'''
    predictor = None
    if single_path != "NoFile":
        predictor = sleap.load_model([single_path], batch_size=16)
    else:
        predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    video = sleap.load_video(vid_path).get_frames(vid_idxs)
    print(video.shape, video.dtype)

    # Load frames
    #imgs = np.array([video[idx,:,:,:] for idx in vid_idxs])
    #print(f"imgs.shape: {imgs.shape}")

    # Predict on the array.
    predictions = predictor.predict(video)
    #predictions = predictor.predict(imgs)
    predictions.export(write_path)
    return predictions

def PlotSkeleton(frame, processed_dict, skeleton_color="black", nodes_mark=[], ax=None):
    '''Uses processed dict from process_hdf5_data to visualize the skeleton
    in matplotlib figure. DOES NOT SHOW GRAPH AUTOMATICALLY (use plt.show()).
    ---
    Params: 
    frame: the current frame in the video that needs to be plotted
    processed_dict: result of processing hdf5 file
    skeleton_color: color of the edges
    nodes_mark: boolean mask for which nodes may be outliers
    ---
    Returns: None
    '''
    #plt.gca().invert_yaxis() #Images have 0,0 at top left and positive down
    for node_idx in range(len(frame)):
        if len(nodes_mark) != 0  and nodes_mark[node_idx] == 1:
            node_color="red"
        else:
            node_color=skeleton_color
        if ax != None:
            ax.scatter(frame[node_idx,0], frame[node_idx,1], color=node_color)
        else:
            plt.scatter(frame[node_idx,0], frame[node_idx,1], color=node_color)
        offset_x=20
        if skeleton_color == "black" or ((skeleton_color=="green") and (nodes_mark[node_idx] == 1)):
            if ax != None:
                ax.text(frame[node_idx,0]+offset_x, frame[node_idx,1],processed_dict["node_names"][node_idx])
            else:
                plt.text(frame[node_idx,0]+offset_x, frame[node_idx,1],processed_dict["node_names"][node_idx])
        #plt.legend()

    for edge_ind in processed_dict["edge_inds"]:
        #Get the first and second indices and use them to get the x position
        x = [frame[edge_ind[0]][0], frame[edge_ind[1]][0]]
        #Get the first and second indices and use them to get the y position
        y = [frame[edge_ind[0]][1], frame[edge_ind[1]][1]]
        #Plot the current edge
        if ax != None:
            ax.plot(x,y,color=skeleton_color)
        else:
            plt.plot(x, y, color = skeleton_color)

vid_path = "/Users/alex/Documents/Hanks Lab/TestWorkflow/Test_0001_reformat.mp4"
single_path = "NoFile"
centroid_path = "/Users/alex/Documents/Hanks Lab/TestWorkflow/250219_122558.centroid.n=696"
centered_path = "/Users/alex/Documents/Hanks Lab/TestWorkflow/250219_150141.centered_instance.n=696"
write_path = "/Users/alex/Documents/Hanks Lab/TestWorkflow/ModelValidationPredictions.hdf5"
vid_idxs = [label.frame_idx for label in validation_labels]
print(vid_idxs)

def process_hdf5_data(filename):
    '''Returns a 'processed dictionary with information from the hdf5 file
    ---
    Params: filename (path for the hdf5 analysis file)
    ---
    Returns: dict with  the following keys
        locations  shape: shape of the locations data
        node_names: name of the nodes in the order they are found in locations
        dset_names: all the keys in original file
        locations: position data (#frames x #nodes x 2{x,y})
    '''
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        edge_inds = [edgeInd for edgeInd in f["edge_inds"]]
        
        frame_count, node_count, _, instance_count = locations.shape

        return {'locations_shape': locations.shape, 
                'node_names': node_names, 
                'dset_names': dset_names, 
                'locations': locations,
                'edge_inds': edge_inds}
use_saved_predictions = False
test_labels=None
predictions_dict=None
if use_saved_predictions:
    #test_labels=sleap.Labels.load_file(write_path)
    predictions_dict=process_hdf5_data(write_path)
else:
    test_labels = RunInference(vid_path, single_path, centroid_path, centered_path, write_path, vid_idxs).labeled_frames
#Overlay image, the two skeletons, circles around the nodes for each label
video_obj = sleap.load_video(vid_path)

if len(test_labels) != len(validation_labels):
    print("MISMATCHED VALIDATION LABELS AND PREDICTIONS")
for idx in range(len(test_labels)):
    vid_idx = test_labels[idx].frame_idx
    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(video_obj.get_frames(vid_idx)))
    sleap.nn.viz.plot_instances(test_labels[idx].instances)
    sleap.nn.viz.plot_instances(validation_labels[idx].instances)
    print(validation_labels[idx].instances[0].points)
    for point in validation_labels[idx].instances[0].points:
        circle = CirclePolygon((point.x, point.y), 30, fill=False)
        plt.gca().add_patch(circle)
    plt.show()
#for labeled_frame in test_labels:
    #Adds image
    #frame_pic = video_obj
    #ax.imshow(frame_pic)
    #plt.show()
    #Plots the manual and automatic skeletons
    #labeled_frame.plot()x
    
    #PlotSkeleton(validation_labels[labeled_frame.frame_idx], ax=ax)
    #PlotSkeleton(test_labels[labeled_frame.frame_idx], ax=ax)
    #Plots circles of correct radii around manual labels
    #for key, value in radii:
    #    node_location = FIX[labeled_frame.frame_idx, key]
    #    circle = CirclePolygon(node_location, radii[key])
    #    ax.add_patch(circle)
#Plot percentage for each node as a bar graph
#Plot percentage correct for each pose as a bar graph