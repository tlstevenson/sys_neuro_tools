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
import pandas as pd
import seaborn as sns

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
    print(predictions)
    #predictions = predictor.predict(imgs)
    #predictions.export(write_path)
    return predictions

def GetPose(frame_idx):
    pose_1 = [710, 2525, 102091, 115703, 13613] #center
    pose_2 = [16463, 20419, 36543, 74867, 81673] #left straight
    pose_3 = [1, 16497, 21781, 61255, 68061] #left tilted
    pose_4 = [54449, 1681, 3795, 56552, 78731] #right straight
    pose_5 = [2300, 2452, 47643, 6807, 1710] #right tilted
    pose_6 = [5275, 74238, 74299, 82780, 84006] #rearing
    pose_7 = [27225, 27255, 30435, 122597, 122509, 1157] #grooming
    pose_8 = [30413, 34031, 40837, 88479, 95285, 56276] #idling
    
    vid_idx_corrected = frame_idx+1
    if vid_idx_corrected in pose_1:
        return "center"
    elif vid_idx_corrected in pose_2:
        return "left_s"
    elif vid_idx_corrected in pose_3:
        return "left_t"
    elif vid_idx_corrected in pose_4:
        return "right_s"
    elif vid_idx_corrected in pose_5:
        return "right_t"
    elif vid_idx_corrected in pose_6:
        return "rearing"
    elif vid_idx_corrected in pose_7:
        return "grooming"
    elif vid_idx_corrected in pose_8:
        return "idling"
    else:
        print(f"Error: Frame {vid_idx} not found in poses")

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

vid_path = "/Users/alex/Documents/HanksLab/TestWorkflow/Test_0001_reformat.mp4"
single_path = "/Users/alex/Downloads/250729_123240.single_instance.n=67"#"NoFile"
centroid_path = "NoFile"#"/Users/alex/Documents/Hanks Lab/TestWorkflow/250219_122558.centroid.n=696"
centered_path = "NoFile"#"/Users/alex/Documents/Hanks Lab/TestWorkflow/250219_150141.centered_instance.n=696"
write_path = "/Users/alex/Documents/HanksLab/TestWorkflow/Model2ValidationPredictions.hdf5"
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

plot_res=False
eval_limit = 30 #Make sure this agrees with the circle drawing line

if plot_res:
    if len(test_labels) != len(validation_labels):
        print("MISMATCHED VALIDATION LABELS AND PREDICTIONS")
    for idx in range(len(validation_labels)):
        if len(validation_labels[idx].instances) > 0:
            vid_idx = validation_labels[idx].frame_idx
            print(vid_idx)
            fig, ax = plt.subplots()
            ax.set_title(GetPose(vid_idx))
            ax.imshow(np.squeeze(video_obj.get_frames(vid_idx)))
            sleap.nn.viz.plot_instances(test_labels[idx].instances, cmap = sns.color_palette("flare"))
            sleap.nn.viz.plot_instances(validation_labels[idx].instances)
            #print(validation_labels[idx].instances[0].points)
            print(f"The frame {vid_idx} has {len(validation_labels[idx].instances)} instances")
            for point in validation_labels[idx].instances[0].points:
                circle = CirclePolygon((point.x, point.y), eval_limit, fill=False)
                plt.gca().add_patch(circle)
            plt.show()
        else:
            print(f"The frame {test_labels[idx].frame_idx} has {len(validation_labels[idx].instances)} instances")
            
print("Manual names: ")
print(validation_labels[0].instances[0].skeleton.node_names)
print(f"Number of test labels: {len(test_labels)}")
for i in range(len(test_labels)):
    print(f"Labeled frame {i} number of instances: {len(test_labels[i].instances)}")
print(f"Auto names: {test_labels[0].instances[0].skeleton.node_names}")
#print(test_labels[0].instances[0].skeleton.node_names)

#Plotting code for other analysis
#Index 0 shown as frame 1 so any index needs to have +1 for frame #
#score_df_rows = [] #Stores each frame before concatenation
column_names = [name for name in validation_labels[0].instances[0].skeleton.node_names]
column_names.append("pose")
score_df = pd.DataFrame(columns=column_names)
print(score_df)

for idx in range(len(validation_labels)):
    if len(validation_labels[idx].instances) > 0:
        #Gets frame in the video and sets up dataframe columns
        vid_idx = validation_labels[idx].frame_idx
        node_correct = []
        #column_names = [0 for i in range(len(validation_labels[idx].instances[0].skeleton.node_names) + 1)]
        
        #Iterate through nodes and calculate if they pass the distance test
        for n in range(len(validation_labels[idx].instances[0].skeleton.node_names)):
            test_node_name = validation_labels[idx].instances[0].skeleton.node_names[n]
            val_node_name = validation_labels[idx].instances[0].skeleton.node_names[n]
            #Manual: tail_end Test: tail_tip
            if (validation_labels[idx].instances[0].skeleton.node_names[n] == 'tail_end'):
                test_node_name = 'tail_tip'
                #test_node_idx = test_labels[idx].instances[0].skeleton.node_names.index('tail_tip')
                #test_node_idx = test_labels[idx].instances[0].skeleton.node_names.index(test_node_name)
            #print(f"The current node is {test_node_name}")
            valid_label = False
            try:
                test_point=test_labels[idx].instances[0][test_node_name]
                val_point=validation_labels[idx].instances[0][val_node_name]
                print("Successfully got both points")

                test_loc=np.array([test_point.x, test_point.y])
                val_loc=np.array([val_point.x, val_point.y])
                print("Successfully setup point vectors")
                
                diff = val_loc-test_loc
                print(f"Distance: {np.sqrt(np.dot(diff, diff))}")
                valid_label=(np.sqrt(np.dot(diff, diff)) < eval_limit)
            except:
                print("Could not calculate distance (probably unidentified point)")
            node_correct.append(valid_label)
            #column_names[n] = validation_labels[idx].instances[0].skeleton.node_names[n]
        #column_names[-1] = "pose"
        new_row=None
        
        #Determine which pose this was
        node_correct.append(GetPose(vid_idx))
        score_df.loc[vid_idx] = node_correct
        #new_row = pd.DataFrame([node_correct], columns=column_names, index=[vid_idx])
        #score_df_rows.append(new_row)
    else:
        print("Instance not found.")
        vid_idx = validation_labels[idx].frame_idx
        rows = np.squeeze([False for i in range(len(score_df.columns))]) 
        #Determine which pose this was
        rows[-1] = GetPose(vid_idx)
        score_df.loc[vid_idx] = rows
        print("Entered Empty Row")
            
#score_df = pd.concat(score_df_rows)
#score_dict[(vid_idx, validation_labels[idx].instances[0].skeleton.node_names[n])] = valid_label
#Plot percentage correct for each pose as a bar graph
poses = ["center", "left_s", "left_t", "right_s", "right_t", "rearing", "grooming", "idling"]
fig, ax = plt.subplots()
total_scores = []
for pose in poses:
    pose_df = score_df[score_df["pose"] == pose]
    pose_scores=[]
    for idx, row in pose_df.iterrows():
        count_correct = 0
        total = len(row.values)-1 #Subtract 1 for the pose column
        for val in row.values:
            if val == True:
                count_correct = count_correct + 1
        pose_scores.append(count_correct/total)
    print(pose_scores)
    total_scores.append(np.average(pose_scores))
ax.bar(poses, total_scores)
ax.set_title(f"Percent Correct By Pose: {eval_limit} pixel radiius")
plt.show()

#Plot percentage for each node as a bar graph
poses = ["center", "left_s", "left_t", "right_s", "right_t", "rearing", "grooming", "idling"]
fig_node, ax_node = plt.subplots()
total_scores = []
for col in score_df.columns:
    if col != "pose":
        score = len(score_df[score_df[col] == True]) / len(score_df[col])
        total_scores.append(score)
ax_node.bar(score_df.columns[:-1], total_scores) #+1 for pose -1 for starting at 1
ax.set_title(f"Percent Correct By Node: {eval_limit} pixel radiius")
plt.show()