#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:32:46 2025

@author: alex
"""
import sys
import init
import matplotlib.pyplot as plt
from pyutils import utils
from pyutils import file_select_ui as fsu
from sys_neuro_tools import math_utils
import json
import os
import h5py as h5
import numpy as np
import pandas as pd

#SLEAP Settings Functions
def get_settings_path(path = None):
    if path == None:
        path = utils.get_user_home()
    else:
        path = path
    return f"{path}/sleap_settings.json"

def load_sleap_settings(path=None):
    path = get_settings_path(path)
    data = None
    try:
        with open(path, "r") as file:
            data = json.load(file)
            return data
            if data == None:
                raise Exception("There is no data to read in the json file")
    except Exception as e:
        print(e)

def update_sleap_settings(path=None, new_model = False, change_python_loc = False, new_write_loc = False, json_exists=False, changed_script_loc=False):
    path = get_settings_path(path)
    data = {}
    if json_exists:
        #Read current path settings
        try:
            with open(path, "r") as file:
                data = json.load(file)
                print(data)
        except Exception as e:
            print(e)
        #Make necessary edits
        #Always asks for a new video
        print("Adding new video")
        data["vid_path"] = fsu.GetFile("Select Video Location")
        print("Select Analysis File Write Directory")
        #Find where the last backslash and period are found to extract the name
        slash_idx = data["vid_path"].rindex("/")
        dot_idx = data["vid_path"].rindex(".")
        hdf5_file_name = data["vid_path"][slash_idx + 1 : dot_idx] + "_labels.hdf5"
        if new_write_loc:
            data["write_dir"] = fsu.GetDirectory("Select Analysis File Write Directory") + "/" + hdf5_file_name
        else:
            data["write_dir"] = data["write_dir"][:data["write_dir"].rfind("/")] + "/" + hdf5_file_name#input("Name file(no .hdf5): ") + ".hdf5"   
        if changed_script_loc:
            data["script_loc"] = fsu.GetFile("Select sleap_utils_env.py")
        if new_model:
            model_type = input("Select Model Type (1: single animal 2: centroid centered): ")
            if model_type == "1":
                print("Select Single Instance Model Parent Directory")
                data["single_path"] =  fsu.GetDirectory("Select Single Instance Model Parent Directory")
                data["centroid_path"] = "NoFile"
                data["center_path"] = "NoFile"
            else:
                #Print because directory dialogues dont have titles
                data["single_path"] =  "NoFile"
                print("Select the Centroid Model Parent Directory")
                data["centroid_path"] = fsu.GetDirectory("Select the Centroid Model Parent Directory")
                print("Select the Center Model Parent Directory")
                data["center_path"] = fsu.GetDirectory("Select the Center Model Parent Directory")
        if change_python_loc:
            data["sleap_python"] = fsu.GetFile("Select SLEAP Python Location")
            #data["sleap_python"] = fsu.GetDirectory("Select SLEAP Python Location") + "/python"
            print("Changing python location")
        #Push to file
        try:
            with open(path, "w") as file:
                json.dump(data, file)
        except Exception as e:
            print(e)
    else:
        #Add all entries
        data["vid_path"] = fsu.GetFile("Select Video Location")
        slash_idx = data["vid_path"].rindex("/")
        dot_idx = data["vid_path"].rindex(".")
        hdf5_file_name = data["vid_path"][slash_idx + 1 : dot_idx] + "_labels.hdf5"
        data["write_dir"] = fsu.GetDirectory("Select Analysis File Write Directory") + "/" + hdf5_file_name
        model_type = input("Select Model Type (1: single animal 2: centroid centered): ")
        if model_type == "1":
            print("Select Single Instance Model Parent Directory")
            data["single_path"] =  fsu.GetDirectory("Select Single Instance Model Parent Directory")
            data["centroid_path"] = None
            data["center_path"] = None
        else:
            #Print because directory dialogues dont have titles
            data["single_path"] =  None
            print("Select the Centroid Model Parent Directory")
            data["centroid_path"] = fsu.GetDirectory("Select the Centroid Model Parent Directory")
            print("Select the Center Model Parent Directory")
            data["center_path"] = fsu.GetDirectory("Select the Center Model Parent Directory")
        print("Select Analysis File Write Directory")
        print(data["write_dir"])
        data["sleap_python"] = fsu.GetFile("Select SLEAP Python Location")
        #data["sleap_python"] = fsu.GetDirectory("Select SLEAP Python Location") + "/python"
        #Push to file
        try:
            with open(path, "w") as file:
                json.dump(data, file)
        except Exception as e:
            print(e)

#Data Processing Functions
def NodePositionsLocal(frame, processed_dict, origin_node="body", basis_node="neck", right_ortho=True):
    '''returns the node positions in a local coordinate system.
    The first bases vector is from the body to the neck.
    The second basis vector is orthogonal and on the right side of the body.
    ---
    Params: 
    processed_dict (result of processing hdf5 file)
    origin: the point (0,0)
    basis_node: the node used to define the first basis vector
    right_ortho: is the second basis vector on the right side of the body'''
    local_locations = np.zeros(np.shape(frame))
    #get the positions of the body, neck
    origin_idx = processed_dict["node_names"].index(origin_node)
    basis_idx = processed_dict["node_names"].index(basis_node)
    p_origin = np.array(frame[origin_idx])
    p_basis = np.array(frame[basis_idx])
    #Create basis vectors
    b1 = np.subtract(p_basis, p_origin)
    b2 = np.array([b1[1], -b1[0]]) 
    for n in range(len(frame)): 
        #Format the position to allow matrix multiplication
        #v = np.append(frame[n], 1)
        #v = np.reshape(v, (3, 1))
        centered_pos = frame[n] - p_origin
        #print((centered_pos + p_origin)[1] == frame[n][1])
        original_rot = math_utils.RotationMatrix(np.array([1,0]), b1)
        reformat_rot_matrix = np.reshape(original_rot, (original_rot.shape[0], original_rot.shape[1]))
        new_pos = np.dot(reformat_rot_matrix, centered_pos)
        local_locations[n][:] = np.squeeze(new_pos)
    return local_locations

def AngleToPorts(frame, processed_dict, port_pos_list):
    '''Returns the angle between the head-nose vector and all three ports
    ---
    Params:
    frame: the frame where the angles are being found
    processed_dict (result of processing hdf5 file)
    port_pos_list: a list of all the ports' positions (2xn numpy array)'''
    #KNOWN ERROR: When the nose disappears, the angle cannot be calculated
    #Use a.b = |a||b|cos(theta)
    angles = np.zeros((port_pos_list.shape[1], ))
    head_idx = processed_dict["node_names"].index("head")    
    nose_idx = processed_dict["node_names"].index("nose")
    p_head = frame[head_idx]
    p_nose = frame[nose_idx]
    nose_head_v = p_nose-p_head
    for c_idx in range(port_pos_list.shape[1]):
        port_pos = port_pos_list[:,c_idx]
        #Makes positions into column vectors for the next step
        port_pos = np.reshape(port_pos, (2,1)) 
        #Calculate vectors
        port_head_v = np.subtract(port_pos, p_head)
        #Use helper function to calculate angle b.w. two vectors
        angle = math_utils.Angle(nose_head_v, port_head_v)
        angles[c_idx] = angle
    return angles

def VelocityOutlierDetection(locations, z_score=2):
    '''Calculates differences in position over time and marks any suspiciously 
    rapid movement.
    
    Parameters
    ---
    locations: locations data from hdf5 file(frames x nodes x (x,y) x 1)
    confidence: percentage of data that should be found within
    
    Returns
    ---
    flags: same shape with 0 for normal and 1 for outliers'''
    mask = np.zeros(np.shape(locations))
    print("Checking for outliers")
    for n_idx in range(locations.shape[1]):
        print(n_idx)
        x_list = locations[:,n_idx,0,0]
        y_list = locations[:,n_idx,1,0]
        x_list = pd.Series(x_list)
        y_list = pd.Series(y_list)
        
        #print("num nan raw")
        #print(np.sum(np.isnan(x_list)))
        #print(np.sum(np.isnan(y_list)))
        
        x_list = x_list.interpolate(method='linear')
        y_list = y_list.interpolate(method='linear')
        
        #print("num nan lin")
        #print(np.sum(np.isnan(x_list)))
        #print(np.sum(np.isnan(y_list)))


        
        x_diff = np.diff(x_list)
        y_diff = np.diff(y_list)
        
        x_std = np.std(x_diff)
        y_std = np.std((y_diff))
        #print("std")
        #print(x_std)
        #print(y_std)
        
        x_mean = np.mean(x_diff)
        y_mean = np.mean(y_diff)
        #print("Mean")
        #print(x_mean)
        #print(y_mean)
        
        #Creates bounds num_std standard deviations above/below the mean
        #Currently uses z_scores for outlier detection
        #Could use 1.5*IQR in the future if it works better
        x_bounds = [x_mean-x_std*z_score, x_mean+x_std*z_score]
        y_bounds = [y_mean-y_std*z_score, y_mean+y_std*z_score]
        
        #Checks those bounds at all points
        for frame_idx in range(1,locations.shape[0]):
            mask[frame_idx, n_idx, 0,0] = (x_diff[frame_idx-1] < x_bounds[0] or x_diff[frame_idx-1] > x_bounds[1])
            mask[frame_idx, n_idx, 1,0] = (y_diff[frame_idx-1] < y_bounds[0] or y_diff[frame_idx-1] > y_bounds[1])
    plt.imshow(mask[:,:,0,0], aspect='auto')
    plt.colorbar()
    plt.show()
    plt.imshow(mask[:,:,1,0], aspect='auto')
    plt.colorbar()
    plt.show()
    return mask

#Animal Plotting Functions
def PlotPorts(port_pos_list):
    for c_idx in range(port_pos_list.shape[1]):
        port_pos = port_pos_list[:,c_idx]
        plt.scatter(port_pos[0], port_pos[1], label=f"port {c_idx + 1}")
        
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
    plt.gca().invert_yaxis() #Images have 0,0 at top left and positive down
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

def PlotLocalPosNode(processed_dict, node_name, local_pos_list):
    print(np.shape(local_pos_list[:,processed_dict["node_names"].index(node_name)]))
    maxInd = len(local_pos_list)-1
    randomPoints = np.random.randint(low = 0, high = maxInd, size = (30,))
    for rp in randomPoints:
        coordinates = local_pos_list[rp, processed_dict["node_names"].index(node_name), :]
        plt.scatter(coordinates[0], coordinates[1])
    plt.xlabel("Body Axis Position (pixels)")
    plt.ylabel("Right Axis Position (pixels)")
    plt.suptitle(node_name)
    plt.show();
        
def PlotAnglesToPorts(angles, port_pos_list, offset, degrees=True):
    if len(angles) != port_pos_list.shape[1]:
        print("Error : Number of ports and angles do not match")
        return
    if degrees:
        for i in range (len(angles)):
            angles[i] = np.rad2deg(angles[i])
    for c_idx in range(port_pos_list.shape[1]):
        port_pos = port_pos_list[:,c_idx]
        if degrees:
            plt.text(port_pos[0]+offset[0], port_pos[1]+offset[1], f"{round(angles[c_idx],2)} degrees")
        else:
            plt.text(port_pos[0]+offset[0], port_pos[1]+offset[1], f"{round(angles[c_idx],2)} rad")


#h5 Formatting Functions
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
    with h5.File(filename, "r") as f:
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

def view_hdf5_data(unpacked_hdf5):
    '''Takes the dictionary given by proces_hdf5_data and prints out keys and shape
    ---
    Params: 
    unpacked_hdf5: the return value of process_hdf5_data (Or any dictionary)
    ---
    Returns: None
    ---
    Prints: The key and then shape for entries indexed by that key
    '''
    for key in unpacked_hdf5:
        print(key)
        print(np.shape(unpacked_hdf5[key]))

def LocationToDataframe(locations_data, node_names):
    df = pd.DataFrame()
    for name in node_names:
        df[f"{name}_x"] = []
        df[f"{name}_y"] = []
    #print(df)
    for frame_idx in range(len(locations_data)):
        row_data = np.zeros((df.shape[1],))
        for n in range(len(locations_data[frame_idx])):
            #print(locations_data[frame_idx][n][0])
            #print(locations_data[frame_idx,n,0,:])
            row_data[2*n] = float(locations_data[frame_idx,n,0])
            row_data[2*n+1] = float(locations_data[frame_idx,n,1])
        df.loc[len(df)] = row_data
    return df

def ReformatToOriginal(df):
    '''Calculates differences in position over time and marks any suspiciously 
    rapid movement.
    
    Parameters
    ---
    locations: locations data from hdf5 file(frames x nodes x (x,y) x 1)
    
    Returns
    ---
    flags: same shape with 0 for normal and 1 for outliers'''
    num_nodes = (len(df.iloc[0])-5)//2 #4 components + cluster
    locations = np.zeros((len(df),num_nodes, 2))
    node_names = []
    for f_idx in range(len(df)):
        for n_idx in range(num_nodes):
            locations[f_idx, n_idx, 0] = df.iloc[f_idx][df.columns.values[n_idx*2]]
            locations[f_idx, n_idx, 1] = df.iloc[f_idx][df.columns.values[n_idx*2 + 1]]
            node_names.append(df.columns.values[n_idx*2][:-2])
    return {"node_names" : node_names,
            "locations" : locations}
