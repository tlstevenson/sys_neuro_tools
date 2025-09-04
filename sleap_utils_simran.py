# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:49:46 2025

@author: hankslab
"""

import sleap
sleap.disable_preallocation()
sleap.versions()
sleap.system_summary()

from sleap import Labels, Video
import h5py
import numpy as np
import pandas as pd
import ephys_utils
from scipy.signal import convolve


#this may not be required
def create_sleap_project(video_path, project_path):
    labels = Labels()
    video = Video.from_filename(video_path)
    labels.add_video(video)
    
    labels.save(project_path)
    
    return labels


def run_inference_and_export(video_path, centroid_model_path, instance_model_path, output_labels_path, hdf5_output_path):
   
    predictor = sleap.load_model([centroid_model_path, instance_model_path])
    
    video = Video.from_filename(video_path)
    
    predictions = predictor.predict(video)
    predictions.save(output_labels_path)
    predictions.export(hdf5_output_path)
    
    return hdf5_output_path


def get_hdf5_data(filename):
    
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        
        frame_count, node_count, _, instance_count = locations.shape
        
        return {'locations shape': locations.shape, 
                'node names': node_names, 
                'dset_names': dset_names, 
                'locations': locations}


def get_smoothed_body_positions(body_positions, n_frames=9):
    
    kernel = ephys_utils.get_filter_kernel(n_frames, 'gauss', 1)
    weights = kernel['weights']

    if len(weights) % 2 == 0:
        pad_width = int(len(weights) / 2)
    else:
        pad_width = int((len(weights) - 1) / 2)
    
    n_dims = len(body_positions.shape)
    n_pad = [(pad_width, pad_width)]
    
    for i in range(n_dims-1):
        n_pad.append((0,0))
        weights = np.expand_dims(weights, axis=-1)
        
    padded_body_positions = np.pad(body_positions, n_pad, 'edge')

    smoothed_body_positions = convolve(padded_body_positions, weights, mode='valid', method='direct')
    
    return smoothed_body_positions


def calculate_vectors(smoothed_node_locations, port_locations):
    
    num_frames = smoothed_node_locations.shape[0]
    
    head_nose_vectors = np.zeros((num_frames, 2))

    for frame in range(num_frames):
        head_position = smoothed_node_locations[frame, 6, :]
        nose_position = smoothed_node_locations[frame, 0, :]
        
        head_nose_vectors[frame] = nose_position - head_position

            
    r = np.sqrt(head_nose_vectors[:,0]**2 + head_nose_vectors[:,1]**2)
    theta = np.arctan2(head_nose_vectors[:,1], head_nose_vectors[:,0])

    head_nose_polar = np.zeros((head_nose_vectors.shape[0], 2))
    head_nose_polar[:,0] = r
    head_nose_polar[:,1] = theta

    head_ports_vectors = np.zeros((num_frames, 3, 2))
    num_ports = port_locations.shape[1]

    for frame in range(num_frames):  
        head_position = smoothed_node_locations[frame, 6, :]
        
        for port in range(num_ports):
            port_position = port_locations[frame, port, :]
            
            head_ports_vectors[frame, port] = port_position - head_position
            
    head_ports_polar = np.zeros((num_frames, num_ports, 2))

    for frame in range(num_frames):
        for port in range(num_ports):
            x, y = head_ports_vectors[frame, port, :]
            d = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            head_ports_polar[frame, port, 0] = d
            head_ports_polar[frame, port, 1] = theta
            
    return {'head_nose_vectors': head_nose_vectors, 
            'head_nose_polar': head_nose_polar, 
            'head_ports_vectors': head_ports_vectors, 
            'head_ports_polar': head_ports_polar}


def create_df(node_locations_data, port_locations_data, smoothed_node_locations, port_locations, head_nose_polar, head_ports_polar):
    
    head_nose_thetas = head_nose_polar[:,1]
    head_ports_thetas = head_ports_polar[:,:,1]

    theta_right = head_ports_thetas[:,0] - head_nose_thetas
    theta_center = head_ports_thetas[:,1] - head_nose_thetas
    theta_left = head_ports_thetas[:,2] - head_nose_thetas


    nodes_ports_data = {}

    for i, node in enumerate(node_locations_data['node names'][:11]):
        nodes_ports_data[f'{node} x'] = smoothed_node_locations[:, i, 0]
        nodes_ports_data[f'{node} y'] = smoothed_node_locations[:, i, 1]
        
    for i, port in enumerate(port_locations_data['node names']):
        nodes_ports_data[f'{port} x'] = port_locations[:, i, 0]
        nodes_ports_data[f'{port} y'] = port_locations[:, i, 1]
        
    nodes_ports_df = pd.DataFrame(nodes_ports_data)

    theta_d_data = { 
        'theta left': theta_left,
        'd left': head_ports_polar[:,0,0],
        'theta center': theta_center,
        'd center': head_ports_polar[:,1,0],
        'theta right': theta_right,
        'd right': head_ports_polar[:,2,0]
        }

    theta_d_df = pd.DataFrame(theta_d_data)

    nodes_ports_df = pd.concat([nodes_ports_df, theta_d_df], axis=1)
    
    return nodes_ports_df

