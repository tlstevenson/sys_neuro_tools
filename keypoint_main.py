#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:43:48 2025

@author: alex
"""

import sys
import keypoint_moseq as kpms
import matplotlib.pyplot as plt
import os
# Get the list of all files and directories
path = '/Users/alex/Documents/Hanks Lab/demo_project/videos/'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
# prints all files
print(dir_list)

#Creates a new project
#project_dir = sys.argv[1] + "/demo_project"
project_dir  = '/Users/alex/Documents/Hanks Lab/demo_project'

config = lambda: kpms.load_config(project_dir)

#sleap_file = sys.argv[2]
sleap_file  = '/Users/alex/Documents/Hanks Lab/demo_project/videos/'
kpms.setup_project(project_dir, sleap_file=sleap_file)

#Update if needed
kpms.update_config(
    project_dir,
    video_dir='/Users/alex/Documents/Hanks Lab/demo_project/videos/',
    #anterior_bodyparts=["nose"],
    #posterior_bodyparts=[""],
    #use_bodyparts=["spine4", "spine3", "spine2", "spine1", "head", "nose", "right ear", "left ear"],
    fps=30
)

# load data from sleap
#keypoint_data_path = sys.argv[3] # can be a file, a directory, or a list of files
#keypoint_data_path = '/Users/alex/Documents/Hanks Lab/TestWorkflow'
#coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, "sleap")
coordinates, confidences, bodyparts = kpms.load_keypoints(filepath_pattern=sleap_file, 
                                                          format = "sleap",
                                                          extension='hdf5')


# format data for modeling
data, metadata = kpms.format_data(coordinates, confidences, **config())

#may need a seperate cell
kpms.noise_calibration(project_dir, coordinates, confidences, **config(), video_extension="mp4")

#plt.close('all')
#%matplotlib inline
#Fit PCA Model
pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

# use the following to load an already fit model
# pca = kpms.load_pca(project_dir)

# Use this to select how many PCs explain 90% variance
kpms.update_config(project_dir, latent_dim=7)

# initialize the model
model = kpms.init_model(data, pca=pca, **config())

# optionally modify kappa
model = kpms.update_hypparams(model, kappa=2000)

#Fit AR HHM
num_ar_iters = 50

model, model_name = kpms.fit_model(
    model, data, metadata, project_dir, ar_only=True, num_iters=num_ar_iters
)

# load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters
)

# modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=1e4)

# run fitting for an additional 500 iters
model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + 500,
)[0]

# modify a saved checkpoint so syllables are ordered by frequency
kpms.reindex_syllables_in_checkpoint(project_dir, model_name);

# load the most recent model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract results
results = kpms.extract_results(model, metadata, project_dir, model_name)

# optionally save results as csv
kpms.save_results_as_csv(results, project_dir, model_name)

#Generate plots of syllables
results = kpms.load_results(project_dir, model_name)
kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())

#Plot that shows the symilarity of different syllables
kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())

# %%
kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config());


