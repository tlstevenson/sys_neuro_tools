#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:40:38 2025

@author: alex
"""

import init
import subprocess
from pyutils import file_select_ui as fsui

#Use the file selecion to get python with keypoint installed
python_path = fsui.GetFile("Select keypoint moseq python file.")
#Run the main file in a subprocess 
project_dir = fsui.GetDirectory("Select Project Directory")
sleap_file = fsui.GetFile("Select Analysis File")
video_file = fsui.GetFile("Select Video File")
subprocess.run([python_path, "keypoint_main.py", project_dir, sleap_file, video_file])