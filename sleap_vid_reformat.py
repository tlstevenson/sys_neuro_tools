# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:26:04 2025

@author: hankslab
"""
import os
import subprocess

def process_videos(directory):
    """
    Processes video files in a specified directory.
    
    Args:
    directory (str): The path to the directory containing video files.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if the file is a video (you can add more extensions if needed)
        if filename.endswith('.mp4'):
        # Skip the video if its name ends with '_r'
            if filename.endswith('_r.mp4') or filename.endswith('_r_l.mp4'):
                print(f"Skipping video: {filename} (ends with '_r')")
                continue

        # Check for a corresponding video ending with '_r'
        name_without_ext, ext = os.path.splitext(filename)
        r_file_path = os.path.join(directory, f"{name_without_ext}_r{ext}")
        if os.path.exists(r_file_path):
            print(f"Skipping video: {filename} (corresponding '{name_without_ext}_r{ext}' found)")
            continue
        # Define the output file path for the processed video
        output_file = os.path.join(directory, f"{name_without_ext}_r{ext}")

        # Define the FFmpeg command
        command = [
        'ffmpeg',
        '-y', # Overwrite output files without asking
        '-i', file_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'superfast',
        '-crf', '23',
        output_file
        ]

        print(f"Processing video: {filename}")
        try:
            # Run the FFmpeg command
            subprocess.run(command, check=True)
            print(f"Successfully processed {filename} -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename}: {e}")
        except FileNotFoundError:
            print("Error: FFmpeg is not installed or not in your system's PATH.")
            print("Please install FFmpeg to run this script.")
            break

# Example usage:
# Replace 'your_video_directory' with the path to your folder
video_directory = r'C:\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos'
process_videos(video_directory)