#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:48:23 2025

@author: alex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Angle(v1, v2):
    return np.arccos(np.dot(np.transpose(v1),v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def TranslationMatrix(point, inverted=False):
    '''Returns a matrix that translates a vector by point
    ---
    Params: 
    point: a vector to be added or subtracted from other points
    inverted: if True it subtracts point instead of adding
    ---
    Returns: Matrix that translates by point
    '''
    ans = None
    if(inverted):
        ans = [[1,0,float(-point[0])],
               [0,1,float(-point[1])]]
    else:
        ans = [[1,0,float(point[0])],
               [0,1,float(point[1])]]
    return np.array(ans)

def BasisChangeMatrix(basis1, basis2):
    #Basis vectors must be normalized to prevent stretching
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    #basis vectors must be 2x1
    if np.shape(basis1) != (2,1):
        basis1=np.transpose(basis1)
    if np.shape(basis2) != (2,1):
        basis2=np.transpose(basis2)
    #Create change of basis matrix
    M = np.concatenate((basis1, basis2), axis=1) 
    M_old = [[1,0],[0,1]]
    change_of_basis_matrix = np.linalg.inv(M) @ M_old 
    return change_of_basis_matrix

def RotationMatrix(original_x_vect, new_x_vect):
    #Check to see if over or un
    print("Hi")
    cross_product = original_x_vect[0]*new_x_vect[1]-original_x_vect[1]*new_x_vect[0]
    #+ is rotated counterclockwise (Corrected clockwise to local)
    #- is rotated clockwise (Corrected counterclockise to local)
    angle = Angle(original_x_vect, new_x_vect)
    if cross_product > 0:
        return np.array([[np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]])
    
def GetDiff(vect, plot=False):
    '''Returns and plots the differences between neighboring entries of an array.
    Parameters
    ---
    vect: the vector to take a derivative of
    plot: should the returned vector be plotted? default:True
    Returns
    ---
    vect_diff: an array of differences between values (one less entry than vect)'''
    #Reshapes vect if it has a second dimension of one
    vect=np.reshape(vect, (np.shape(vect)[0],))
    print(np.shape(vect))
    #Removes nan by interpolation
    vect = pd.Series(vect)
    vect = vect.interpolate(method='linear')
    vect_diff = np.diff(vect)
    
    #Plots the vector if needed
    if plot:
        plt.plot(range(len(vect_diff)), vect_diff)
        plt.show()
        print(np.mean(vect))
        print(np.std(vect))
    return vect_diff
    
