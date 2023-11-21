# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 08:52:32 2022

Library provided by Doric to load a .doric file

@author: ING57
"""

import h5py
import numpy as np
import acq_utils as acq
import pyutils.utils as utils
import pickle

def ish5dataset(item):
    return isinstance(item, h5py.Dataset)


def h5printR(item, leading = ''):
    for key in item:
        if ish5dataset(item[key]):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a .doric file
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')


def h5read(filename, where):
    '''
    Reads specific information from doric file

    Examples:
    SignalIn, SignalInInfo = dr.h5read(filename,['DataAcquisition','FPConsole','Signals','Series0001','AnalogIn','AIN01']);
    TimeIn, TimeInInfo = dr.h5read(filename,['DataAcquisition','FPConsole','Signals','Series0001','AnalogIn','Time']);
    '''

    data = []
    with h5py.File(filename, 'r') as h:
        item = h
        for w in where:
            if ish5dataset(item[w]):
                data = np.array(item[w])
                DataInfo = {atrib: item[w].attrs[atrib] for atrib in item[w].attrs}
            else:
                item = item[w]

    return data, DataInfo


def h5getDatasetR(item, leading = ''):
    r = []
    for key in item:
        # First have to check if the next layer is a dataset or not
        firstkey = list(item[key].keys())[0]
        if ish5dataset(item[key][firstkey]):
            r = r+[{'Name':leading+'_'+key, 'Data':
                                                [{'Name': k, 'Data': np.array(item[key][k]),
                                                    'DataInfo': {atrib: item[key][k].attrs[atrib] for atrib in item[key][k].attrs}} for k in item[key]]}]
        else:
            r = r+h5getDatasetR(item[key], leading + '_' + key)

    return r


# Extact Data from a doric file
def ExtractDataAcquisition(filename):
    ''' Reads all acquired data from a doric file '''
    with h5py.File(filename, 'r') as h:
        #print(filename)
        return h5getDatasetR(h['DataAcquisition'], filename)

# Additional Functions Written by Tanner

def get_flattened_data(filename):
    ''' Helper method to flatten the acquired data into a more readily useable format '''
    dr_data = ExtractDataAcquisition(filename)
    flat_data = {}

    for signal in dr_data:
        signal_data = {}

        # get default name, if no better name found
        signal_name = signal['Name'].split('_')[-1]

        for entry in signal['Data']:
            # save values
            signal_data[entry['Name']] = entry['Data']

            # get a better name, if there is only one signal and a time signal
            if len(signal['Data']) == 2:
                if 'Username' in entry['DataInfo']:
                    signal_name = entry['DataInfo']['Username']
                elif 'Name' in entry['DataInfo']:
                    signal_name = entry['DataInfo']['Name']

        flat_data[signal_name] = signal_data

    return flat_data

def get_and_check_data(filename):
    ''' Checks the doric data for any issues, and returns issue descriptions if found'''

    data = get_flattened_data(filename)

    issues = []

    # check timstamps for any skipped information
    for name in data.keys():
        ts_diffs = np.diff(data[name]['Time'])
        close = np.isclose(ts_diffs[0], ts_diffs)

        if not all(close):
            idxs = np.where(~close)[0]
            issues.append('There may be timestep skips in signal {0} at {1}'.format(
                name, ', '.join([str(i) for i in idxs])))

    return data, issues