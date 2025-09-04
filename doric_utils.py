# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 08:52:32 2022

Library provided by Doric to load a .doric file

@author: ING57
"""

import h5py
import numpy as np
import pyutils.utils as utils
from scipy.stats import mode
from collections import Counter

def ish5dataset(item):
    return isinstance(item, h5py.Dataset)


def h5printR(item, leading = '', print_attr=False):
    for key in item:
        if ish5dataset(item[key]):
            print(leading + key + ': ' + str(item[key].shape))
            if print_attr:
                for a in item[key].attrs:
                    print(' ' + leading + a + ': ' + repr(item[key].attrs[a]))
        else:
            print(leading + key)
            if print_attr:
                for a in item[key].attrs:
                    print(' ' + leading + a + ': ' + repr(item[key].attrs[a]))
            h5printR(item[key], leading + '  ', print_attr)


# Print structure of a .doric file
def h5print(filename, print_attr=False):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ', print_attr)


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
                     [{'Name': k,
                       'Data': np.array(item[key][k]),
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

def get_specific_data(filename, data_path, signal_name_dict):
    ''' Helper method to grab specific data from the doric file and return a flattened dictionary of the data '''

    specific_data = {}
    # keep track of previously loaded data to not duplicate i/o work
    loaded_data = {}

    with h5py.File(filename, 'r') as f:
        for key, signal_dict in signal_name_dict.items():
            value_dict = {}
            for value_name, value_path in signal_dict.items():
                # if data has already been loaded, just copy it
                if value_path in loaded_data:
                    value_dict[value_name] = loaded_data[value_path].copy()
                elif (data_path + value_path) in f:
                    value_dict[value_name] = np.array(f[data_path + value_path])
                    loaded_data[value_path] = value_dict[value_name].copy()

            if len(value_dict) > 0: 
                specific_data[key] = value_dict

    return specific_data

def get_flattened_data(filename, signals_of_interest=[]):
    ''' Helper method to flatten the acquired data into a more readily useable format '''
    dr_data = ExtractDataAcquisition(filename)
    flat_data = {}

    for signal in dr_data:
        # get signal name
        signal_name = signal['Name'].split('_')[-1]

        if len(signals_of_interest) > 0 and not any([soi in signal_name for soi in signals_of_interest]):
            continue

        signal_data = {}
        for entry in signal['Data']:
            # save values
            signal_data[entry['Name']] = entry['Data']

        flat_data[signal_name] = signal_data

    return flat_data

def fill_missing_data(flat_data, time_key = 'time'):
    ''' Checks the doric data for any issues with skipped data points,
        and corrects any skipped timepoints by putting NaNs for the values
        at the skipped timepoints.
        Returns the corrected data and a description of any issues that were found '''

    issues = []
    #skipped_idxs = {}
    ts_skipped_thresh = 0.01

    # check timstamps for any skipped information
    for name in flat_data.keys():
        time = flat_data[name][time_key]
        ts_diffs = np.diff(time)

        # find the timestep difference that is closest to the rest of the differences
        unique_diffs = np.unique(np.round(ts_diffs, 8))
        diff_close_counts = [sum(np.isclose(d, ts_diffs, atol=2e-6, rtol=0)) for d in unique_diffs]
        best_diff = unique_diffs[np.argmax(diff_close_counts)]
        close = np.isclose(best_diff, ts_diffs, atol=2e-6, rtol=0)
        dt = np.round(np.mean(ts_diffs[close]), 8)

        if not all(close):
            idxs = np.where(~close)[0]+1
            # round to nearest decimal because timestamps can be off by a multiple of a step, based on the decimation
            ts_skipped = utils.convert_to_multiple(ts_diffs[~close]/dt, 0.001)-1

            #skipped_idxs[name] = {'idxs': idxs, 'skipped steps': ts_skipped}

            signal_names = [k for k in flat_data[name].keys() if k != time_key]

            # adjust the timestamps and add in Nans where necessary
            idx_offset = 0

            for idx, n_skip in zip(idxs, ts_skipped):

                # if we skipped one timestep or less, this is because of a bug with writing data to file
                # where the data is written at the next cycle, but the timestamp still increments,
                # so we need to reduce the rest of the timestamps by the skipped amount
                if n_skip <= 1:
                    if abs(n_skip) < ts_skipped_thresh:
                        continue
                    else:
                        time[idx+idx_offset:] -= dt*n_skip

                else: # we skipped multiple steps, so need to insert nans

                    # see whether there were any partial steps skipped as well
                    sub_skips = n_skip % 1
                    full_skips = int(n_skip)

                    # update signals
                    for sig_name in signal_names:
                        flat_data[name][sig_name] = np.insert(flat_data[name][sig_name],
                                                             idx+idx_offset,
                                                             np.full(full_skips, np.nan))

                    # update time array
                    time = np.insert(time, idx+idx_offset,
                                     time[idx+idx_offset-1] + dt*np.arange(1, full_skips+1))

                    idx_offset += full_skips

                    # account for partial skips
                    if sub_skips != 0 and abs(sub_skips) > ts_skipped_thresh:
                        time[idx+idx_offset:] -= dt*sub_skips

            # log issues
            skip_counts = Counter(ts_skipped)
            issues.append('Signal {0} had {1} total time skips:\n{2}'.format(
                name, np.sum(ts_skipped),
                '\n'.join(['  {0} step(s): {1} time(s)'.format(k,v) for k,v in skip_counts.items()])))

        # make sure time array is evenly spaced by the dt
        # account for any initial offset of the first point that is less than dt
        offset = time[0] % dt
        time = utils.convert_to_multiple(time-offset, dt) + offset

        if np.sum(~np.isclose(dt, np.diff(time), atol=2e-6, rtol=0)) != 0:
            raise ValueError('Time array for signal {} could not be evenly spaced by dt: {}'.format(name, dt))

        flat_data[name][time_key] = time


    return flat_data, issues
