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
                else:
                    value_dict[value_name] = np.array(f[data_path + value_path])
                    loaded_data[value_path] = value_dict[value_name].copy()

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

    # check timstamps for any skipped information
    for name in flat_data.keys():
        time = flat_data[name][time_key]
        ts_diffs = np.diff(time)
        close = np.isclose(mode(ts_diffs)[0], ts_diffs, atol=2e-6, rtol=0)

        if not all(close):
            dt = np.round(np.mean(ts_diffs[close]), 8)
            idxs = np.where(~close)[0]+1
            # round to nearest decimal because timestamps can be off by half a step
            ts_skipped = utils.convert_to_multiple(ts_diffs[~close]/dt, 0.5)-1

            signal_names = [k for k in flat_data[name].keys() if k != time_key]

            # add in Nans
            idx_offset = 0
            up_half_step = False
            prev_half_step_idx = 0

            for idx, n_skip in zip(idxs, ts_skipped):

                # if we skipped a whole number of steps
                if n_skip % 1 == 0:
                    n_skip = int(n_skip)

                    # if we are up a half step, we need to correct all the rest between the last skip and this
                    if up_half_step:
                        time[prev_half_step_idx:idx+idx_offset] += dt/2
                        prev_half_step_idx = idx + idx_offset + n_skip

                else: # we skipped a half step
                    if up_half_step: # if this is the second time we are skipping a half step
                        # correct all the rest between last skip and this
                        time[prev_half_step_idx:idx+idx_offset] += dt/2

                        # we already added the extra timestep, so subtract 0.5 from n_skips
                        n_skip = int(n_skip - 0.5)
                        up_half_step = False
                    else:
                        # add an extra timestep at the beginning of the skip
                        n_skip = int(n_skip + 0.5)
                        up_half_step = True

                        # update the previous half step index
                        prev_half_step_idx = idx + idx_offset + n_skip


                # update time array
                time = np.insert(time, idx+idx_offset,
                                 time[idx+idx_offset-1]+dt*np.arange(1,n_skip+1))
                # update signals
                for sig_name in signal_names:
                    flat_data[name][sig_name] = np.insert(flat_data[name][sig_name],
                                                         idx+idx_offset,
                                                         np.full(n_skip, np.nan))

                idx_offset += n_skip

            # if there were uneven numbers of half steps, shift remaining timestamps
            if up_half_step:
                time[prev_half_step_idx:] += dt/2

            # if the starting time is not a multiple of dt, shift all t by a half step
            if utils.convert_to_multiple(time[0]/dt, 0.5) % 1 != 0:
                time -= dt/2

            flat_data[name][time_key] = utils.convert_to_multiple(time, dt)

            if np.sum(~np.isclose(dt, np.diff(flat_data[name][time_key]), atol=2e-6, rtol=0)) != 0:
                raise ValueError('Missing data in signal {} could not be filled properly'.format(name))

            # log issues
            skip_counts = Counter(ts_skipped)
            issues.append('Signal {0} had {1} total time skips:\n{2}'.format(
                name, np.sum(ts_skipped),
                '\n'.join(['  {0} step(s): {1} time(s)'.format(k,v) for k,v in skip_counts.items()])))

    return flat_data, issues