# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:39:47 2023

@author: tanne
"""

import numpy as np
import pyutils.utils as utils
import math

def parse_trial_times(ttl_signal, t, pulse_width=0.001, n_trial_bits=15, signal_type='continuous'):
    '''
    Parses trial start times and numbers from the provided ttl signal.
    The structure of the ttl signal is a single pulse at the beginning of each trial
    followed by a number of binary bits denoting the trial number

    Parameters
    ----------
    ttl_signal : The ttl signal
    t : The timestamps corrseponding to the signal in seconds
    pulse_width : The width of each ttl puls in seconds
    n_trial_bits : The number of bits to encode the trial number
    signal_type : 'continuous' for a continuous signal where the ttl signal is either 1 or 0
                  'discrete' for a discrete signal where the ttl signal is only the changes (1 for rise, -1 for fall) and their associated timestamps

    Returns
    -------
    trial_start_ts : The timestamps associated with the start of each trial
    trial_nums : The decoded trial numbers associated with each trial timestamp

    '''

    # Verify dimensions are the same
    if len(ttl_signal) != len(t):
        raise ValueError('The length of the TTL signal ({0}) does not match the length of timestamps ({1})'.format(
            len(ttl_signal), len(t)))

    # convert a continuous signal into discrete events
    if signal_type == 'continuous':
        # get pulse edges
        ttl_diffs = np.concatenate(([0], np.diff(ttl_signal)))
        event_idxs = np.where((ttl_diffs != 0) & ~np.isnan(ttl_diffs))[0]
        ttl_signal = ttl_diffs[event_idxs]
        t = t[event_idxs]

    # make sure for every rise, there is a fall
    if sum(ttl_signal) != 0:
        raise ValueError('The number of TTL rises ({0}) does not match the number of falls ({1})'.format(
            sum(ttl_signal == 1), sum(ttl_signal == -1)))

    # make sure the rises and falls alternate
    if not all(abs(np.diff(ttl_signal)) == 2):
        raise ValueError('The TTL rises and falls did not alternate')

    # convert ttl rises and falls into binary representation
    ttl_diffs = np.diff(ttl_signal)
    ttl_binary = np.zeros_like(ttl_diffs)
    # if signal is high then low, which counts as 1 in binary, diff will be negative
    ttl_binary[ttl_diffs < 0] = 1

    # find time intervals between ttl events
    event_intervals = utils.convert_to_multiple(np.diff(t), pulse_width)

    # find trial starts and parse trial numbers from binary
    trial_start_ts = []
    trial_nums = []

    i = 0
    while i < len(event_intervals):
        # find first bit of each trial
        if event_intervals[i] == pulse_width and ttl_binary[i] == 1:
            trial_start_ts.append(t[i])

            # next parse binary
            i += 1
            trial_binary = np.zeros(n_trial_bits)
            bin_start = 0

            while i < len(event_intervals) and event_intervals[i] < n_trial_bits * pulse_width:
                # ttl can be high or low for multiple binary positions in a row
                # figure out how many positions this binary value is for
                bin_len = int(event_intervals[i] / pulse_width)
                trial_binary[bin_start : bin_start+bin_len] = ttl_binary[i]
                bin_start += bin_len
                i += 1

            binary_string = ''.join([str(int(b)) for b in trial_binary])
            trial_nums.append(int(binary_string, 2))
        else:
            i += 1

    return np.array(trial_start_ts), np.array(trial_nums)

def decimate_data(data, target_dt = None, target_sf = None, time_key = 'time', timestamp_pos = 'mid'):
    '''
    Perform data decimation. If there are multiple sets of timestamps among the signals,
    will consolidate all signals to use the same timestamps.

    Parameters
    ----------
    data : The data dictionary consisting of a set of data names that key a timestamp array and
        any number of signals associated with that set of timestamps
    target_dt : The target decimated timestep. Defaults to 5 ms
    target_sf : The target decimated sample rate. Defaults to 200 Hz
    time_key : The key for the timestamp array in each data entry. Default is 'time'.
    timestamp_pos : The position of the decimated timestamp:
        'begin' or 'beginning' for the start of the decimation interval
        'mid' or 'middle' for the middle of the decimation interval
        'end' for the end of the decimation interval
        The default is 'mid'.

    Returns
    -------
    dec_time : Vector of timestamps associated with the decimated signals
    dec_signals : Dictionary of decimated signals keyed by data name and signal name
    dec_info : dictionary of information about the decimation process

    '''
    # find the superset time array for signals of interest
    times = [data[key][time_key] for key in data.keys()]
    first_t = np.min([t[0] for t in times])
    last_t = np.max([t[-1] for t in times])
    time_size = [len(t) for t in times]
    time = times[np.argmax(time_size)].copy()
    dt = np.mean(np.diff(time))

    # mke sure the time array spans the entire length of time in the signals
    if not math.isclose(time[0], first_t):
        time = np.insert(time, 0, time[0]+dt*np.arange(utils.to_int((first_t-time[0])/dt), 0))

    if not math.isclose(time[-1], last_t):
        time = np.append(time, time[-1]+dt*np.arange(1, utils.to_int((last_t-time[-1])/dt)))

    # Extract all signals and make sure they are the same length as the time array
    signals = {}
    for name in data.keys():
        signal_t = data[name][time_key]

        signal_names = [k for k in data[name].keys() if k != time_key]

        for signal_name in signal_names:
            signal_vals = data[name][signal_name].copy()

            # see if need to append NaNs onto beginning or end of signal
            if not math.isclose(time[0], signal_t[0]):
                n_steps = utils.to_int((signal_t[0]-time[0])/dt)
                signal_vals = np.insert(signal_vals, 0, np.full(n_steps, np.nan))

            if not math.isclose(time[-1], signal_t[-1]):
                n_steps = utils.to_int((time[-1]-signal_t[-1])/dt)
                signal_vals = np.append(signal_vals, np.full(n_steps, np.nan))

            if len(signal_names) > 1:
                signals[name + '_' + signal_name] = signal_vals
            else:
                signals[name] = signal_vals

            if len(signal_vals) != len(time):
                raise ValueError('Signal values array could not be matched to the time array. Make sure missing data has been filled in')

    # decimate signals
    # calculate desired decimation factor
    if target_dt is None and target_sf is None:
        target_dt = 0.005
    elif target_dt is None:
        target_dt = 1/target_sf

    if target_dt > dt:
        decimation = utils.to_int(target_dt/dt)
    else:
        decimation = 1

    # instead of using signal processing decimation methods, where there is assumed to be a true underlying signal
    # simply average over the number of bins given by the decimation factor to get the downsampled data
    if decimation > 1:
        dec_signals = {}
        # first do signals
        for name, signal in signals.items():
            # reshape the array into rows where columns are the values to be averaged over for decimation
            # make the signal size divisible by the decimation factor
            signal = np.append(signal, np.full(decimation - (len(signal) % decimation), np.nan))
            reshape_signal = np.reshape(signal, (-1, decimation))
            dec_signals[name] = np.nanmean(reshape_signal, axis=1)
            signal_length = len(dec_signals[name])

        # then do the time stamps based on provided positioning variable
        match timestamp_pos:
            case 'begin' | 'beginning':
                start_t_offset = 0
            case 'mid' | 'middle':
                if decimation % 2 == 0:
                    start_t_offset = (decimation+1)/2 * dt
                else:
                    start_t_offset = decimation/2 * dt
            case 'end':
                start_t_offset = decimation * dt

        dec_time = time[0] + start_t_offset + (np.arange(signal_length)*decimation*dt)
    else:
        dec_signals = signals
        dec_time = time

    dec_info = {'decimation': decimation, 'timestamp_pos': timestamp_pos, 'initial dt': dt, 'decimated dt': decimation*dt}

    return dec_time, dec_signals, dec_info