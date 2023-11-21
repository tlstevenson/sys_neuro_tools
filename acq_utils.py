# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:39:47 2023

@author: tanne
"""

import numpy as np
import pyutils.utils as utils

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
        event_idxs = np.where(ttl_diffs != 0)[0]
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
