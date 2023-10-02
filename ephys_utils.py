# -*- coding: utf-8 -*-
"""
Set of functions to organize and manipulate spiking data

@author: tanner stevenson
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import pyutils.utils as utils
import warnings


def get_trial_spike_times(spike_times, trial_start_times):
    '''
    Organizes spike times by trial

    Parameters
    ----------
    spike_times : List of spike times
    trial_start_times : List of trial start times

    Returns
    -------
    A list of numpy arrays containing the spike times within each trial relative to the start of each trial
    '''

    # make sure the spike times are a numpy array for logical comparisons
    spike_times = np.array(spike_times)
    trial_spike_times = []

    for i in range(len(trial_start_times)):
        if i < len(trial_start_times)-1:
            spike_select = np.logical_and(spike_times > trial_start_times[i], spike_times < trial_start_times[i+1])
        else:
            spike_select = spike_times > trial_start_times[i]

        trial_spike_times.append(spike_times[spike_select] - trial_start_times[i])

    # convert to pandas series for easier usage
    return pd.Series(trial_spike_times)


def get_binned_spike_counts(spike_times, start_time=0, end_time=np.inf, bin_width=5e-3):
    '''
    Gets binned spike counts for the given spike times between the start and end time points with the given bin width

    Parameters
    ----------
    spike_times : List of spike times
    start_time : (optional) The start time of the bins. The default is 0.
    end_time : (optional) The end time of the bins. The default is one bin width beyond the last spike time
    bin_width : (optional) The width of the bins. The default is 5e-3.

    Returns
    -------
    counts : An array of spike counts within each bin
    bin_edges : The bin edges. Will have one more element than the counts
    '''

    # handle default end time
    if np.isinf(end_time):
        end_time = spike_times[-1]

    # make sure the end time will be included in the bins
    bin_edges = np.arange(start_time, end_time+bin_width, bin_width)
    counts, _ = np.histogram(spike_times, bin_edges)

    return counts, bin_edges


def get_filter_kernel(width=0.2, filter_type='half_gauss', bin_width=5e-3):
    '''
    Gets a dictionary with entries that contain information used by filtering routines

    Parameters
    ----------
    width : (optional) The width of the filter in seconds. The default is 0.2 s.
    filter_type : (optional) The type of filter. Acceptable values: 'avg', 'causal_avg', 'gauss', 'half_gauss', 'exp', and 'none'.
        The default is 'half_gauss'.
    bin_width : (optional) The width of the bin in seconds. The default is 5e-5 s.

    Returns
    -------
    A dictionary with entries:
        type : the filter type
        weights : the weights used when filtering
        bin_width : the width of the filter bin
        center_idx : the index of the center of the filter
    '''

    # average filter with window centered on the current bin
    if filter_type == 'avg':
        window_limit = utils.convert_to_multiple(width/2, bin_width)
        x = np.arange(-window_limit, window_limit, bin_width)
        weights = np.ones_like(x)

    # causal average filter that only considers prior signal with equal weights
    elif filter_type == 'causal_avg':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width, bin_width)
        x = np.arange(0, window_limit, bin_width)
        weights = np.ones_like(x)

    # gaussian filter with max centered on current bin
    elif filter_type == 'gauss':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width/2, bin_width)
        x = np.arange(-window_limit, window_limit, bin_width)
        weights = norm.pdf(x, 0, window_limit/4)

    # causal filter that only considers prior signal with gaussian weights
    elif filter_type == 'half_gauss':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width, bin_width)
        x = np.arange(0, window_limit, bin_width)
        weights = norm.pdf(x, 0, window_limit/4)

    # causal filter that only considers prior signal with exponentially decaying weights
    elif filter_type == 'exp':
        # set the window limit to a multiple of bin width
        window_limit = utils.convert_to_multiple(width, bin_width)
        x = np.arange(0, window_limit, bin_width)
        weights = np.exp(-x*4/window_limit)

    # no filter
    elif filter_type == 'none':
        x = 0
        weights = 1

    else:
        raise ValueError('Invalid filter type. Acceptable types: avg, causal_avg, gauss, half_gauss, exp, and none')

    return {'type': filter_type,
            'weights': weights/np.sum(weights),  # normalize sum to one
            'bin_width': bin_width,
            'center_idx': np.where(x == 0)[0][0]}


def get_smoothed_firing_rate(spike_times, kernel=None, start_time=0, end_time=np.inf):
    '''
    Will calculate a smoothed firing rate based on the spike times between the given start and end times

    Parameters
    ----------
    spike_times : List of spike times
    kernel : (optional) A kernel dictionary from get_filter_kernel. Defaults to a half-gaussian of 0.2 s
    start_time : (optional) The start time of smoothed signal. The default is 0.
    end_time : (optional) The end time of the smoothed signal. The default is the last spike time

    Returns
    -------
    signal : A smoothed firing rate with points separated by the bin width specified by the kernel
    time : The time values corresponding to the signal values
    '''

    if kernel is None:
        kernel = get_filter_kernel()

    bin_width = kernel['bin_width']

    if np.isinf(end_time):
        end_time = utils.convert_to_multiple(spike_times[-1]-start_time, bin_width) + start_time

    # compute buffers around the start and end times to include spikes that should be included in the filter
    # shift them by half a bin width to make the resulting time have a value at t=0
    pre_buff = (len(kernel['weights']) - kernel['center_idx'] - 1) * bin_width + bin_width/2
    post_buff = (kernel['center_idx']) * bin_width + bin_width/2

    # compute signal and smooth it with a filter
    signal, bin_edges = get_binned_spike_counts(spike_times, start_time-pre_buff, end_time+post_buff, bin_width)
    signal = signal/bin_width
    signal = np.convolve(signal, kernel['weights'])
    # signal = convolve(signal, kernel['weights']) # scipy version might be faster, need to check

    # remove extra bins created from filtering
    filter_pre_cutoff = len(kernel['weights']) - 1
    filter_post_cutoff = len(kernel['weights']) - 1
    signal = signal[filter_pre_cutoff:-filter_post_cutoff]

    time = np.arange(start_time, end_time+bin_width, bin_width)

    return signal, time


def get_psth(spike_times, align_times, window, kernel=None, mask_bounds=None):
    '''
    Will calculate a peri-stimulus time histogram (PSTH) of the average firing rate aligned to the specified alignment points

    Parameters
    ----------
    spike_times : A list of spike times, either a single list or a list of spike times separated by trial (N trials)
    align_times : A list of N event times to align the firing rates.
        Can be a single list or a list of lists if there are multiple (K*) alignment points per trial.
        Time is relative to start of the trial
    kernel : The smoothing filter kernel
    window : The window (pre, post) around the alignment points that define the bounds of the psth.
        Time is relative to alignment point
    mask_bounds : (optional) The boundaries (pre, post) per trial past which any signal should be removed before averaging.
        Either a Nx2 matrix of boundaries or a list of N K*x2 matrices if there are K* alignment points per trial.
        Time is relative to start of the trial.

    Returns
    -------
    A dictionary with entries:
        signal_avg : The smoothed average signal
        signal_se : The signal standard error
        time : The time vector corresponding to the signal
        all_signals : All smoothed signals
    '''

    ## Handle multiple forms of inputs and check dimensions ##

    if kernel is None:
        kernel = get_filter_kernel()

    # handle pandas series in input
    # resetting index will make the indices reset to 0 based
    if isinstance(spike_times, pd.Series):
        spike_times = spike_times.reset_index(drop=True)

    if isinstance(align_times, pd.Series):
        align_times = align_times.reset_index(drop=True)

    n_trials = len(spike_times)

    # check the number of trials matches up
    if len(align_times) != n_trials:
        raise ValueError('The number of alignment points ({0}) does not match the number of trials ({1})'.format(
            len(align_times), n_trials))

    # handle the mask bounds
    has_mask = not mask_bounds is None
    if has_mask:
        # convert a pandas data frame to a numpy array
        if isinstance(mask_bounds, pd.DataFrame):
            mask_bounds = mask_bounds.to_numpy()

        # convert a list of two lists to a numpy array
        if isinstance(mask_bounds, list):
            mask_bounds = np.hstack([np.array(bounds).reshape(-1, 1) for bounds in mask_bounds])

        # check dimensions on the mask bounds
        if isinstance(mask_bounds, np.ndarray):
            # check there is a start and end to the mask
            if mask_bounds.shape[1] != 2:
                raise ValueError('The mask bounds must have start and end times in separate columns. Instead found {0} columns.'.format(
                    mask_bounds.shape[1]))

        # check the number of trials matches up
        if len(mask_bounds) != n_trials:
            raise ValueError('The number of mask bounds ({0}) does not match the number of trials ({1})'.format(
                len(mask_bounds), n_trials))

    ## Perform aligning and smoothing ##

    _, time = get_smoothed_firing_rate([], kernel, window[0], window[1])

    if has_mask:
        time_bin_edges = np.append(time - kernel['bin_width']/2, time[-1] + kernel['bin_width']/2)

    all_signals = []
    aligned_spikes = []

    for i in range(n_trials):
        trial_spike_times = np.array(spike_times[i])
        trial_align_times = align_times[i]

        # cast to list to allow for generic handling of one or multiple alignment points
        if utils.is_scalar(trial_align_times):
            trial_align_times = [trial_align_times]

        if has_mask:
            # make the mask a 2d array
            trial_mask_bounds = np.array(mask_bounds[i]).reshape(-1, 2)

            # make sure number of mask bounds is the same as number of alignment points
            if trial_mask_bounds.shape[0] != 1 and trial_mask_bounds.shape[0] != len(trial_align_times):
                raise ValueError('The number of trial mask bounds ({0}) does not match the number of alignment points ({1}) in trial {2}'.format(
                    trial_mask_bounds.shape[0], len(trial_align_times), i))

        for j, align_ts in enumerate(trial_align_times):
            if has_mask:
                if trial_mask_bounds.shape[0] == 1:
                    align_mask = trial_mask_bounds[0, :]
                else:
                    align_mask = trial_mask_bounds[j, :]

                # ignore alignment points outside of the mask
                if align_ts < align_mask[0] or align_ts > align_mask[1]:
                    continue

            offset_ts = trial_spike_times - align_ts
            signal, _ = get_smoothed_firing_rate(offset_ts, kernel, window[0], window[1])
            signal_spikes = offset_ts[np.logical_and(offset_ts > window[0], offset_ts < window[1])]

            # mask the signal
            if has_mask:

                # find mask indices
                mask_start = align_mask[0] - align_ts
                mask_end = align_mask[1] - align_ts
                mask_start_idx = np.argmax(time_bin_edges > mask_start)
                mask_end_idx = np.argmax(time_bin_edges > mask_end) - 1

                # mask with nans
                if mask_start_idx > 0:
                    signal[0:mask_start_idx] = np.nan
                if mask_end_idx < len(signal)-1:
                    signal[mask_end_idx:] = np.nan

                signal_spikes[np.logical_or(signal_spikes < mask_start, signal_spikes > mask_end)] = np.nan

            all_signals.append(signal)
            aligned_spikes.append(signal_spikes)

    # convert all signals list to matrix
    all_signals = np.array(all_signals)

    # ignore warnings that nanmean throws if all values are nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # compute average and standard error
        return {'signal_avg': np.nanmean(all_signals, axis=0),
                'signal_se': utils.stderr(all_signals),
                'time': time,
                'all_signals': all_signals,
                'aligned_spikes': aligned_spikes}


def get_fr_matrix_by_trial(spike_ts, trial_start_ts, kernel=None, trial_bounds=None, trial_select=None):
    '''
    Takes a list of spike times for each unit in a single session along with the trial start timestamps
    and outputs a pandas series of smoothed firing rate matrices for all units (T timesteps x N units)
    in each trial, optionally limited to a specified set of bounds on a per-trial basis.

    Parameters
    ----------
    spike_ts : A list of spike timestamps for each unit in the session
    trial_start_ts: A list of trial start timestamps for the session
    kernel : (optional) The filter kernel. Defaults to a half-gaussian of width 0.2 s
    trial_bounds : (optional) The trial bounds. Defaults to the whole trial
    trial_select : (optional) A boolean list indicating which trials should be included. Defaults to all trials

    Returns
    -------
    Returns a pandas series of smoothed firing rate matrices for all units (T timesteps x N units) organized by trial.
    '''

    n_trials = len(trial_start_ts)

    if trial_bounds is None:
        # compute the ends of the trials as the beginning of the next trial
        trial_ends = trial_start_ts[1:] - trial_start_ts[:-1]
        trial_ends = np.append(trial_ends.reshape(-1, 1), np.inf)
        trial_bounds = np.concatenate((np.zeros_like(trial_ends), trial_ends), axis=1)
    else:
        # convert a pandas data frame to a numpy array
        if isinstance(trial_bounds, pd.DataFrame):
            trial_bounds = trial_bounds.to_numpy()

        # check dimensions on the mask bounds
        if isinstance(trial_bounds, np.ndarray):
            # check there is a start and end to the mask
            if trial_bounds.shape[1] != 2:
                raise ValueError('The trial bounds must have start and end times in separate columns. Instead found {0} columns.'.format(
                    trial_bounds.shape[1]))

        # check the number of trials matches up
        if len(trial_bounds) != n_trials:
            raise ValueError('The number of trial bounds ({0}) does not match the number of trials ({1})'.format(
                len(trial_bounds), n_trials))

    # handle the trial select
    if trial_select is None:
        trial_select = [True] * n_trials
    else:
        # check the number of trials matches up
        if len(trial_select) != n_trials:
            raise ValueError('The number of trial selects ({0}) does not match the number of trials ({1})'.format(
                len(trial_select), n_trials))

    # go through trials and build each matrix of smoothed firing rates
    frs_by_trial = [np.array([get_smoothed_firing_rate(spike_ts[j] - trial_start_ts[i], kernel, trial_bounds[i, 0], trial_bounds[i, 1])[0]
                              for j in range(len(spike_ts))]).T
                    for i in range(n_trials) if trial_select[i]]

    frs_by_trial = pd.Series(frs_by_trial)

    return frs_by_trial
