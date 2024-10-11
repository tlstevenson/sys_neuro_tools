# -*- coding: utf-8 -*-
"""
Set of functions to organize and manipulate spiking data

@author: tanner stevenson
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import pyutils.utils as utils
from scipy.signal import convolve
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


def get_binned_spike_counts(spike_times, start_time=0, end_time=np.inf, bin_width=5e-3, align='start'):
    '''
    Gets binned spike counts for the given spike times between the start and end time points with the given bin width

    Parameters
    ----------
    spike_times : List of spike times
    start_time : (optional) The start time of the bins. The default is 0.
    end_time : (optional) The end time of the bins. The default is one bin width beyond the last spike time
    bin_width : (optional) The width of the bins. The default is 5e-3.
    align : (optional) How to force alignment of bins. Either to the start or end

    Returns
    -------
    counts : An array of spike counts within each bin
    bin_edges : The bin edges. Will have one more element than the counts
    sub_spike_times : The spike times that are located between the start and end times
    '''

    # handle default end time
    if np.isinf(end_time):
        end_time = spike_times[-1]

    spike_times = np.array(spike_times)
    sub_spike_times = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
    
    match align:
        case 'start':
            # make sure the end time will be included in the last bin
            # do this to account for rounding issues with arange
            new_end_time = utils.convert_to_multiple(end_time-start_time, bin_width) + start_time
            # don't do anything if the end time is just a rounding error away from the new end time
            if ~np.isclose(end_time, new_end_time):
                new_end_time = utils.convert_to_multiple(end_time-start_time, bin_width, direction='up') + start_time
                
            bin_edges = np.arange(start_time, new_end_time+bin_width, bin_width)
            # then make sure to remove any extra bins caused by rounding errors
            if end_time < bin_edges[-2] or np.isclose(end_time, bin_edges[-2]):
                bin_edges = bin_edges[:-1]

        case 'end':
            # make sure the start time will be included in the first bin
            # do this to account for rounding issues with arange
            new_start_time = end_time - utils.convert_to_multiple(end_time-start_time, bin_width)
            # don't do anything if the end time is just a rounding error away from the new end time
            if ~np.isclose(start_time, new_start_time):
                new_start_time = end_time - utils.convert_to_multiple(end_time-start_time, bin_width, direction='down')

            bin_edges = np.flip(np.arange(end_time, new_start_time-bin_width, -bin_width))
            # then make sure to remove any extra bins caused by rounding errors
            if start_time > bin_edges[1] or np.isclose(start_time, bin_edges[1]):
                bin_edges = bin_edges[1:]
            
    counts, _ = np.histogram(sub_spike_times, bin_edges)

    return counts, bin_edges, sub_spike_times


def get_filter_kernel(width=0.2, filter_type='half_gauss', bin_width=5e-3):
    '''
    Gets a dictionary with entries that contain information used by filtering routines

    Parameters
    ----------
    width : (optional) The width of the filter in seconds. The default is 0.2 s.
    filter_type : (optional) The type of filter. Acceptable values: 'avg', 'causal_avg', 'gauss', 'half_gauss', 'exp', and 'none'.
        The default is 'half_gauss'.
    bin_width : (optional) The width of the bin in seconds. The default is 5e-3 s.

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
        weights = np.array([1])

    else:
        raise ValueError('Invalid filter type. Acceptable types: avg, causal_avg, gauss, half_gauss, exp, and none')

    return {'type': filter_type,
            'weights': weights/np.sum(weights),  # normalize sum to one
            'bin_width': bin_width,
            'center_idx': np.where(x == 0)[0][0],
            't': x}


def get_smoothed_firing_rate(spike_times, kernel, start_time, end_time, align='start'):
    '''
    Will calculate a smoothed firing rate based on the spike times between the given start and end times

    Parameters
    ----------
    spike_times : List of spike times, or list of lists of spike times
    kernel : (optional) A kernel dictionary from get_filter_kernel. Defaults to a half-gaussian of 0.2 s
    start_time : (optional) The start time of smoothed signal. The default is 0.
    end_time : (optional) The end time of the smoothed signal. The default is the last spike time

    Returns
    -------
    signal : A smoothed firing rate with points separated by the bin width specified by the kernel
    time : The time values corresponding to the signal values
    '''

    bin_width = kernel['bin_width']

    # compute buffers around the start and end times to include spikes that should be included in the filter
    pre_buff = (len(kernel['weights']) - kernel['center_idx'] - 1) * bin_width
    post_buff = kernel['center_idx'] * bin_width
    
    # return time values as the center of the time bin
    time_bin_edges = get_binned_spike_counts([], start_time, end_time, bin_width, align=align)[1]
    time = time_bin_edges[:-1] + bin_width/2

    # compute signal and smooth it with a filter
    # determine if we need to convolve multiple trials at once
    if len(spike_times) > 0 and utils.is_list(spike_times[0]):
        signal = np.vstack([get_binned_spike_counts(times, start_time-pre_buff, end_time+post_buff, bin_width, align=align)[0][None,:] for times in spike_times])
        kernel_weights = kernel['weights'][None,:]
    else:
        signal = get_binned_spike_counts(spike_times, start_time-pre_buff, end_time+post_buff, bin_width, align=align)[0]
        kernel_weights = kernel['weights']
        
    signal = signal/bin_width
    signal = convolve(signal, kernel_weights)

    # remove extra bins created from filtering
    filter_pre_cutoff = len(kernel['weights']) - 1
    filter_post_cutoff = len(kernel['weights']) - 1
    if signal.ndim > 1:
        if filter_post_cutoff > 0:
            signal = signal[:, filter_pre_cutoff:-filter_post_cutoff]
        else:
            signal = signal[:, filter_pre_cutoff:]
            
        if signal.shape[1] != len(time):
            print('Mismatched dimensions found for smoothed signal and time')
    else:
        if filter_post_cutoff > 0:
            signal = signal[filter_pre_cutoff:-filter_post_cutoff]
        else:
            signal = signal[filter_pre_cutoff:]
            
        if len(signal) != len(time):
            print('Mismatched dimensions found for smoothed signal and time')
    
    return signal, time


def get_psth(spike_times, align_times, window, kernel=None, mask_bounds=None, align='start'):
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

    _, time = get_smoothed_firing_rate([], kernel, window[0], window[1], align=align)

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
            signal, _ = get_smoothed_firing_rate(offset_ts, kernel, window[0], window[1], align=align)
            signal_spikes = offset_ts[(offset_ts > window[0]) & (offset_ts < window[1])]

            # mask the signal
            if has_mask:

                # find mask indices
                mask_start = align_mask[0] - align_ts
                mask_end = align_mask[1] - align_ts
                mask_start_idx = np.argmax(time_bin_edges > mask_start)
                if any(time_bin_edges > mask_end):
                    mask_end_idx = np.argmax(time_bin_edges > mask_end) - 1
                else:
                    mask_end_idx = len(signal)

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


def get_fr_matrix_by_trial(spike_ts, trial_start_ts, kernel=None, trial_bounds=None, trial_select=None, align='start'):
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
    
    if isinstance(spike_ts, pd.Series):
        spike_ts = spike_ts.to_numpy()
    
    # make a list of lists for ease of use in the loop below
    if not len(spike_ts) > 0 or not utils.is_list(spike_ts[0]):
        spike_ts = [spike_ts]

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
    
    # make sure select is a numpy array
    trial_select = np.array(trial_select)

    # go through trials and build each matrix of smoothed firing rates aligned to trial start
    frs_by_trial = []
    time_by_trial = []
    for i in range(n_trials):
        if trial_select[i]:
            # get all unit spike times relative to start of trial
            trial_spikes = [ts - trial_start_ts[i] for ts in spike_ts]
            fr, t = get_smoothed_firing_rate(trial_spikes, kernel, trial_bounds[i, 0], trial_bounds[i, 1], align=align)
            frs_by_trial.append(fr.T)
            time_by_trial.append(t)

    # make a pandas series because making a np array throws an error because of inhomogeneous shapes
    return pd.Series(frs_by_trial), pd.Series(time_by_trial)

def get_avg_period_fr(spike_ts, period_bounds, period_select=None):
    '''
    Takes a list of spike times for each unit in a single session along with time period bounds and computes 
    the average firing rate within the bounds. Outputs a pandas series of average firing rate per unit.

    Parameters
    ----------
    spike_ts : A list of spike timestamps for each unit in the session
    period_bounds : The bounds defining the period to calculate average firing rate over

    Returns
    -------
    Returns a pandas series of period average firing rates for all units
    '''
    
    if isinstance(period_bounds, pd.DataFrame):
        period_bounds = period_bounds.to_numpy()

    # check dimensions on the period bounds
    if isinstance(period_bounds, np.ndarray):
        # check there is a start and end to the mask
        if period_bounds.shape[1] != 2:
            raise ValueError('The trial bounds must have start and end times in separate columns. Instead found {0} columns.'.format(
                period_bounds.shape[1]))
            
    n_periods = period_bounds.shape[0]
    n_units = len(spike_ts)
            
    # handle the trial select
    if period_select is None:
        period_select = [True] * n_periods
    else:
        # check the number of trials matches up
        if len(period_select) != n_periods:
            raise ValueError('The number of perios selects ({0}) does not match the number of periods ({1})'.format(
                len(period_select), n_periods))
            
    # make sure select is a numpy array
    period_select = np.array(period_select)

    period_durs = np.diff(period_bounds[period_select,:], axis=1)
    spike_counts = np.zeros((n_units, np.sum(period_select)))
        
    for i, spikes in enumerate(spike_ts):
        spike_counts[i, :] = np.array([np.nansum((spikes > period_bounds[j,0]) & (spikes < period_bounds[j,1])) for j in range(n_periods) if period_select[j]])
        
    avg_fr = np.nansum(spike_counts, axis=1)/np.nansum(period_durs)

    return avg_fr
