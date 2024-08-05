# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:33:47 2023

@author: tanne
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

import warnings

def filter_signal(signal, cutoff_f, sr):
    '''
    Low-pass filters a signal with a zero-phase butterworth filter of the given cutoff frequency

    Parameters
    ----------
    signal : The signal to filter
    cutoff_f : The cutoff frequency
    sr : The sampling rate

    Returns
    -------
    The filtered signal

    '''

    sos = butter(2, cutoff_f, btype='lowpass', fs=sr, output='sos')
    denoised_signal = sosfiltfilt(sos, signal)

    return denoised_signal


def calc_iso_dff(lig_signal, iso_signal, t, vary_t=True):
    '''
    Calculates dF/F based on an isosbestic signal by regressing the isosbestic signal onto the ligand-dependent signal,
    then using this fit isosbestic signal as the baseline

    Parameters
    ----------
    lig_signal : The ligand-dependent signal
    iso_signal : The isosbestic signal

    Returns
    -------
    The isosbestic corrected dF/F signal
    The fitted isosbestic signal

    '''

    fitted_iso = fit_signal(iso_signal, lig_signal, t, vary_t)
    # calculate dF/F
    dff = ((lig_signal - fitted_iso)/fitted_iso)*100

    return dff, fitted_iso


def fit_signal(signal_to_fit, signal, t, vary_t=True):
    '''
    Scales and shift one signal to match another using linear regression

    Parameters
    ----------
    signal_to_fit : The signal being fitted
    signal : The signal being fit to

    Returns
    -------
    The fitted signal

    '''

    # # find all NaNs and drop them from both signals before regressing
    nans = np.isnan(signal) | np.isnan(signal_to_fit)

    # fit the iso signal to the ligand signal
    # reg = LinearRegression(positive=True)
    # reg.fit(signal_to_fit[~nans,None], signal[~nans])
    # fitted_signal = np.full_like(signal_to_fit, np.nan)
    # fitted_signal[~nans] = reg.predict(signal_to_fit[~nans,None])

    if vary_t:
        form = lambda x, a, b, c: a*x[0,:] + b*x[1,:] + c
        s_to_fit = np.vstack((signal_to_fit[None,~nans], t[None,~nans]))
        bounds = ([      0, -np.inf, -np.inf],
                  [ np.inf,  np.inf,  np.inf])
    else:
        form = lambda x, a, b: a*x + b
        s_to_fit = signal_to_fit[None,~nans]
        bounds = ([      0, -np.inf],
                  [ np.inf,  np.inf])

    params = curve_fit(form, s_to_fit, signal[~nans], bounds=bounds)[0]
    fitted_signal = np.full_like(signal_to_fit, np.nan)
    fitted_signal[~nans] = form(s_to_fit, *params)

    return fitted_signal


def fit_baseline(signal, n_points_min=10, baseline_form=None, bounds=None):
    '''
    Fits a baseline to the signal using a baseline formula equation.
    Defaults to a double exponential decay function:
        A*e^(-t/B) + C*e^(-t/(B*D)) + E
    Fits the baseline to the minimum value every n data points

    Parameters
    ----------
    signal : The signal to fit
    n_points_min: The number of data points to take the min over, optional

    Returns
    -------
    The fit baseline

    '''

    n_points_max = 2000

    if n_points_min > n_points_max:
        raise RuntimeError('Could not fit baseline. Try another formula or method.')

    if bounds is None:
        if baseline_form is None:
            bounds = ([-np.inf,      1, -np.inf, 0, -np.inf],
                      [ np.inf, np.inf,  np.inf, 1,  np.inf])
        else:
            bounds = (-np.inf, np.inf)

    if baseline_form is None:
        baseline_form = lambda x, a, b, c, d, e: a*np.exp(-x/b) + c*np.exp(-x/(b*d)) + e

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.simplefilter('error', category=OptimizeWarning)

        try:
            if len(signal) % n_points_min != 0:
                min_signal = np.append(signal, np.full(n_points_min - (len(signal) % n_points_min), np.nan))
            else:
                min_signal = signal

            # compute the minimum value every n points
            min_signal = np.nanmin(np.reshape(min_signal, (-1, n_points_min)), axis=1)

            # ignore nans in fit
            nans = np.isnan(min_signal)

            x = np.arange(len(min_signal))
            params = curve_fit(baseline_form, x[~nans], min_signal[~nans], bounds=bounds)[0]

            # return full baseline of the same length as the signal
            x = np.arange(len(signal))/n_points_min
            return baseline_form(x, *params)
        except (OptimizeWarning, RuntimeError):
            print('Baseline fit was unseccessful. Expanding the signal minimum window to {}..'.format(n_points_min*2))
            return fit_baseline(signal, n_points_min=n_points_min*2, baseline_form=baseline_form)


def build_signal_matrix(signal, ts, align_ts, pre, post, align_sel=[]):
    '''
    Build a matrix of signals aligned to the specified timestamp with the given window

    Parameters
    ----------
    signal : The entire signal
    ts : The timestamps associated with the signal
    align_ts : The alignment timestamps
    pre : The time pre alignment point to show
    post : The time post alignment point to show
    align_sel : Boolean selection of alignment points, optional

    Returns
    -------
    signal_mat : Matrix of signals where each row is the signal aligned to the specified alignment point
    t : The relative time array associated with the columns of the signal matrix

    '''

    if len(align_sel) > 0:
        align_ts = align_ts[align_sel]

    dt = np.mean(np.diff(ts))
    t = np.arange(-pre, post+dt, dt)
    center_idx = np.argmin(np.abs(t))
    pre_idxs = center_idx
    post_idxs = len(t) - center_idx

    signal_mat = np.full((len(align_ts), len(t)), np.nan)

    for i, align_t in enumerate(align_ts):

        if align_t is None or np.isnan(align_t):
            continue

        rel_ts = ts - align_t
        # find start and stop idxs of aligned signal
        rel_center_idx = np.argmin(np.abs(rel_ts))

        # ignore alignments with windows that extend beyond the signal
        if rel_center_idx - pre_idxs < 0 or rel_center_idx + post_idxs > len(signal):
            continue

        signal_mat[i,:] = signal[rel_center_idx - pre_idxs : rel_center_idx + post_idxs]

    return signal_mat, t


def build_trial_dff_signal_matrix(raw_signal, ts, align_ts, pre, post, baseline_windows, align_sel=[]):
    '''
    Build a matrix of signals aligned to the specified timestamp with the given window where df/f is calculated separately
    for each alignment point based on the average signal within the provided baseline window limits


    Parameters
    ----------
    raw_signal : The entire raw fluorescent signal
    The timestamps associated with the signal
    align_ts : The alignment timestamps
    pre : The time pre alignment point to show
    post : The time post alignment point to show
    baseline_windows : The windows defining the per-trial baseline to use for calculating df/f. These are relative to the alignment ts
    align_sel : Boolean selection of alignment points, optional

    Returns
    -------
    signal_mat : Matrix of signals where each row is the signal aligned to the specified alignment point
    t : The relative time array associated with the columns of the signal matrix

    '''

    if len(align_sel) > 0:
        align_ts = align_ts[align_sel]
        baseline_windows = baseline_windows[align_sel,:]

    # baseline windows are relative to each alignment point
    dt = np.mean(np.diff(ts))
    t = np.arange(-pre, post+dt, dt)
    center_idx = np.argmin(np.abs(t))
    pre_idxs = center_idx
    post_idxs = len(t) - center_idx

    signal_mat = np.full((len(align_ts), len(t)), np.nan)

    for i, align_t in enumerate(align_ts):

        rel_ts = ts - align_t
        # get aligned signal
        rel_center_idx = np.argmin(np.abs(rel_ts))

        # ignore alignments with windows that extend beyond the signal
        if rel_center_idx - pre_idxs < 0 or rel_center_idx + post_idxs > len(raw_signal):
            continue

        aligned_signal = raw_signal[rel_center_idx - pre_idxs : rel_center_idx + post_idxs]

        # calculate dFF for aligned signal based on signal in baseline window
        window = baseline_windows[i]
        window_start_idx = np.argmin(np.abs(rel_ts-window[0]))
        window_end_idx = np.argmin(np.abs(rel_ts-window[1]))
        baseline_signal = np.nanmean(raw_signal[window_start_idx:window_end_idx+1])

        signal_mat[i,:] = (aligned_signal-baseline_signal)/baseline_signal*100

    return signal_mat, t


def build_time_norm_signal_matrix(signal, ts, start_align_ts, end_align_ts, n_bins, align_sel=[], interp_deg=2):
    '''
    Build a matrix of signals where the time of the signal is normalized to the specified start and end timestamps
    And the signal is discretized into the given number of bins. This is done through spline interpolation.

    Parameters
    ----------
    signal : The entire signal
    ts : The timestamps associated with the signal
    start_align_ts : The timestamps for the start of the alignment
    end_align_ts : The timestamps for the end of the alignment
    n_bins : The number of bins in which to break the signal between alignment points
    align_sel : Boolean selection of alignment points, optional
    interp_deg : The degree of the interpolation splines, default=2

    Returns
    -------
    signal_mat : Matrix of signals where each row is the signal aligned to the specified alignment point

    '''

    if len(align_sel) > 0:
        start_align_ts = start_align_ts[align_sel]
        end_align_ts = end_align_ts[align_sel]

    if len(start_align_ts) != len(end_align_ts):
        raise ValueError('Start and end timestamps have unequal numbers of elements')

    n_bins = int(n_bins)

    signal_mat = np.full((len(start_align_ts), n_bins), np.nan)

    for i, (start_t, end_t) in enumerate(zip(start_align_ts, end_align_ts)):

        if start_t is None or np.isnan(start_t) or end_t is None or np.isnan(end_t) or end_t < start_t:
            continue

        # find start and stop idxs of aligned signal
        rel_start_ts = ts - start_t
        rel_end_ts = ts - end_t
        # include one extra data point for better interpolation
        rel_start_idx = np.argmin(np.abs(rel_start_ts))-1
        rel_end_idx = np.argmin(np.abs(rel_end_ts))+2 # added 2 because indexing ignores last value

        # build interpolating spline
        sub_ts = ts[rel_start_idx:rel_end_idx]
        sub_signal = signal[rel_start_idx:rel_end_idx]
        # remove any nans
        nan_sel = ~np.isnan(sub_signal)
        # ignore if all timepoints are nan
        if all(~nan_sel):
            continue

        interp_sig = make_interp_spline(sub_ts[nan_sel], sub_signal[nan_sel], k=interp_deg)

        # use interpolation to get values at new bin centers
        # have the new bins contain the start and end timepoints in the center of the bin
        bin_centers = np.linspace(start_t, end_t, n_bins)
        norm_sig = interp_sig(bin_centers)
        signal_mat[i,:] = norm_sig

    return signal_mat
