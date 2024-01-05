# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:33:47 2023

@author: tanne
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


def calc_iso_dff(lig_signal, iso_signal):
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

    '''

    fitted_iso = fit_signal(iso_signal, lig_signal)

    # calculate and return dF/F
    return ((lig_signal - fitted_iso)/fitted_iso)*100


def fit_signal(signal_to_fit, signal):
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

    reg = LinearRegression()
    # find all NaNs and drop them from both signals before regressing
    nans = np.isnan(signal) | np.isnan(signal_to_fit)
    # fit the iso signal to the ligand signal
    reg.fit(signal_to_fit[~nans,None], signal[~nans])
    fitted_signal = np.full_like(signal_to_fit, np.nan)
    fitted_signal[~nans] = reg.predict(signal_to_fit[~nans,None])

    return fitted_signal


def fit_baseline(signal, n_points_min=100):
    '''
    Fits a baseline to the signal using an exponential decay function with an additional linear term:
        A*e^(-Bt) - Ct + D
    Fits the baseline to the minimum value every n data points

    Parameters
    ----------
    signal : The signal to fit
    n_points_min: The number of data points to take the min over, optional

    Returns
    -------
    The fit baseline

    '''

    baseline_form = lambda x, a, b, c, d: a*np.exp(-b*x) - c*x + d

    if len(signal) % n_points_min != 0:
        min_signal = np.append(signal, np.full(n_points_min - (len(signal) % n_points_min), np.nan))

    # compute the minimum value every n points
    min_signal = np.nanmin(np.reshape(min_signal, (-1, n_points_min)), axis=1)

    # ignore nans in fit
    nans = np.isnan(min_signal)

    # fit curve to these minimum values
    x = np.arange(len(min_signal))
    params = curve_fit(baseline_form, x[~nans], min_signal[~nans])[0]

    # return full baseline of the same length as the signal
    x = np.arange(len(signal))/n_points_min
    return baseline_form(x, *params)


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