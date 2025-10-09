# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:33:47 2023

@author: tanne
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import utils

try:
    import cupy as cp
    if cp.cuda.is_available():
        xp = cp
        from cupyx.scipy.interpolate import interp1d
        from cupyx.scipy.signal import butter, sosfiltfilt
        gpu_available = True
        
except:
    xp = np
    from scipy.interpolate import interp1d
    from scipy.signal import butter, sosfiltfilt
    gpu_available = False

import warnings

def to_cupy(x):
    if gpu_available and isinstance(x, np.ndarray):
        return cp.asarray(x)
    return x

def to_numpy(x):
    if gpu_available and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x

def fill_signal_nans(signal):
    nan_idxs = np.isnan(signal)
    
    if any(nan_idxs):
        return pd.Series(signal).interpolate(method='linear', limit_direction='both').to_numpy(copy=True), nan_idxs
    else:
        return signal, nan_idxs

def filter_signal(signal, cutoff_f, sr, filter_type='lowpass', order=3):
    '''
    Low-pass filters a signal with a zero-phase butterworth filter of the given cutoff frequency

    Parameters
    ----------
    signal : The signal to filter
    cutoff_f : The cutoff frequency(s)
    sr : The sampling rate
    filter_type : 'lowpass', 'highpass', 'bandpass', 'bandstop'

    Returns
    -------
    The filtered signal

    '''

    # handle nan values so the filtering works correctly
    # Since we don't want to just remove the nans, first interpolate them, filter, then add them back in
    signal, nans = fill_signal_nans(signal)

    sos = butter(order, cutoff_f, btype=filter_type, fs=sr, output='sos')
    filtered_signal = sosfiltfilt(sos, to_cupy(signal), padtype='even')

    filtered_signal[nans] = xp.nan

    return to_numpy(filtered_signal)


def calc_iso_dff(lig_signal, iso_signal, t, vary_t=False):
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

    fitted_iso, fit_info = fit_signal(iso_signal, lig_signal, t, vary_t)
    # calculate dF/F
    dff = ((lig_signal - fitted_iso)/fitted_iso)*100

    return dff, fitted_iso, fit_info


def fit_signal(signal_to_fit, signal, t, vary_t=False, fit_bands=False, f_bands=None):
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

    if vary_t and not fit_bands:
        form = lambda x, a, b, c: a*x[0,:] + b*x[1,:] + c
        s_to_fit = np.vstack((signal_to_fit[None,~nans], t[None,~nans]))
        bounds = ([      0, -np.inf, -np.inf],
                  [ np.inf,  np.inf,  np.inf])
    
    elif fit_bands:
        if f_bands is None:
            f_bands = [[0,0.01], [0.01,0.1], [0.1,1], [1,10]] 

        # dynamically create the formula to optimize
        param_names = ['b{}'.format(i) for i in range(len(f_bands))]
        
        if vary_t:
            param_names.append('b{}'.format(len(param_names)))
        
        expr = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names)]) + ' + c'

        param_names = param_names + ['c']

        # Build the lambda string and evaluate it 
        form = eval('lambda x, {}: {}'.format(', '.join(param_names), expr))
        
        # all coefficients except the intercept must be positive
        bounds = (np.zeros(len(f_bands)).tolist(), np.inf)
        bounds[0].append(-np.inf)
        
        if vary_t:
            bounds[0].append(-np.inf)
        
        sr = 1/np.mean(np.diff(t))
        
        # separate the signal to fit into different frequency band components
        s_to_fit = np.zeros((len(f_bands), len(signal_to_fit)))
        for i, band in enumerate(f_bands):

            if band[0] == 0:
                s_to_fit[i,:] = filter_signal(signal_to_fit, band[1], sr=sr, filter_type='lowpass')
            elif len(band) == 1:
                s_to_fit[i,:] = filter_signal(signal_to_fit, band[0], sr=sr, filter_type='highpass')
            else:
                s_to_fit[i,:] = filter_signal(signal_to_fit, band, sr=sr, filter_type='bandpass')
                
        if vary_t:
            s_to_fit = np.vstack([s_to_fit, t[None,:]])
            
        s_to_fit = s_to_fit[:,~nans]
        
    else:
        form = lambda x, a, b: a*x + b
        s_to_fit = signal_to_fit[~nans]
        bounds = ([      0, -np.inf],
                  [ np.inf,  np.inf])

    params = curve_fit(form, s_to_fit, signal[~nans], bounds=bounds)[0]
    fitted_signal = np.full_like(signal_to_fit, np.nan)
    fitted_signal[~nans] = form(s_to_fit, *params)

    return fitted_signal, {'params': params, 'formula': form}


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
            bounds = ([-np.inf,      0, -np.inf, 0, -np.inf],
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
            return baseline_form(x, *params), {'params': params, 'formula': baseline_form}
        
        except (OptimizeWarning, RuntimeError):
            print('Baseline fit was unseccessful. Expanding the signal minimum window to {}..'.format(n_points_min*2))
            return fit_baseline(signal, n_points_min=n_points_min*2, baseline_form=baseline_form)


def build_signal_matrix(signal, ts, align_ts, pre, post, align_sel=[], mask_lims=None):
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
    mask_lims : Trial-by-trial limits to mask the signal beyond (Nx2 array for N alignment points)

    Returns
    -------
    signal_mat : Matrix of signals where each row is the signal aligned to the specified alignment point
    t : The relative time array associated with the columns of the signal matrix

    '''

    if len(align_sel) > 0:
        align_ts = align_ts[align_sel]

    if not mask_lims is None and not len(mask_lims) == 0:
        mask_lims = np.array(mask_lims)

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

        # handle alignments with windows that extend beyond the signal
        if rel_center_idx - pre_idxs < 0:
            align_sig = np.concatenate([np.full(pre_idxs - rel_center_idx, np.nan), signal[:rel_center_idx + post_idxs].copy()])
        elif rel_center_idx + post_idxs > len(signal):
            align_sig = np.concatenate([signal[rel_center_idx - pre_idxs:].copy(), np.full(post_idxs - len(signal) + rel_center_idx, np.nan)])
        else:
            align_sig = signal[rel_center_idx - pre_idxs : rel_center_idx + post_idxs].copy()

        if not mask_lims is None:
            rel_mask_lim = mask_lims[i,:] - align_t
            align_sig[(t < rel_mask_lim[0]) | (t > rel_mask_lim[1])] = np.nan

        signal_mat[i,:] = align_sig

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


def build_time_norm_signal_matrix(signal, ts, start_align_ts, end_align_ts, n_bins, align_sel=None, include_end=False):
    '''
    Build a matrix of signals where the time of the signal is normalized to the specified start and end timestamps
    And the signal is discretized into the given number of bins. This is done through linear interpolation.

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

    if len(start_align_ts) != len(end_align_ts):
        raise ValueError('Start and end timestamps have unequal numbers of elements')
        
    if align_sel is None:
        align_sel = np.full_like(start_align_ts, True)

    n_bins = int(n_bins)

    signal_mat = np.full((len(start_align_ts), n_bins), np.nan)
    sig_nans = np.isnan(signal)

    if any(sig_nans):
        signal, _ = fill_signal_nans(signal)

    for i, (start_t, end_t) in enumerate(zip(start_align_ts, end_align_ts)):

        if start_t is None or np.isnan(start_t) or end_t is None or np.isnan(end_t) or end_t < start_t or not align_sel[i]:
            continue

        # find start and stop idxs of aligned signal
        start_idx = np.argmin(np.abs(ts - start_t))
        end_idx = np.argmin(np.abs(ts - end_t))
        
        if include_end:
            end_idx += 1 # added 1 because indexing ignores last value
        
        if (end_idx - start_idx) == n_bins:
            norm_sig = signal[start_idx:end_idx]
            
        else:
            # include extra points for better interpolation
            rel_start_idx = start_idx - 1
            rel_end_idx = end_idx + 1

            # handle nans
            nan_sel = sig_nans[rel_start_idx:rel_end_idx]
            # ignore if all timepoints are nan
            if all(nan_sel):
                continue
    
            # build interpolation
            sub_ts = ts[rel_start_idx:rel_end_idx]
            sub_signal = signal[rel_start_idx:rel_end_idx]
            interp_sig = interp1d(sub_ts, sub_signal, kind='linear', bounds_error=False, fill_value='extrapolate')
    
            # use interpolation to get values at new bin centers
            # have the new bins contain the start and end timepoints in the center of the bin
            bin_centers = np.linspace(start_t, end_t, n_bins, endpoint=include_end)
            norm_sig = interp_sig(bin_centers)
            
            # remove any datapoints that are within nans in full signal
            if any(nan_sel):
                # first get edges of nan segments
                # pad with False at both ends so we can catch edges
                padded_nan_sel = np.concatenate([np.array([False]), nan_sel, np.array([False])])
    
                diff = np.diff(padded_nan_sel.astype(int))
                
                # start of segment is 1, end is -1
                start_idxs = np.where(diff == 1)[0]
                end_idxs = np.where(diff == -1)[0]
                
                # vectorized masking
                left_bounds  = sub_ts[start_idxs]
                right_bounds = sub_ts[end_idxs-1]
    
                mask_matrix = (bin_centers[:, None] >= left_bounds) & (bin_centers[:, None] <= right_bounds)
                norm_nans = np.any(mask_matrix, axis=1)
                
                norm_sig[norm_nans] = np.nan

        signal_mat[i,:] = norm_sig

    return signal_mat


def correlate(x, y, dt, max_lag=10, batch_size=200, t_sel=None):
    '''
    Compute Pearson cross-correlation between two signals for lags in [-max_lag, +max_lag],
    ignoring NaNs in either signal.
    
    Returns correlation values in [-1, 1] for each lag along with the associated lag value
    '''
    
    if len(x) != len(y):
        raise ValueError('x and y must be the same length')
        
    t = len(x)
    if t_sel is None:
        t_sel = np.full(t, True)

    centered_x = utils.z_score(x)
    centered_y = utils.z_score(y)
    
    # mask out anything not in t_sel
    centered_x[~t_sel] = np.nan
    centered_y[~t_sel] = np.nan
    
    max_lag = utils.convert_to_multiple(max_lag, dt)
    n_lags = int(max_lag/dt)
    lag_steps = np.arange(-n_lags, n_lags+1)

    # Make a 2D lagged matrix of y (shape: n_lags × t)
    # Each row is y shifted by a lag, with NaN fill
    corrs = []
    
    # set nans to 0 so they do not contribute to the resulting calculation
    nans = np.isnan(centered_x)
    centered_x[nans] = 0
    
    for i in range(0, len(lag_steps)+batch_size, batch_size):
        
        if i+batch_size > len(lag_steps):
            batch_lags = lag_steps[i:]
        else:
            batch_lags = lag_steps[i:i+batch_size]
        
        y_mat = np.full((len(batch_lags), t), np.nan, dtype=float)
        for i, lag in enumerate(batch_lags):
            if lag < 0:
                y_mat[i, :lag] = centered_y[-lag:]   # shift second signal backward so x is leading
            elif lag > 0:
                y_mat[i, lag:] = centered_y[:-lag]   # shift second signal forward so x is lagging
            else:
                y_mat[i, :] = centered_y
                
        x_mat = np.broadcast_to(centered_x, (len(batch_lags), t))
        
        # Mask shifted NaNs
        mask = np.broadcast_to(~nans, (len(batch_lags), t)) & ~np.isnan(y_mat)
        y_mat[~mask] = 0

        counts = np.sum(mask, axis=1)
    
        # compute pearson correlation coefficient for each lag
        corrs.extend(np.sum(x_mat * y_mat, axis=1)/counts)
    
    return np.array(corrs), np.arange(-max_lag, max_lag+dt, dt)

def correlate_over_time(x, y, dt, t_width=0.5):
    '''
    Compute Pearson cross-correlation between two signals over time for a sliding centered window,
    ignoring NaNs in either signal.
    
    Returns correlation values in [-1, 1] for each bin over time
    '''
    
    if len(x) != len(y):
        raise ValueError('x and y must be the same length')

    centered_x = utils.z_score(x)
    centered_y = utils.z_score(y)

    t_width = utils.convert_to_multiple(t_width, dt)
    n_bins = int(t_width/dt)
    # make sure n bins is odd so it is centered
    if n_bins % 2 == 0:
        n_bins += 1
        
    weights = np.ones(n_bins)
    
    # calculate correlation
    corr = centered_x * centered_y
    not_nans = ~np.isnan(corr)
    corr[~not_nans] = 0
    
    # get moving sums through convolution
    t_corr = np.convolve(corr, weights, mode='same')
    counts = np.convolve(not_nans.astype(int), weights, mode='same')
    
    with np.errstate(invalid='ignore', divide='ignore'):
        t_corr = t_corr / counts
    
    count_sel = counts < 2
    
    t_corr[count_sel] = np.nan
    
    return t_corr