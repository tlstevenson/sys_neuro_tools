# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:33:47 2023

@author: tanne
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import utils
import scipy.signal as sig
import statsmodels.api as sm
import scipy.sparse as sp 
from sklearn.linear_model import LinearRegression
from scipy.stats import t, f, chi2

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
    
def decimate(signal, time, dec_factor):
    '''
    Decimate the signal by the given factor by averaging over the signal in that many bins

    Parameters
    ----------
    signal : The signal to decimate
    time : The timestamps associated with each data point
    dec_factor : The factor by which to decimate

    Returns
    -------
    The decimated signal and new timestamp array with the timestamps being the midpoint of the bin

    '''
    
    reshape_signal = np.append(signal, np.full(dec_factor - (len(signal) % dec_factor), np.nan))
    reshape_signal = np.reshape(reshape_signal, (-1, dec_factor))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dec_signal = np.nanmean(reshape_signal, axis=1)
    
    reshape_time = np.append(time, np.full(dec_factor - (len(time) % dec_factor), np.nan))
    reshape_time = np.reshape(reshape_time, (-1, dec_factor))
    dec_time = np.nanmean(reshape_time, axis=1)
    
    return dec_signal, dec_time

def filter_signal(signal, cutoff_f, sr, filter_type='lowpass', order=3, trend_pad_len=None):
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
    
    if trend_pad_len is None:
        filtered_signal = sosfiltfilt(sos, to_cupy(signal), padtype='even')
    else:
        # pad the signal on each end with the extrapolated linear trend over that pad length before or after the edge
        left_pad = extrapolate_edge(signal[:trend_pad_len], pad_len=trend_pad_len, direction='left')
        right_pad = extrapolate_edge(signal[-trend_pad_len:], pad_len=trend_pad_len, direction='right')
        pad_signal = np.concatenate([left_pad, signal, right_pad])
        
        filtered_signal = sosfiltfilt(sos, to_cupy(pad_signal), padtype=None)
        filtered_signal = filtered_signal[trend_pad_len:-trend_pad_len]

    filtered_signal[nans] = xp.nan

    return to_numpy(filtered_signal)


def decompose_signal_fbands(signal, f_bands, sr, custom_pad=False):
    '''
    Decompose a signal into its frequency bands

    Parameters
    ----------
    signal : The signal to decompose
    f_bands : The frequency bands
    sr : The sample rate

    Returns
    -------
    A matrix of shape len(f_bands) x len(signal) containing the portion of the signal in each frequency band

    '''
    decom_signal = np.zeros((len(f_bands), len(signal)))
    for i, band in enumerate(f_bands):
        
        if custom_pad:
            base_f = band[0] if not band[0] == 0 else band[1]
            trend_pad_len = int((1/(2*base_f))*sr)
        else:
            trend_pad_len = None

        if len(band) == 1:
            decom_signal[i,:] = filter_signal(signal, band[0], sr=sr, filter_type='highpass', trend_pad_len=trend_pad_len)
        elif band[0] == 0:
            if band[1] == np.inf:
                decom_signal[i,:] = signal
            else:
                decom_signal[i,:] = filter_signal(signal, band[1], sr=sr, filter_type='lowpass', trend_pad_len=trend_pad_len)
        else:
            decom_signal[i,:] = filter_signal(signal, band, sr=sr, filter_type='bandpass', trend_pad_len=trend_pad_len)
            
    return decom_signal
            

def extrapolate_edge(y_seg, pad_len=None, direction='right'):
    # y_seg is the segment used for fitting in chronological order
    n = len(y_seg)
    # sample indices relative to segment start
    t = np.arange(n)
    # fit linear trend y = a*t + b
    a, b = np.polyfit(t, y_seg, 1)
    
    if pad_len is None:
        pad_len = n

    # times for extrapolation
    if direction == 'right':
        t_extra = np.arange(1, pad_len+1) + t[-1]  # continue after last sample
    else:
        # for left pad, go before t=0: negative times
        t_extra = t[0] - np.arange(pad_len, 0, -1)
        
    y_extra = a * t_extra + b
    return y_extra


def calc_iso_dff(lig_signal, iso_signal, t, vary_t=False, allow_neg_coeffs=False):
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

    fitted_iso, fit_info = fit_signal(iso_signal, lig_signal, t, vary_t=vary_t, allow_neg_coeffs=allow_neg_coeffs)
    # calculate dF/F
    dff = ((lig_signal - fitted_iso)/fitted_iso)*100

    return dff, fitted_iso, fit_info


def fit_signal(signal_to_fit, signal, t, vary_t=False, fit_bands=False, f_bands=None, allow_neg_coeffs=False):
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
        if allow_neg_coeffs:
            bounds = ([-np.inf, -np.inf, -np.inf],
                      [ np.inf,  np.inf,  np.inf])
        else:
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
        
        if allow_neg_coeffs:
            bounds = (np.full(len(f_bands), -np.inf).tolist(), np.inf)
        else:
            # all coefficients except the intercept must be positive
            bounds = (np.zeros(len(f_bands)).tolist(), np.inf)
        bounds[0].append(-np.inf)
        
        if vary_t:
            bounds[0].append(-np.inf)
        
        sr = 1/np.mean(np.diff(t))
        
        # separate the signal to fit into different frequency band components
        s_to_fit = decompose_signal_fbands(signal_to_fit, f_bands, sr)
        
        if vary_t:
            s_to_fit = np.vstack([s_to_fit, t[None,:]])
            
        s_to_fit = s_to_fit[:,~nans]
        
    else:
        form = lambda x, a, b: a*x + b
        s_to_fit = signal_to_fit[~nans]
        if allow_neg_coeffs:
            bounds = ([-np.inf, -np.inf],
                      [ np.inf,  np.inf])  
        else:
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


def correlate(x, y, dt, neg_lag=2, pos_lag=2, batch_size=200, t_sel=None, z_score=True, z_score_t_sel=False):
    '''
    Compute Pearson cross-correlation between two signals for lags in [-neg_lag, +pos_lag],
    ignoring NaNs in either signal.
    
    Returns correlation values in [-1, 1] for each lag along with the associated lag value
    
    SHould update to use:
        acf = sig.correlate(x, y, mode='full')
        rho = acf[N:] / acf[N-1]
    '''
    
    if len(x) != len(y):
        raise ValueError('x and y must be the same length')
        
    t = len(x)
    if t_sel is None:
        t_sel = np.full(t, True)

    if z_score:
        centered_x = utils.z_score(x)
        centered_y = utils.z_score(y)
    else:
        centered_x = x.copy()
        centered_y = y.copy()
    
    # mask out anything not in t_sel
    centered_x[~t_sel] = np.nan
    centered_y[~t_sel] = np.nan
    
    if z_score_t_sel:
        centered_x = utils.z_score(centered_x)
        centered_y = utils.z_score(centered_y)
    
    neg_lag = utils.convert_to_multiple(neg_lag, dt)
    pos_lag = utils.convert_to_multiple(pos_lag, dt)
    lag_steps = np.arange(-int(neg_lag/dt), int(pos_lag/dt)+1)

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
    
    return np.array(corrs), lag_steps*dt

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


def calc_power_spectra(signal, dt, f_min=0.005):

    nperseg = round(1/dt*2/f_min)
    
    # get rid of nans by interpolation
    tmp_sig, _ = fill_signal_nans(signal)

    freqs, ps = sig.welch(tmp_sig, fs=1/dt, nperseg=nperseg, scaling='density')

    return freqs, ps


def rectify_ica_output(A, S):
    '''
    Rearrange columns of mixing matrix and independent components so that the diagonal has the largest values that are all positive

    Parameters
    ----------
    A : (n_channels, n_ic)  mixing matrix
    S : (T, n_ic) IC components

    Returns
    -------
    A : rectified mixing matrix
    S : rectified IC components

    '''

    n_channels, n_ic = A.shape
    col_order = []
    col_rect = []
    all_cols = np.arange(n_ic)
    for i in range(n_ic):
        max_idx = np.argmax(np.abs(A[i,all_cols]))
        max_idx = all_cols[max_idx]
        all_cols = all_cols[all_cols != max_idx]
        col_order.append(max_idx)
        max_col_val_idx = np.argmax(np.abs(A[:,max_idx]))
        if A[max_col_val_idx, max_idx] < 0:
            col_rect.append(-1)
        else:
            col_rect.append(1)
            
    rect_mult = np.array(col_rect)[None,:]
    
    A = A[:, col_order]*rect_mult
    S = S[:, col_order]*rect_mult
    
    return A, S, {'col_order': col_order, 'rect_mult': rect_mult}


def reconstruct_ica(A, S):
    """
    Compute explained power of ICA components.

    Parameters
    ----------
    A : (n_channels, n_ic)  mixing matrix
    S : (T, n_ic) IC components

    Returns
    -------
    reconstruction : (T, n_channels, n_ic)
        signal reconstruction from each IC component
    """

    # calculate reconstruction of each channel from each IC

    return np.einsum('ti,ci->tci', S, A)

def ic_explained_power(A, S, X):
    """
    Compute explained power (R^2) of ICA components.

    Parameters
    ----------
    A : (n_channels, n_ic)  mixing matrix
    S : (T, n_ic) IC components
    X : (T, n_channels) Data matrix

    Returns
    -------
    power_total : 
        fraction of total dataset power explained by IC reconstruction
    power_ic_total : (n_ic,)
        fraction of total dataset power explained by each IC
    power_ic_channel : (n_channels, n_ic)
        fraction of power explained per channel by each IC
    """

    n_channels, n_ic = A.shape

    data_channel_power = np.sum(X**2, axis=0)
    data_power = np.sum(data_channel_power)

    power_ic_total = np.zeros(n_ic)
    power_ic_channel = np.zeros((n_channels, n_ic))
    
    recons = reconstruct_ica(A, S)
    
    Xhat = recons.sum(axis=2)
    power_total = 1 - np.sum((X - Xhat)**2)/data_power

    for i in range(n_ic):

        Xhat_i = recons[:,:,i]

        power_ic_total[i] = 1 - np.sum((X - Xhat_i)**2) / data_power
        power_ic_channel[:, i] =  1 - np.sum((X - Xhat_i)**2, axis=0) / data_channel_power

    return power_total, power_ic_total, power_ic_channel


def kernel_regression(signal_dict, ts_dict, event_ts_dict, kernel_edges, dt, n_perms=100, crossval=False):
    
    """
    Perform event-triggered kernel regression and assess significance with event time permutation bootstrapping
    
    Parameters
    ----------
    signal_dict : dictionary of signals keyed by session id
    ts_dict : dictionary of timestamps for each signal
    event_ts_dict : nested dictionary of event times keyed by session id and event name
    kernel_edges : either an array of kernel [start end] times or a dictionary of edges for each event, in seconds
    dt : sampling period in seconds
        
    Returns
    -------
    a dictionary with the following elements:
        kernels: a dictionary of kernel shapes and associated times keyed by event
        kernel_stats: a dictionary of stats for each timepoint in each kernel including
            95% confidence intervals, standard errors, and p-values
        event_stats: a dictionary of stats for each event including
            F statistic, F p-value, reduced model residual variance, degrees of freedom,
            and fraction of total variance explained by event (ΔR^2)
        model_stats: a dictionary of stats for the entire model including
            total signal variance, model residual variance, total model R^2
        sess_predictors: a dictionary of the predictors by session and the associated column indices
    """
    
    event_ts_dict = _check_convert_event_ts(event_ts_dict)
    
    sess_ids = list(signal_dict.keys())
    event_names = list(event_ts_dict[sess_ids[0]].keys())
    
    # compute kernel time and column indices
    if not utils.is_dict(kernel_edges):
        kernel_edges = {e: kernel_edges for e in event_names}
        
    # extrapolate kernel edges to similar event names with different suffixes
    new_kernel_edges = kernel_edges.copy()
    for e in event_names:
        if e not in kernel_edges:
            # find matching name
            e_key = [k for k in kernel_edges.keys() if k in e]
            if len(e_key) == 1:
                new_kernel_edges[e] = kernel_edges[e_key[0]]
            else:
                raise KeyError('Kernel edges has {} matching keys for {}'.format(len(e_key), e))
                
    kernel_edges = new_kernel_edges
    
    kernel_ts = {}
    kernel_delays = {}
    event_col_idxs = {}
    col_start = 0
    P = 0 # total width of predictor matrix
    max_k = 0
    for e in event_names:
        kernel_t = np.concatenate((np.flip(np.arange(0, kernel_edges[e][0]-dt, -dt)), np.arange(dt, kernel_edges[e][1]+dt, dt)))
        kernel_ts[e] = kernel_t
        kernel_zero_idx = np.argmin(np.abs(kernel_t))
        kernel_delays[e] = np.arange(len(kernel_t)) - kernel_zero_idx
        
        k = len(kernel_t)
        P += k
        if k > max_k:
            max_k = k
        col_end = col_start + len(kernel_t)
        event_col_idxs[e] = np.arange(col_start, col_end)
        col_start = col_end

    # Build stacked signal vector and sparse predictor matrix

    stacked_signals = []
    predictor_mat = []
    all_sess_predictors = {}
    sess_rows = {}
    sess_nans = {}
    row_start = 0

    for sess_id in sess_ids:
        
        sig = signal_dict[sess_id]
        sig_ts = ts_dict[sess_id]

        sess_predictors = _build_event_matrix(event_ts_dict[sess_id], kernel_delays, event_col_idxs, event_names, sig_ts, dt)

        all_sess_predictors[sess_id] = sess_predictors

        # remove nans
        nan_sel = ~np.isnan(sig)
        sess_nans[sess_id] = nan_sel
        
        sub_sig = sig[nan_sel].astype(np.float32)

        stacked_signals.append(sub_sig)
        predictor_mat.append(sess_predictors[nan_sel,:])
        
        row_end = row_start + len(sub_sig)
        sess_rows[sess_id] = np.arange(row_start, row_end)
        row_start = row_end

    stacked_signals = np.concatenate(stacked_signals)
    predictor_mat = sp.vstack(predictor_mat)

    # get a mask for timepoints that don't have any predictors to compare stats
    row_mask = predictor_mat.getnnz(axis=1) > 0
    masked_signals = stacked_signals[row_mask]
    
    # Perform full regression
    full_model = LinearRegression()
    full_model.fit(predictor_mat, stacked_signals)
    betas = full_model.coef_.flatten()
    
    pred_signals = full_model.predict(predictor_mat)
    
    # compute full model stats
    full_resid = stacked_signals - pred_signals

    masked_RSS = np.sum(full_resid[row_mask]**2)
    masked_TSS = np.sum((masked_signals - np.mean(masked_signals))**2)

    full_RSS = np.sum(full_resid**2)
    full_TSS = np.sum((stacked_signals - np.mean(stacked_signals))**2)
    
    tot_T = len(stacked_signals)
    mask_T = sum(row_mask)

    mask_r2 = 1 - masked_RSS / masked_TSS
    full_r2 = 1 - full_RSS / full_TSS

    # perform leave-one-out cross validation
    cv_results = {'masked_R2': [], 'full_R2': []}
    cv_betas = []

    if crossval:
        for sess_id in sess_ids:
            cv_test_rows = sess_rows[sess_id]
            if len(cv_test_rows) == 0:
                continue
            
            cv_train_rows = np.concatenate([sess_rows[s] for s in sess_ids if s != sess_id])
            cv_row_mask_test = row_mask[cv_test_rows]
    
            cv_test_signals = stacked_signals[cv_test_rows,:]
            cv_test_masked_signals = cv_test_signals[cv_row_mask_test]
            
            # Fit to training sessions
            cv_model = LinearRegression()
            cv_model.fit(predictor_mat[cv_train_rows,:], stacked_signals[cv_train_rows,:])
            cv_betas.append(cv_model.coef_.flatten())
            
            # Evaluate on test session
            cv_pred_test_signals = cv_model.predict(predictor_mat[cv_test_rows,:])
            
            cv_masked_RSS = np.sum((cv_test_masked_signals - cv_pred_test_signals[cv_row_mask_test])**2)
            cv_masked_TSS = np.sum((cv_test_masked_signals - np.mean(cv_test_masked_signals))**2)
    
            cv_full_RSS = np.sum((cv_test_signals - cv_pred_test_signals)**2)
            cv_full_TSS = np.sum((cv_test_signals - np.mean(cv_test_signals))**2)
    
            cv_mask_r2 = 1 - cv_masked_RSS / cv_masked_TSS
            cv_full_r2 = 1 - cv_full_RSS / cv_full_TSS
            
            cv_results['masked_R2'].append(cv_mask_r2)
            cv_results['full_R2'].append(cv_full_r2)

    model_stats = {
        'R2_masked': mask_r2,
        'RSS_masked': masked_RSS,
        'TSS_masked': masked_TSS,
        'R2_full': full_r2,
        'RSS_full': full_RSS,
        'TSS_full': full_TSS,
        'n_preds': P,
        'T_full': tot_T,
        'T_masked': mask_T,
        'cv_results': cv_results
    }

    kernel_stats = {}
    event_stats = {}

    # compute stats on coefficients using Newey-West covariance
    df = mask_T-P
    autocorr_k = _compute_resid_autocorr_lag(full_resid, sess_rows)
    max_lag = min(max_k, autocorr_k)
    cov_beta = _hac_covariance(predictor_mat, full_resid, sess_rows, max_lag)
    cov_beta *= mask_T/df
    se_betas = np.sqrt(np.diag(cov_beta))
    
    t_stats = betas / se_betas
    p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df))

    ci_alpha = 0.05
    c = t.ppf(1-ci_alpha/2, df)
    cis_low = betas - c * se_betas
    cis_high = betas + c * se_betas

    # compute contribution of each event by leaving each one out and comparing to shuffled event times
    all_cols = np.arange(P)
    
    for e in event_names:
        
        # compute event stats
        event_cols = event_col_idxs[e]
        keep_cols = np.setdiff1d(all_cols, event_cols)

        red_pred_mat = predictor_mat[:, keep_cols]
        event_row_sel = predictor_mat[:, event_cols].getnnz(axis=1) > 0

        # fit reduced design model 

        model_red = LinearRegression()
        model_red.fit(red_pred_mat, stacked_signals)
        
        red_pred_signals = model_red.predict(red_pred_mat)

        red_masked_RSS = np.sum((masked_signals - red_pred_signals[row_mask])**2)
        red_full_RSS = np.sum((stacked_signals - red_pred_signals)**2)
        # also compute only during event kernel
        event_signals = stacked_signals[event_row_sel]
        red_event_RSS = np.sum((event_signals - red_pred_signals[event_row_sel])**2)
        full_event_RSS = np.sum((event_signals - pred_signals[event_row_sel])**2)
        event_TSS = np.sum((event_signals - np.mean(stacked_signals[event_row_sel]))**2)
        
        # ΔR² (fraction of total variance explained by event)- effect size
        delta_R2_masked = (red_masked_RSS - masked_RSS) / masked_TSS
        delta_R2_full = (red_full_RSS - full_RSS) / full_TSS
        delta_R2_event = (red_event_RSS - full_event_RSS) / event_TSS

        # Wald Statistic
        beta_e = betas[event_cols]
        cov_e = cov_beta[np.ix_(event_cols, event_cols)]

        W = beta_e.T @ np.linalg.inv(cov_e) @ beta_e
        p_event = 1 - chi2.cdf(W, len(event_cols))
        
        # perform event permutation regressions
        perm_results = {'delta_R2_masked': [], 'delta_R2_full': []}
        perm_betas = []
        perm_event_cols = {e: np.arange(len(event_cols))}
        act_perm_cols = len(keep_cols) + perm_event_cols[e]
        
        for i in range(n_perms):
            
            # build permuted predictor matrix
            perm_predictor_mat = []
            
            for sess_id in sess_ids:
                sig_ts = ts_dict[sess_id]
                shifted_events = _circular_shift_events(event_ts_dict[sess_id][e], sig_ts[0], sig_ts[-1])

                perm_predictors = _build_event_matrix({e: shifted_events}, kernel_delays, perm_event_cols, [e], sig_ts, dt)
                perm_predictor_mat.append(perm_predictors[sess_nans[sess_id],:])
            
            perm_predictor_mat = sp.vstack(perm_predictor_mat)
            
            # add permuted event columns to the end of the matrix
            perm_predictor_mat = sp.hstack([red_pred_mat, perm_predictor_mat])

            # perform permuted regression    
            model_perm = LinearRegression()
            model_perm.fit(perm_predictor_mat, stacked_signals)
            perm_betas.append(model_perm.coef_.flatten()[act_perm_cols])
            
            perm_pred_signals = model_perm.predict(perm_predictor_mat)

            # calculate ΔR² of reduced predicted signals to the permuted predicted signals
            perm_masked_RSS = np.sum((masked_signals - perm_pred_signals[row_mask])**2)
            perm_full_RSS = np.sum((stacked_signals - perm_pred_signals)**2)

            perm_delta_R2_masked = (red_masked_RSS - perm_masked_RSS) / masked_TSS
            perm_delta_R2_full = (red_full_RSS - perm_full_RSS) / full_TSS
                
            perm_results['delta_R2_masked'].append(perm_delta_R2_masked)
            perm_results['delta_R2_full'].append(perm_delta_R2_full)
        
        event_stats[e] = {
            'delta_R2_masked': delta_R2_masked,
            'RSS_reduced_masked': red_masked_RSS,
            'delta_R2_full': delta_R2_full,
            'RSS_reduced_full': red_full_RSS,
            'delta_R2_event': delta_R2_event,
            'RSS_event_reduced': red_event_RSS,
            'RSS_event_full': full_event_RSS,
            'TSS_event': event_TSS,
            'wald_stat': W,
            'p_event': p_event,
            'perm_results': perm_results
        }
        
        kernel_stats[e] = {
            't': kernel_ts[e], 
            'full_betas': betas[event_cols],
            'se': se_betas[event_cols],
            'ci_low': cis_low[event_cols],
            'ci_high': cis_high[event_cols],
            't_stats': t_stats[event_cols],
            'p_vals': p_vals[event_cols],
            'cv_betas': [b[event_cols] for b in cv_betas],
            'perm_betas': perm_betas
        }

    # save model info
    model_info = {'predictors': all_sess_predictors, 'event_cols': event_col_idxs, 'beta_cov': cov_beta}

    return {'kernel_stats': kernel_stats, 'event_stats': event_stats, 'model_stats': model_stats, 
            'model_info': model_info}

def _check_convert_event_ts(event_ts_dict):
    for s in event_ts_dict.keys():
        for e in event_ts_dict[s].keys():
            if not isinstance(event_ts_dict[s][e], np.ndarray):
                event_ts_dict[s][e] = np.array(event_ts_dict[s][e])
                
    return event_ts_dict

def _build_event_matrix(event_ts_dict, kernel_delays, event_col_idxs, event_names, sig_ts, dt):

    T = len(sig_ts)
    P = 0    
    pred_rows = []
    pred_cols = []

    for e in event_names:

        # convert event timestamps into indices to build predictor matrix
        event_ts = event_ts_dict[e]
        event_ts = event_ts[~np.isnan(event_ts)]

        event_idxs = np.round((event_ts - sig_ts[0])/dt).astype(int)
        event_idxs = event_idxs[(event_idxs >= 0) & (event_idxs < T)]
        
        if len(event_idxs) == 0:
            continue
        
        # get rows and columns of predictor matrix that will have values of 1
        cols = event_col_idxs[e]
        delays = kernel_delays[e]
        
        delay_idxs = event_idxs[:, None] + delays[None, :]
        row_idxs = delay_idxs.ravel()
        col_idxs = np.tile(cols, len(event_idxs))

        valid = (row_idxs >= 0) & (row_idxs < T)

        pred_rows.append(row_idxs[valid])
        pred_cols.append(col_idxs[valid])
        
        P += len(cols)

    pred_rows = np.concatenate(pred_rows)
    pred_cols = np.concatenate(pred_cols)

    data = np.ones_like(pred_rows, dtype=np.float32)

    return sp.coo_matrix((data, (pred_rows, pred_cols)), shape=(T, P)).tocsr()

def _circular_shift_events(event_ts, t_min, t_max, min_shift=2):
    
    duration = t_max - t_min
    
    shift = 0
    while np.abs(shift) < min_shift:
        shift = np.random.uniform(-duration, duration)
    
    shifted_ts = event_ts + shift
    
    # wrap around
    shifted_ts = t_min + (shifted_ts - t_min) % duration
    
    return shifted_ts    

def _hac_covariance(predictor_mat, residuals, session_row_idxs, max_lag):
    """
    Compute Newey-West (HAC) covariance for beta.
    
    Returns:
        cov_beta (P x P)
    """

    _, P = predictor_mat.shape
    
    S = np.zeros((P, P))
    weights = 1 - np.arange(max_lag+1) / (max_lag+1)

    for sess_id, idx in session_row_idxs.items():
        
        X_s = predictor_mat[idx,:]
        r_s = residuals[idx]
        
        Z = X_s.multiply(r_s[:, None]).tocsr()
        
        # lag 0
        S += (Z.T @ Z).toarray()

        for k in range(1, min(max_lag+1, len(idx))):

            Z1 = Z[k:,:]
            Z2 = Z[:-k,:]
            
            G = (Z1.T @ Z2).toarray()
            
            S += weights[k] * (G + G.T)
    
    XtX = (predictor_mat.T @ predictor_mat).toarray()
    XtX_inv = np.linalg.inv(XtX)
    
    cov_beta = XtX_inv @ S @ XtX_inv
    
    return cov_beta

def _whiten_mats(residuals, predictors, signals, session_row_idxs):
    # perform AR(1) whitening on a session-by-session basis
    w_preds = []
    w_signals = []

    for sess_id, idx in session_row_idxs.items():
        # first estimate AR(1) decay constant
        r = residuals[idx]
        r -= np.mean(r)
        
        num = np.sum(r[1:] * r[:-1])
        den = np.sum(r[:-1]**2)
        
        rho = num/den
        
        # whiten signals and predictors
        x = predictors[idx,:]
        y = signals[idx]
        x_w = x.tolil(copy=True)
        y_w = y.copy()
        x_w[1:,:] = x[1:,:] - rho * x[:-1,:]
        y_w[1:] = y[1:] - rho * y[:-1]
        
        w_preds.append(x_w.tocsr()[1:,:])
        w_signals.append(y_w[1:])
        
    return sp.vstack(w_preds), np.concatenate(w_signals)

def _compute_resid_autocorr_lag(residuals, session_row_idxs, cutoff=0.05):
    
    # compute the number of timesteps that the signal is autocorrelated for

    ks = []

    for sess_id, idx in session_row_idxs.items():

        r = residuals[idx]
        r = r - np.mean(r)
        
        N = len(r)
        
        if N == 0:
            continue
        
        acf = sig.correlate(r, r, mode='full')
        rho = acf[N-1:] / acf[N-1]

        k = np.where(rho <= cutoff)[0]
        if len(k) > 0:
            k = k[0]
        else:
            k = N

        ks.append(k)

    return int(np.median(ks))

def _compute_signal_df_correction(signal, session_row_idxs):
    
    # compute effective degrees of freedom in the full signal residuals

    taus = []

    for sess_id, idx in session_row_idxs.items():

        s = signal[idx]
        s = s - np.mean(s)
        
        N = len(s)
        
        acf = sig.correlate(s, s, mode='full')
        rho = acf[N:] / acf[N-1]

        cutoff = np.where(rho <= 0.05)[0]
        if len(cutoff) > 0:
            rho = rho[:cutoff[0]]
        
        weights = (N - np.arange(1, len(rho)+1))/N
        tau_s = 1 + 2 * np.sum(rho.flatten()*weights)

        taus.append(tau_s)

    return np.median(taus)

def _compute_event_df(signal, event_pred_mat, session_row_idxs, kernel_len):
    
    # compute residual autocorrelation during the event to correct the degrees of freedom

    buffer = kernel_len // 2
    aligned_sig = []
    for sess_id, idx in session_row_idxs.items():

        r = signal[idx]
        r = r - np.mean(r)
        sess_events = event_pred_mat[idx,:]
        event_start_idxs, _ = (sess_events[:,0] == 1).nonzero()
        
        for start_idx in event_start_idxs:
            e_sig = np.full(kernel_len+2*buffer, np.nan)
            end_idx = min(start_idx+kernel_len+buffer, len(idx))
            e_sig[:end_idx-start_idx+buffer] = signal[start_idx-buffer:end_idx]
            aligned_sig.append(e_sig)

    aligned_sig = np.array(aligned_sig)  # shape: (n_events, K)

    rhos = []
    
    for lag in range(1, kernel_len):
        
        x = aligned_sig[:, :-lag].ravel()
        y = aligned_sig[:, lag:].ravel()
        
        nan_sel = np.isnan(x) | np.isnan(y)
        x = x[~nan_sel]
        y = y[~nan_sel]
        
        if len(x) < 2:
            break

        rho = np.corrcoef(x, y)[0,1]
        
        if rho <= 0:
            break
        
        rhos.append(rho)
    
    N = np.sum(event_pred_mat.getnnz(axis=1) > 0)
    weights = (N - np.arange(1, len(rhos)+1))/N
    tau = 1 + 2 * np.sum(np.array(rhos)*weights)
    
    k_eff = kernel_len / tau
    n_eff = N / tau
    
    df_event = n_eff - k_eff
    
    return {'df_event': df_event, 'n_eff': n_eff, 'k_eff': k_eff, 'tau': tau}