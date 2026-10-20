# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:33:47 2023

@author: tanne
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import utils
import inspect
import matplotlib.pyplot as plt
import scipy.signal as sig
import statsmodels.api as sm
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression
from scipy.stats import t, f, chi2, linregress
from scipy.signal import lfilter, fftconvolve
from scipy.linalg import toeplitz
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt
from scipy.fft import rfftfreq

# try:
#     import cupy as cp
#     if cp.cuda.is_available():
#         xp = cp
#         from cupyx.scipy.interpolate import interp1d
#         from cupyx.scipy.signal import butter, sosfiltfilt
#         gpu_available = True

# except:
#     xp = np
#     from scipy.interpolate import interp1d
#     from scipy.signal import butter, sosfiltfilt
#     gpu_available = False

import warnings

# def to_cupy(x):
#     if gpu_available and isinstance(x, np.ndarray):
#         return cp.asarray(x)
#     return x

# def to_numpy(x):
#     if gpu_available and isinstance(x, cp.ndarray):
#         return cp.asnumpy(x)
#     return x

# %% General Signal Processing

def fill_signal_nans(signal, extrapolate_edges=True):
    nan_idxs = np.isnan(signal)

    if any(nan_idxs) and not all(nan_idxs):
        # fill interpolated points with smoothed splines
        new_signal = pd.Series(signal).interpolate(method='pchip', limit_direction='both', limit_area='inside').to_numpy(copy=True)
        # extrapolate edges with multiscale linear extrapolations
        if extrapolate_edges:
            valid = ~np.isnan(new_signal)
            if np.sum(valid) != len(new_signal):
                x = np.arange(len(signal))
                valid_inds = x[valid]

                left_nan_span = np.sum(x < valid_inds[0])
                right_nan_span = np.sum(x > valid_inds[-1])

                new_signal = multiscale_pad(new_signal[valid], [left_nan_span, right_nan_span])

        return new_signal, nan_idxs
    else:
        return signal, nan_idxs

def decimate(signal, time, dec_factor, dt=None):
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

    if dec_factor == 1:
        return signal, time

    if dt is None:
        dt = np.mean(np.diff(time))

    pad_len = (-len(signal)) % dec_factor

    if pad_len > 0:
        reshape_signal = np.concatenate([signal, np.full(pad_len, np.nan)])
    else:
        reshape_signal = signal

    reshape_signal = np.reshape(reshape_signal, (-1, dec_factor))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dec_signal = np.nanmean(reshape_signal, axis=1)

    # enforce uniform timestamps (midpoint)
    start_t = time[0] + (dec_factor - 1)/2 * dt
    dec_time = start_t + np.arange(len(dec_signal)) * dec_factor * dt

    return dec_signal, dec_time

def filter_signal(signal, cutoff_f, sr, filter_type='lowpass', order=3, custom_pad=True, use_multiscale_pad=True):
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

    if all(nans):
        return signal

    sos = butter(order, cutoff_f, btype=filter_type, fs=sr, output='sos')

    if custom_pad:
        base_f = np.min(cutoff_f)
        pad_len = int((1/(2*base_f))*sr)

        span = None if use_multiscale_pad else pad_len

        pad_signal = multiscale_pad(signal, pad_len, spans=span)

        filtered_signal = sosfiltfilt(sos, pad_signal, padtype=None)
        filtered_signal = filtered_signal[pad_len:-pad_len]
    else:
        filtered_signal = sosfiltfilt(sos, signal, padtype='even')

    filtered_signal[nans] = np.nan

    return filtered_signal


def decompose_signal_fbands(signal, f_bands, sr, custom_pad=True):
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

    # prepad signal at once
    if custom_pad:
        all_f = np.unique(utils.flatten(f_bands))
        all_f = all_f[all_f != 0]
        base_f = np.min(all_f)
        pad_len = int((1/(2*base_f))*sr)
        pad_signal = multiscale_pad(signal, pad_len)
    else:
        pad_signal = signal

    # fill nans once
    pad_signal, nans = fill_signal_nans(pad_signal)

    decom_signal = np.zeros((len(f_bands), len(pad_signal)))

    for i, band in enumerate(f_bands):
        if len(band) == 1:
            decom_signal[i,:] = filter_signal(pad_signal, band[0], sr=sr, filter_type='highpass', custom_pad=False)
        elif band[0] == 0:
            if band[1] == np.inf:
                # [0, inf] means don't filter
                decom_signal[i,:] = pad_signal
            else:
                decom_signal[i,:] = filter_signal(pad_signal, band[1], sr=sr, filter_type='lowpass', custom_pad=False)
        else:
            decom_signal[i,:] = filter_signal(pad_signal, band, sr=sr, filter_type='bandpass', custom_pad=False)

    decom_signal[:, nans] = np.nan

    if custom_pad:
        decom_signal = decom_signal[:,pad_len:-pad_len]

    return decom_signal


def multiscale_pad(x, pad_width, spans=None, degree=1, end_anchor=None, anchor_weight=1):
    """
    Multiscale polynomial (linear default) extrapolation padding with distance-dependent gaussian weighting
    """

    x = np.asarray(x)

    N = len(x)

    if utils.is_list(pad_width):
        left_pad_len = pad_width[0]
        right_pad_len = pad_width[1]
    else:
        left_pad_len = pad_width
        right_pad_len = pad_width

    if utils.is_list(end_anchor):
        left_anchor = end_anchor[0]
        right_anchor = end_anchor[1]
    else:
        left_anchor = end_anchor
        right_anchor = end_anchor

    if spans is None:
        spans_left = 2 ** np.arange(degree, int(np.ceil(np.log2(min(left_pad_len, N)+1))) + 1)
        spans_right = 2 ** np.arange(degree, int(np.ceil(np.log2(min(right_pad_len, N)+1))) + 1)
    elif utils.is_scalar(spans):
        spans_left = [spans]
        spans_right = [spans]
    else:
        spans_left = spans
        spans_right = spans

    if len(spans_left) == 0:
        spans_left = [max(degree, left_pad_len)+1]
    if len(spans_right) == 0:
        spans_right = [max(degree, right_pad_len)+1]

    d_left = np.arange(-left_pad_len, 0)
    d_right = np.arange(1, right_pad_len + 1)
    t_right = d_right + N - 1

    # pad left
    if left_pad_len > 0:
        left_preds = []
        left_weights = []

        for span in spans_left:

            min_span = min(span, N)

            y = x[:min_span]
            t = np.arange(min_span)

            # ignore nans
            valid = ~np.isnan(y)

            if np.sum(valid) == 0:
                continue

            t_fit = t[valid]
            y_fit = y[valid]
            w = np.ones(len(t_fit))

            if left_anchor is not None:
                # place anchor one span beyond the first sample
                t_fit = np.concatenate([t_fit, [-span]])
                y_fit = np.concatenate([y_fit, [left_anchor]])
                w = np.concatenate([w, [anchor_weight]])

            coeffs = np.polyfit(t_fit, y_fit, degree, w=w)
            pred = np.polyval(coeffs, d_left)

            # gaussian distance weighting
            weight = np.exp(-(d_left**2) / (2*span**2))

            left_preds.append(pred)
            left_weights.append(weight)

        if len(left_preds) > 1:
            left_preds = np.array(left_preds)
            left_weights = np.array(left_weights)

            # make sure the longest span has some probability mass everywhere
            left_weights[-1,:] += 1e-12
            left_weights /= left_weights.sum(axis=1, keepdims=True)
            left_weights /= left_weights.sum(axis=0, keepdims=True)

            left_pad = np.sum(left_preds * left_weights, axis=0)
        elif len(left_preds) == 1:
            left_pad = left_preds[0]
        else:
            left_pad = []
    else:
        left_pad = []

    # pad right
    if right_pad_len > 0:
        right_preds = []
        right_weights = []

        for span in spans_right:

            min_span = min(span, N)

            y = x[-min_span:]
            t = np.arange(N - min_span, N)

            # ignore nans
            valid = ~np.isnan(y)

            if np.sum(valid) == 0:
                continue

            t_fit = t[valid]
            y_fit = y[valid]
            w = np.ones(len(t_fit))

            if right_anchor is not None:
                # place anchor one span beyond the final sample
                t_fit = np.concatenate([t_fit, [N - 1 + span]])
                y_fit = np.concatenate([y_fit, [right_anchor]])
                w = np.concatenate([w, [anchor_weight]])

            coeffs = np.polyfit(t_fit, y_fit, degree, w=w)
            pred = np.polyval(coeffs, t_right)

            # gaussian distance weighting
            weight = np.exp(-(d_right**2) / (2*span**2))

            right_preds.append(pred)
            right_weights.append(weight)

        if len(right_preds) > 1:
            right_preds = np.array(right_preds)
            right_weights = np.array(right_weights)

            # make sure the longest span has some probability mass everywhere
            right_weights[-1,:] += 1e-12
            right_weights /= right_weights.sum(axis=1, keepdims=True)
            right_weights /= right_weights.sum(axis=0, keepdims=True)

            right_pad = np.sum(right_preds * right_weights, axis=0)
        elif len(right_preds) == 1:
            right_pad = right_preds[0]
        else:
            right_pad = []
    else:
        right_pad = []

    return np.concatenate([left_pad, x, right_pad])


def correlate(x, y, dt, neg_lag=2, pos_lag=2, batch_size=200, t_sel=None, mode='correlation'):
    '''
    Compute Pearson cross-correlation between two signals for lags in [-neg_lag, +pos_lag],
    ignoring NaNs in either signal.

    Returns correlation values in [-1, 1] for each lag along with the associated lag value

    '''

    if len(x) != len(y):
        raise ValueError('x and y must be the same length')

    t = len(x)
    if t_sel is None:
        t_sel = np.full(t, True)

    # mask out anything not in t_sel
    centered_x = x.copy()
    centered_y = y.copy()
    centered_x[~t_sel] = np.nan
    centered_y[~t_sel] = np.nan

    if mode == 'correlation':
        centered_x = utils.z_score(centered_x)
        centered_y = utils.z_score(centered_y)
    elif mode == 'covariance':
        centered_x = x - np.nanmean(x)
        centered_y = y - np.nanmean(y)
    else:
        raise ValueError('mode can only be correlation or covariance')

    neg_lag = utils.convert_to_multiple(neg_lag, dt)
    pos_lag = utils.convert_to_multiple(pos_lag, dt)
    lag_steps = np.arange(-int(np.round(neg_lag/dt)), int(np.round(pos_lag/dt))+1)

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


def calc_power_spectra(signal, dt, f_min=0.005, t_sel=None):
    '''
    Calculate the power spectrum over contiguous non-NaN time intervals.

    If t_sel is provided, timepoints outside t_sel are treated as NaNs and
    break the signal into separate intervals.
    '''

    max_nperseg = round(1/dt*2/f_min)
    if max_nperseg < 1:
        raise ValueError('dt and f_min must produce a max_nperseg of at least 1')

    tmp_sig = np.array(signal, dtype=float, copy=True)

    if t_sel is not None:
        t_sel = np.asarray(t_sel, dtype=bool)

        if t_sel.shape != tmp_sig.shape:
            raise ValueError('t_sel must be the same shape as signal')

        tmp_sig[~t_sel] = np.nan

    valid = ~np.isnan(tmp_sig)

    if not np.any(valid):
        raise ValueError('No valid samples selected for power spectrum calculation')

    padded_valid = np.concatenate([np.array([False]), valid, np.array([False])])
    valid_diff = np.diff(padded_valid.astype(int))
    start_idxs = np.where(valid_diff == 1)[0]
    end_idxs = np.where(valid_diff == -1)[0]
    seg_lens = end_idxs - start_idxs

    freqs = rfftfreq(max_nperseg, d=dt)
    all_ps = np.full((len(seg_lens), len(freqs)), np.nan)
    weights = seg_lens[:,None].astype(float)

    for i, (start_idx, end_idx, seg_len) in enumerate(zip(start_idxs, end_idxs, seg_lens)):

        nperseg = min(seg_len, max_nperseg)
        if nperseg < 2:
            continue

        _, seg_ps = sig.welch(tmp_sig[start_idx:end_idx], fs=1/dt,
                                      nperseg=nperseg, nfft=max_nperseg, scaling='density')

        # mask unsupported frequencies
        seg_ps[freqs < 1/(nperseg*dt)] = np.nan

        all_ps[i,:] = seg_ps

    ps = utils.weighted_mean(all_ps, weights, axis=0)

    return freqs, ps


def calc_coherence_phase(signal1, signal2, dt, f_min=0.005, t_sel=None):
    '''
    Calculate magnitude-squared coherence and cross-spectral phase difference.

    NaNs in either signal and timepoints outside t_sel break both signals
    into matching contiguous intervals. Spectra are duration-weighted before
    calculating coherence and phase.

    Returns freqs, coherence, phase_difference. Phase is signal2 minus signal1
    in radians, wrapped to [-pi, pi].
    '''

    if not np.isfinite(dt) or not np.isfinite(f_min) or dt <= 0 or f_min <= 0:
        raise ValueError('dt and f_min must be finite and positive')

    if signal1.ndim != 1 or signal1.shape != signal2.shape:
        raise ValueError('Signals must be one-dimensional and the same shape')

    max_nperseg = round(1/dt*2/f_min)
    if max_nperseg < 1:
        raise ValueError('dt and f_min must produce a max_nperseg of at least 1')

    tmp_sig1 = np.array(signal1, dtype=float, copy=True)
    tmp_sig2 = np.array(signal2, dtype=float, copy=True)

    valid = np.isfinite(tmp_sig1) & np.isfinite(tmp_sig2)

    if t_sel is not None:
        t_sel = np.asarray(t_sel, dtype=bool)
        if t_sel.shape != tmp_sig1.shape:
            raise ValueError('t_sel must be the same shape as the signals')
        valid &= t_sel

    if not np.any(valid):
        raise ValueError('No valid samples selected for coherence calculation')

    padded_valid = np.concatenate([np.array([False]), valid, np.array([False])])
    valid_diff = np.diff(padded_valid.astype(int))
    start_idxs = np.where(valid_diff == 1)[0]
    end_idxs = np.where(valid_diff == -1)[0]
    seg_lens = end_idxs - start_idxs

    freqs = rfftfreq(max_nperseg, d=dt)
    all_spectra = np.full((len(seg_lens), 4, len(freqs)), np.nan)
    weights = seg_lens[:,None,None].astype(float)
    window_support = np.zeros(len(freqs), dtype=int)

    for i, (start_idx, end_idx, seg_len) in enumerate(zip(start_idxs, end_idxs, seg_lens)):

        nperseg = min(seg_len, max_nperseg)
        if nperseg < 2:
            continue

        x = tmp_sig1[start_idx:end_idx]
        y = tmp_sig2[start_idx:end_idx]
        kwargs = dict(fs=1/dt, nperseg=nperseg, nfft=max_nperseg, scaling='density')

        _, pxx = sig.welch(x, **kwargs)
        _, pyy = sig.welch(y, **kwargs)
        _, pxy = sig.csd(x, y, **kwargs)

        # store iaginary component separately for dtype compatibility
        all_spectra[i,:,:] = np.array([pxx, pyy, pxy.real, pxy.imag])
        all_spectra[i,:,freqs < 1/(nperseg*dt)] = np.nan

        # Count number of supporting windows per frequency
        n_overlap = nperseg // 2
        n_windows = 1 + (seg_len - nperseg) // (nperseg - n_overlap)
        supported = np.all(np.isfinite(all_spectra[i,:,:]), axis=0)
        window_support[supported] += n_windows

    pxx, pyy, pxy_real, pxy_imag = utils.weighted_mean(all_spectra, weights, axis=0)
    pxy = pxy_real + 1j*pxy_imag
    denominator = pxx*pyy

    coherence = np.full(len(freqs), np.nan)
    np.divide(np.abs(pxy)**2, denominator, out=coherence, where=np.isfinite(denominator) & (denominator > 0))
    coherence[window_support <= 2] = np.nan

    phase_difference = np.angle(pxy)
    phase_difference[(denominator <= 0) | (np.abs(pxy) == 0)] = np.nan

    return freqs, coherence, phase_difference

# %% dF/F Methods

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


def fit_signal(signal_to_fit, signal, t, vary_t=False, fit_bands=False, f_bands=None, allow_neg_coeffs=False, interp_coeffs=False):
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
        form_str = 'lambda x, a, t, b: a*x[0,:] + t*x[1,:] + b'
        s_to_fit = np.vstack((signal_to_fit[None,~nans], t[None,~nans]))
        if allow_neg_coeffs:
            bounds = ([-np.inf, -np.inf, -np.inf],
                      [ np.inf,  np.inf,  np.inf])
        else:
            bounds = ([      0, -np.inf, -np.inf],
                      [ np.inf,  np.inf,  np.inf])

    elif fit_bands:
        if f_bands is None:
            f_bands = [[0,0.5], [0.5,5], [5]]

        # dynamically create the formula to optimize
        param_names = ['b{}'.format(i) for i in range(len(f_bands))]

        if vary_t:
            param_names.append('t')

        expr = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names)]) + ' + c'

        param_names = param_names + ['c']

        # Build the lambda string and evaluate it
        form_str = 'lambda x, {}: {}'.format(', '.join(param_names), expr)

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

        # if we want interpretable coefficients, ensure each frequency band has equal variance
        if interp_coeffs:
            s_to_fit = utils.z_score(s_to_fit, axis=1)

        if vary_t:
            s_to_fit = np.vstack([s_to_fit, t[None,:]])

        s_to_fit = s_to_fit[:,~nans]

    else:
        form_str = 'lambda x, a, b: a*x + + b'
        s_to_fit = signal_to_fit[~nans]
        if allow_neg_coeffs:
            bounds = ([-np.inf, -np.inf],
                      [ np.inf,  np.inf])
        else:
            bounds = ([      0, -np.inf],
                      [ np.inf,  np.inf])

    form = eval(form_str)
    params = curve_fit(form, s_to_fit, signal[~nans], bounds=bounds)[0]

    fitted_signal = np.full_like(signal_to_fit, np.nan)
    fitted_signal[~nans] = form(s_to_fit, *params)

    return fitted_signal, {'params': params, 'formula': form_str}


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

# %% PSTH Methods

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

# %% Source Extraction Methods

def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """
    Copy of GetSn from caiman.source_extraction.cnmf
    Estimate noise power through the power spectral density over the range of large frequencies

    Args:
        fluor    : nparray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.

        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is averaged

        method   : string
            method of averaging: Mean, median, exponentiated mean of logvalues (default)

    Returns:
        sn       : noise standard deviation
    """

    ff, Pxx = sig.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]

    if   method == 'mean':
        return np.sqrt(np.mean(Pxx_ind / 2))
    elif method == 'median':
        return np.sqrt(np.median(Pxx_ind / 2))
    elif method == 'logmexp':
        return np.sqrt(np.exp(np.mean(np.log(Pxx_ind / 2))))
    else:
        raise Exception('Invalid method requested for GetSn')

def get_ar_noise(signal):
    tmp_x, nans = fill_signal_nans(signal, extrapolate_edges=False)
    ext_nans = np.isnan(tmp_x)
    return GetSn(tmp_x[~ext_nans])

def get_signal_ar_params(signal, p, nfit_lags):
    sn = get_ar_noise(signal)
    ac, lags = correlate(signal, signal, 1, neg_lag=0, pos_lag=nfit_lags, mode='covariance')

    return get_acf_ar_params(ac, lags, p, nfit_lags, sn=sn)

def get_acf_ar_params(ac, lags, p, nfit_lags, sn=0):
    # Inspired by the Caiman get_signal_timescale method

    ac = np.array(ac)
    if len(ac.shape) == 1:
        ac = ac[:, np.newaxis]

    lag_zero = lags == 0
    if not any(lag_zero):
        raise ValueError('The provided autocorrelation does not include lag 0')

    zero_idx = np.argmax(lag_zero)

    pos_ac = ac[zero_idx:]

    if nfit_lags < p:
        nfit_lags = p

    if len(pos_ac) < nfit_lags+1:
        raise ValueError('Need at least {} positive lags beyond lag 0'.format(nfit_lags))

    A = toeplitz(c=np.ravel(pos_ac[np.arange(nfit_lags)]),
                 r=np.ravel(pos_ac[np.arange(p)])) - sn**2 * np.eye(nfit_lags, p)
    b = pos_ac[1:nfit_lags+1]

    # remove nans
    valid = ~np.isnan(b).ravel()

    g = np.linalg.lstsq(A[valid,:], b[valid], rcond=None)[0]

    # stabilize unstable roots
    gr = _get_ar_roots(g.flatten())
    mask = np.abs(gr) > 1
    if any(mask):
        print('Found unstable roots')
        gr[mask] = gr[mask] / np.abs(gr[mask]) * 0.99

    g = -np.poly(gr)[1:]

    return g.flatten()

def extract_ar_source(x, g):
    # inverse filter to recover the source of the AR process
    a = np.concatenate([[1], -np.array(g)])
    source = lfilter(a, [1], x)
    # ignore the first p samples since they suffer from edge effects
    source[:len(g)] = np.nan

    return source

def get_ar_modes(g, dt):
    gr = _get_ar_roots(g)

    modes = []

    for r in gr:
        # only include one complex conjugate pair
        if np.imag(r) > 0 or np.imag(r) == 0:
            mag = np.abs(r)
            theta = np.angle(r)
            tau = -dt/np.log(mag)
            freq = theta/(2*np.pi*dt)

            modes.append({'root': r, 'mag': mag, 'theta': theta, 'tau': tau, 'freq': freq})

    return modes

def get_ar_params_from_modes(modes, dt):

    if not utils.is_list(modes):
        modes = [modes]

    roots = []

    for mode in modes:
        tau = mode['tau']
        freq = mode['freq']

        mag = np.exp(-dt/tau)

        if np.isclose(freq, 0):
            roots.append(mag)
        else:
            theta = 2*np.pi*freq*dt

            r = mag*np.exp(1j*theta)

            roots.extend([r, np.conj(r)])

    g = -np.poly(roots)[1:]

    # remove tiny imaginary roundoff
    g = np.real_if_close(g)

    return g.flatten()

def _get_ar_roots(g):
    return np.roots(np.concatenate([[1], -np.array(g)]))

def simulate_ar_acf(g, nlags):
    p = len(g)

    rho = np.zeros(nlags+1)
    rho[0] = 1

    # calculate initial values before recursion
    if p == 1:
        rho[1] = g[0]
    else:
        # solve only for rho[1:p]
        A = np.zeros((p-1, p-1))
        b = np.zeros(p-1)

        # build Yule-Walker system with rho[0]=1 fixed
        for k in range(1, p):
            row = k-1
            # rho(k) on left-hand side
            A[row, k-1] = 1.0

            for i in range(1, p+1):
                lag = abs(k-i)

                coeff = g[i-1]

                if lag == 0:
                    # rho(0)=1 moves to RHS
                    b[row] += coeff

                else:
                    # rho(lag) is an unknown
                    A[row, lag-1] -= coeff

        rho[1:p] = np.linalg.solve(A, b)

    for k in range(p, nlags+1):
        rho[k] = np.dot(g, rho[k-p:k][::-1])

    return rho

# %% ICA Methods

def rectify_ica_output(A, S):
    '''
    Rectify the columns of mixing matrix and independent components so that the largest values in each IC are positive

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
    rect_mult = []
    for i in range(n_ic):
        max_col_val_idx = np.argmax(np.abs(A[:,i]))
        if A[max_col_val_idx, i] < 0:
            rect_mult.append(-1)
        else:
            rect_mult.append(1)

    rect_mult = np.array(rect_mult)[None,:]

    A = A*rect_mult
    S = S*rect_mult

    return A, S, rect_mult


def reconstruct_ica(A, S, full=False, component_idx=None):
    """
    Compute explained power of ICA components.

    Parameters
    ----------
    A : (n_channels, n_ic)  mixing matrix
    S : (T, n_ic) IC components
    full : Whether to return the full reconstruction summed over components
    component_idx : if full is False, whether to return the reconstruction for a single component

    Returns
    -------
    reconstruction : (T, n_channels) if full is True or component_idx is supplied. (T, n_channels, n_ic) if full is False and component_idx is not supplied
        signal reconstruction from each IC component
    """

    # calculate reconstruction of each channel from each IC
    if full:
        return np.einsum('ti,ci->tc', S, A)
    elif component_idx is None:
        return np.einsum('ti,ci->tci', S, A)
    else:
        return np.einsum('t,c->tc', S[:, component_idx], A[:, component_idx])

def ic_explained_variance(A, S, X):
    """
    Compute explained variance (R^2) of ICA components.

    Parameters
    ----------
    A : (n_regions, n_ic)  mixing matrix
    S : (T, n_ic) IC components
    X : (T, n_regions) Data matrix

    Returns
    -------
    dict containing:
        'ev_total': fraction of total dataset variance explained by whole IC reconstruction
        'ev_region': (n_regions,) fractions of total region variance explained by whole IC reconstruction
        'ev_component': (n_ic,) fractions of total variance explained by each individual IC reconstruction
        'ev_component_region': (n_ic, n_regions) fractions of total region variance explained by each individual IC reconstruction
        'ev_component_marg': (n_ic,) marginal unique variance explained by each individual IC reconstruction
        'ev_component_region_marg': (n_ic, n_regions) marginal unique region variance explained by each individual IC reconstruction
    """

    n_regions, n_ic = A.shape

    X_mean = np.mean(X, axis=0, keepdims=True)
    data_var = (X - X_mean)**2

    data_var_region = np.sum(data_var, axis=0)
    data_var_full = np.sum(data_var_region)

    X_hat = reconstruct_ica(A, S, full=True)

    sq_err = (X - X_hat)**2

    ev_total = 1 - np.sum(sq_err) / data_var_full
    ev_region = 1 - np.sum(sq_err, axis=0) / data_var_region

    ev_component = np.zeros(n_ic)
    ev_component_region = np.zeros((n_ic, n_regions))

    ev_component_marg = np.zeros(n_ic)
    ev_component_region_marg = np.zeros((n_ic, n_regions))

    for i in range(n_ic):
        X_hat_i = reconstruct_ica(A, S, component_idx=i)
        X_hat_marg_i = X_hat - X_hat_i

        sq_err_i = (X - X_hat_i)**2
        sq_err_marg_i = (X - X_hat_marg_i)**2

        ev_component[i] = 1 - np.sum(sq_err_i) / data_var_full
        ev_component_region[i,:] = 1 - np.sum(sq_err_i, axis=0) / data_var_region

        ev_component_marg[i] = ev_total - (1 - np.sum(sq_err_marg_i) / data_var_full)
        ev_component_region_marg[i,:] = ev_region - (1 - np.sum(sq_err_marg_i, axis=0) / data_var_region)

    return {'ev_total': ev_total, 'ev_region': ev_region,
            'ev_component': ev_component, 'ev_component_region': ev_component_region,
            'ev_component_marg': ev_component_marg, 'ev_component_region_marg': ev_component_region_marg}


def get_overlap_segments(reference, kernel, shift, min_overlap=5):
    """
    Return the overlapping segments of the reference and kernel.
    """

    n_ref_lags = reference.shape[1]
    n_kern_lags = kernel.shape[1]

    kern_start = max(0, -shift)
    kern_end = min(n_kern_lags, n_ref_lags-shift)

    ref_start = max(0, shift)
    ref_end = min(n_ref_lags, n_kern_lags+shift)

    if ref_end - ref_start < min_overlap or kern_end - kern_start < min_overlap:
        return None, None, None

    return (reference[:, ref_start:ref_end],
            kernel[:, kern_start:kern_end],
            {'ref_start': ref_start, 'ref_end': ref_end, 'kern_start': kern_start, 'kern_end': kern_end})

def get_kernel_energy_centroid(kernel, use_deriv=True):
    """
    Get the energy centroid of the kernel
    """
    if use_deriv:
        d_kern = gaussian_filter1d(kernel, sigma=0.5, order=1, axis=-1)
        # rescale the first derivative to match the RMS of the raw kernel
        kern_rms = np.sqrt(np.mean(kernel**2))
        d_kern_rms = np.sqrt(np.mean(d_kern**2))
        d_kern *= kern_rms/d_kern_rms

        stack_kern = np.vstack((kernel, d_kern))
        energy = np.sum(stack_kern**2, axis=0)
    else:
        energy = np.sum(kernel**2, axis=0)
        stack_kern = kernel

    x = np.arange(kernel.shape[1])
    centroid_idx = int(round(np.sum(energy * x) / np.sum(energy)))
    return centroid_idx, energy, stack_kern


def best_kernel_alignment(reference, kernel, search_window=20, min_overlap=5, penalize_shift=False, shift_penalty=0.5,
                          overlap_prop_thresh=0.5, equiv_corr_diff=0.002, use_ev=True, use_deriv=False, use_centroid=True):
    """
    Find the shift in a window around the centroids of each kernel that maximizes the normalized
    Frobenius correlation and energy overlap

    Parameters
    ----------
    reference : (n_regions, n_ref_lags)
    kernel : (n_regions, n_kern_lags)

    Returns
    -------
    corr : float
    shift : int
    """

    ref_len = reference.shape[1]
    kern_len = kernel.shape[1]

    ref_centroid, ref_energy, stack_ref = get_kernel_energy_centroid(reference, use_deriv=use_deriv)
    kern_centroid, kern_energy, stack_kern = get_kernel_energy_centroid(kernel, use_deriv=use_deriv)

    if use_centroid:
        centroid_shift = int(round(ref_centroid - kern_centroid))
        all_shifts = np.arange(centroid_shift-search_window, centroid_shift+search_window+1)
    else:
        all_shifts = np.arange(-search_window, search_window+1)

    all_corrs = np.zeros(all_shifts.shape)
    all_evs = np.zeros(all_shifts.shape)
    ref_overlaps = np.zeros(all_shifts.shape)
    kern_overlaps = np.zeros(all_shifts.shape)

    full_ref_energy = np.sum(ref_energy)
    full_kern_energy = np.sum(kern_energy)

    for i, shift in enumerate(all_shifts):

        # ref_slice, kern_slice, slice_details = get_overlap_segments(reference, kernel, shift, min_overlap=min_overlap)
        ref_slice, kern_slice, slice_details = get_overlap_segments(stack_ref, stack_kern, shift, min_overlap=min_overlap)

        if ref_slice is None or kern_slice is None:
            continue

        slice_ref_ss = np.sum(ref_slice**2)
        slice_kern_ss = np.sum(kern_slice**2)
        slice_ref_energy = np.sum(ref_energy[slice_details['ref_start']:slice_details['ref_end']])
        slice_kern_energy = np.sum(kern_energy[slice_details['kern_start']:slice_details['kern_end']])

        corr = np.sum(ref_slice * kern_slice) / np.sqrt(slice_ref_ss * slice_kern_ss)
        ev = corr**2

        all_corrs[i] = corr
        all_evs[i] = ev
        ref_overlaps[i] = slice_ref_energy/full_ref_energy
        kern_overlaps[i] = slice_kern_energy/full_kern_energy

    # find the best correlation where at least one of the kernels has energy proportion greater than the threshold in the overlapping segments
    valid_idxs = np.flatnonzero((ref_overlaps >= overlap_prop_thresh) | (kern_overlaps >= overlap_prop_thresh))

    # find max valid correlation
    if use_ev:
        metric = all_evs
    else:
        metric = all_corrs

    valid_metric = metric[valid_idxs]

    # find if any values are within an equivalence threshold to prioritize the shift with the greatest overlap
    equiv_idxs = valid_idxs[max(valid_metric) - valid_metric <= equiv_corr_diff]

    # Break ties using the largest overlap of energies
    best_idx = equiv_idxs[np.argmax(np.maximum(ref_overlaps[equiv_idxs], kern_overlaps[equiv_idxs]))]
    best_corr = metric[best_idx]
    best_shift = all_shifts[best_idx]

    # only penalize the correlation value after we have found the maximum unpenalized correlation
    if penalize_shift:
        best_corr *= max(ref_overlaps[best_idx], kern_overlaps[best_idx])**shift_penalty

    return best_corr, best_shift

def estimate_scale(reference, kernel, shift):
    """
    Least-squares scaling over overlapping samples.
    """

    ref_slice, kern_slice, _ = get_overlap_segments(reference, kernel, shift)

    mask = (~np.isnan(ref_slice) & ~np.isnan(kern_slice))

    if not np.any(mask):
        return 1.0

    denom = np.sum(kern_slice[mask]**2)

    if denom == 0:
        return 1.0

    return np.sum(ref_slice[mask] * kern_slice[mask]) / denom


def build_template(kernels, shifts, scales, weights=None):
    """
    Construct the long template from a set of aligned kernels.

    Parameters
    ----------
    kernels : list of (n_regions, n_lags)
    shifts : (N,)
        Integer shifts relative to the reference.
    scales : (N,)

    Returns
    -------
    template : (n_regions, full_length)
    """

    if weights is None:
        weights = np.ones_like(scales)

    n_regions = kernels[0].shape[0]

    left = shifts.min()

    starts = shifts - left
    ends = np.array([start + kern.shape[1] for start, kern in zip(starts, kernels)])

    full_length = ends.max()

    template = np.zeros((n_regions, full_length))
    weight_sum = np.zeros((n_regions, full_length))

    for kernel, start, scale, weight in zip(kernels, starts, scales, weights):
        stop = start + kernel.shape[1]

        template[:, start:stop] += scale * kernel * weight
        weight_sum[:, start:stop] += weight

    template /= weight_sum

    return template


def average_cluster(kernels, weights, search_window=20, overlap_prop_thresh=0.5, use_deriv=True, use_centroid=True):
    """
    Align all kernels to a representative and reconstruct a long kernel.

    Parameters
    ----------
    kernels : list of (n_regions, n_lags)

    similarities : Cluster similarity matrix if using medoid.

    Returns
    -------
    template : (n_regions, full_length)

    """

    n = len(kernels)

    kernel_order = np.argsort(weights)[::-1]

    ref_idx = kernel_order[0]

    shifts = np.zeros(n, dtype=int)
    scales = np.ones(n)

    added_kerns = [ref_idx]

    template = kernels[ref_idx].copy()

    for idx in kernel_order[1:]:

        # find best shift and scale for the added kernel
        _, shift = best_kernel_alignment(template, kernels[idx], search_window=search_window, overlap_prop_thresh=overlap_prop_thresh,
                                         use_deriv=use_deriv, penalize_shift=False, use_centroid=use_centroid)
        scales[idx] = estimate_scale(template, kernels[idx], shift)

        added_kerns.append(idx)

        # update all shifts to be referenced to first template index
        shifts[idx] = shift
        if shift < 0:
            shifts -= shift

        # Rebuild the template
        template = build_template([kernels[i] for i in added_kerns], shifts[added_kerns], scales[added_kerns], weights=weights[added_kerns])

        # refine the scaling and shifts and rebuild the template
        for i in added_kerns:
            scales[i] = estimate_scale(template, kernels[i], shifts[i])

        template = build_template([kernels[i] for i in added_kerns], shifts[added_kerns], scales[added_kerns], weights=weights[added_kerns])

    return template, {'shifts': shifts, 'scales': scales}


def create_average_templates(kernels, labels, weights, search_window=20, overlap_prop_thresh=0.5, use_deriv=False, use_centroid=True):

    # create average templates per cluster
    templates = []
    shifts = np.zeros(labels.shape, dtype=int)
    scales = np.zeros(labels.shape)

    for cluster in np.unique(labels):

        idx = np.where(labels == cluster)[0]

        cluster_kerns = [kernels[i] for i in idx]

        cluster_weights = weights[idx]

        template, fit_details = average_cluster(cluster_kerns, cluster_weights, search_window=search_window, overlap_prop_thresh=overlap_prop_thresh,
                                                use_deriv=use_deriv, use_centroid=use_centroid)

        templates.append(template)
        shifts[idx] = fit_details['shifts']
        scales[idx] = fit_details['scales']

    return templates, {'shifts': shifts, 'scales': scales}

def calc_similarity(kernels, overlap_prop_thresh=0.5, search_window=20, shift_penalty=0.5, use_deriv=True, use_centroid=True):

    n = len(kernels)

    similarity = np.eye(n)

    # first cluster all individual kernels
    for i in range(n):
        for j in range(i + 1, n):

            corr, _ = best_kernel_alignment(kernels[i], kernels[j], search_window=search_window, overlap_prop_thresh=overlap_prop_thresh,
                                            penalize_shift=True, use_deriv=use_deriv, use_centroid=use_centroid, shift_penalty=shift_penalty)

            similarity[i, j] = similarity[j, i] = corr

    return similarity

def cluster_similarity(similarity, threshold=0.1, linkage_method='average', cluster_criterion='distance'):

    distance = 1 - similarity

    Z = linkage(squareform(distance), method=linkage_method)

    labels = fcluster(Z, t=threshold, criterion=cluster_criterion)

    return labels


def cluster_kernels(kernels, similarity_threshold=0.1, overlap_prop_thresh=0.5, search_window=20, shift_penalty=0.5,
                    use_deriv=True, use_centroid=True, linkage_method='average', cluster_criterion='distance', recluster=True):
    """
    Cluster IC kernels and reconstruct one long template kernel per cluster.

    Parameters
    ----------
    kernels : list of (n_regions, n_lags)

    Returns
    -------
    dict
    """

    similarity = calc_similarity(kernels, overlap_prop_thresh=overlap_prop_thresh, search_window=search_window,
                                 shift_penalty=shift_penalty, use_deriv=use_deriv, use_centroid=use_centroid)

    labels = cluster_similarity(similarity, threshold=similarity_threshold, linkage_method=linkage_method, cluster_criterion=cluster_criterion)

    templates, fit_info = create_average_templates(kernels, labels, similarity, search_window=search_window, overlap_prop_thresh=overlap_prop_thresh)

    # next cluster templates to see if any more should be merged
    if recluster:

        temp_similarity = calc_similarity(templates, overlap_prop_thresh=overlap_prop_thresh, search_window=search_window,
                                          shift_penalty=shift_penalty, use_deriv=use_deriv, use_centroid=use_centroid)

        temp_labels = cluster_similarity(temp_similarity, threshold=similarity_threshold, linkage_method=linkage_method, cluster_criterion=cluster_criterion)

        refine_labels = np.zeros(labels.shape, dtype=int)
        for cluster in np.unique(temp_labels):
            temp_idxs = np.where(temp_labels == cluster)[0]
            ref_idxs = utils.flatten([np.where(labels == i)[0] for i in temp_idxs])
            refine_labels[ref_idxs] = cluster

        refine_templates, fit_info = create_average_templates(kernels, refine_labels, similarity, search_window=search_window, overlap_prop_thresh=overlap_prop_thresh)

        return refine_templates, refine_labels, fit_info
    else:
        return templates, labels, fit_info


def stack_centered_kernels(kernels, shifts=None, trunc_min=None, trunc_max=None, smooth=True, truncate=True,
                           trunc_thresh=0.005, min_span=5, stride=2, eps=1e-5, debug_plot=False, extrapolate_edges=False):
    """
    Pad variable-length kernels into a single tensor while aligning the
    energy centroid of each kernel to the center of the output.

    Parameters
    ----------
    kernels : sequence of array-like
        Each element has shape (n_regions, kernel_length).

    Returns
    -------
    stacked : (n_kernels, n_regions, max_length) tensor
    """

    n_regions = kernels[0].shape[0]
    n_kernels = len(kernels)
    max_len = max(k.shape[1] for k in kernels)
    # first expand the stacked kernels to double the longest kernel so no initial template is clipped when centering
    out_len = int(2*max_len)
    center_idx = int((out_len - 1) / 2)

    stacked = np.full((n_kernels, n_regions, out_len), np.nan)

    if trunc_min is None:
        trunc_min = out_len
    if trunc_max is None:
        trunc_max = 0

    if shifts is None:
        force_shifts = False
        shifts = np.zeros(len(kernels), dtype=int)
    else:
        force_shifts = True

    for i, kern in enumerate(kernels):
        L = kern.shape[1]

        if force_shifts:
            shift = shifts[i]
        else:
            centroid_idx, energy, _ = get_kernel_energy_centroid(kern, use_deriv=True)

            # # use the combined raw kernel and first derivative to find the center of the kernel
            # # find the first derivative
            # d_kern = gaussian_filter1d(kern, sigma=0.5, order=1, axis=-1)
            # # rescale the first derivative to match the RMS of the raw kernel
            # kern_rms = np.sqrt(np.mean(kern**2))
            # d_kern_rms = np.sqrt(np.mean(d_kern**2))
            # d_kern *= kern_rms/d_kern_rms

            # # stack both and compute energy profile across time
            # # stack_kern = np.vstack([kern, d_kern])
            # comb_energy = (kern**2 + d_kern**2)/2
            # # energy = np.sum(stack_kern**2, axis=0)
            # # energy = np.sum(comb_energy, axis=0)
            # # comb_energy = np.sum(stack_kern**2, axis=0)
            # sum_comb_energy = np.sum(comb_energy, axis=0)
            # # d_energy = np.sum(d_kern**2, axis=0)

            # x = np.arange(L)
            # centroid_idx = int(round(np.sum(sum_comb_energy * x) / np.sum(sum_comb_energy)))
            # # d_idx = int(round(np.sum(d_energy * x) / np.sum(d_energy)))

            shift = center_idx - centroid_idx

            if truncate:

                # starting from the center index, find the start and end of the kernel as the first and last index
                # past which a linear trend explains 1-trunc_thresh of the variance
                kern_start = centroid_idx - (min_span//2)
                kern_end = kern_start + min_span

                # first rescale the kernel to unit RMS so thresholding is consistent across scales
                rescale_kern = kern / np.sqrt(np.mean(kern**2))

                resid_diff = np.zeros(kern.shape[1])

                # x = np.arange(L)
                # full_rss = np.zeros_like(kern)
                # for j in range(kern.shape[0]):
                #     y_lin_fit = linregress(x, rescale_kern[j,:])
                #     y_hat = y_lin_fit.intercept + y_lin_fit.slope*x
                #     full_rss[j,:] = (rescale_kern[j,:] - y_hat)**2

                # tot_mse = np.sum(np.mean(full_rss, axis=1))

                # find kern start
                # mean = np.sum(comb_energy*rescale_kern, axis=1, keepdims=True) / np.sum(comb_energy, axis=1, keepdims=True)
                mean = np.mean(rescale_kern, axis=1, keepdims=True)
                tot_mse = np.sum(np.mean((rescale_kern-mean)**2, axis=1))
                # prev_mse = np.sum(np.var(rescale_kern[:,:kern_start], axis=1)*kern_start)

                while kern_start > 0:
                    x = np.arange(kern_start)
                    y_rss = np.zeros_like(kern[:,:kern_start])
                    for j in range(kern.shape[0]):
                        y_lin_fit = linregress(x, rescale_kern[j,:kern_start])
                        y_hat = y_lin_fit.intercept + y_lin_fit.slope*x
                        y_rss[j,:] = (rescale_kern[j,:kern_start] - y_hat)**2

                    new_mse = np.sum(np.mean(y_rss, axis=1))

                    resid_diff[kern_start] = new_mse/tot_mse

                    # go to the idx past the threshold
                    kern_start = max(0, kern_start-stride)

                    if new_mse/tot_mse < trunc_thresh:
                        break

                # find kern end
                # prev_mse = np.sum(np.var(rescale_kern[:,kern_end:], axis=1)*(L-kern_end))

                while kern_end < kern.shape[1]:
                    x = np.arange(kern.shape[1]-kern_end)
                    y_rss = np.zeros_like(kern[:,kern_end:])
                    for j in range(kern.shape[0]):
                        y_lin_fit = linregress(x, rescale_kern[j,kern_end:])
                        y_hat = y_lin_fit.intercept + y_lin_fit.slope*x
                        y_rss[j,:] = (rescale_kern[j,kern_end:] - y_hat)**2

                    new_mse = np.sum(np.mean(y_rss, axis=1))

                    resid_diff[kern_end] = new_mse/tot_mse

                    # go to the idx past the threshold
                    kern_end = min(kern.shape[1], kern_end+stride)

                    if new_mse/tot_mse < trunc_thresh:
                        break

                if debug_plot:
                    fig, axs = plt.subplots(2,1,sharex=True)
                    axs[0].plot(kern.T)
                    axs[0].axvline(centroid_idx, color='k')
                    # axs[0].axvline(d_idx, color='r')
                    axs[1].plot(resid_diff)
                    axs[1].axhline(trunc_thresh)

                if shift + kern_start < trunc_min:
                    trunc_min = shift + kern_start

                if shift + kern_end > trunc_max:
                    trunc_max = shift + kern_end

                # kern[:, :kern_start] = np.nan
                # kern[:, kern_end:] = np.nan

            shifts[i] = shift

        dst_start = shift
        dst_end = dst_start + L

        # src_start = 0
        # src_end = L

        # if dst_start < 0:
        #     src_start = -dst_start
        #     dst_start = 0

        # if dst_end > out_len:
        #     src_end -= dst_end - out_len
        #     dst_end = out_len

        stacked[i, :, dst_start:dst_end] = kern

    if not truncate:
        # Crop away padding beyond the last valid sample
        valid = np.any(~np.isnan(stacked), axis=(0,1))
        trunc_min = max(0, np.argmax(valid))
        trunc_max = min(out_len, np.argmax(valid[::-1]))

    if extrapolate_edges:
        for i in range(n_kernels):
            for j in range(n_regions):
                stacked[i,j,0] = 0
                stacked[i,j,-1] = 0
                stacked[i,j,:] = fill_signal_nans(stacked[i,j,:])[0]

    stacked = stacked[:,:,trunc_min:trunc_max]

    # replace any remaining nans small eps
    stacked[np.isnan(stacked)] = eps

    if smooth:
        stacked = gaussian_filter1d(stacked, 1, axis=-1)

    return stacked, {'shifts': shifts, 'trunc_min': trunc_min, 'trunc_max': trunc_max, 'center_idx': center_idx-trunc_min}


# %% Kernel Regression Methods

def kernel_regression(signal_dict, ts_dict, event_ts_dict, kernel_edges, dt, n_perms=100, crossval=False, compute_param_stats=True):

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

    # get masks for timepoints that have and don't have predictors
    all_event_mask = predictor_mat.getnnz(axis=1) > 0
    non_event_mask = ~all_event_mask
    all_event_signals = stacked_signals[all_event_mask]
    non_event_signals = stacked_signals[non_event_mask]

    # calculate signal metrics
    tot_T = len(stacked_signals)
    all_event_T = sum(all_event_mask)
    non_event_T = sum(non_event_mask)

    all_event_mean = np.mean(all_event_signals)
    non_event_mean = np.mean(non_event_signals)
    full_mean = np.mean(stacked_signals)

    all_event_tot_var = np.sum((all_event_signals - all_event_mean)**2)
    non_event_tot_var = np.sum((non_event_signals - non_event_mean)**2)
    full_tot_var = np.sum((stacked_signals - full_mean)**2)

    non_event_std = np.sqrt(non_event_tot_var/non_event_T)

    var_ratio = (all_event_tot_var/all_event_T)/(non_event_tot_var/non_event_T)
    mean_shift = (all_event_mean - non_event_mean)/non_event_std

    # Perform full regression
    full_model = LinearRegression()
    full_model.fit(predictor_mat, stacked_signals)
    betas = full_model.coef_.flatten()

    pred_signals = full_model.predict(predictor_mat)

    # compute full model stats
    full_resid = stacked_signals - pred_signals

    masked_RSS = np.sum(full_resid[all_event_mask]**2)
    full_RSS = np.sum(full_resid**2)

    mask_R2 = 1 - masked_RSS / all_event_tot_var
    full_R2 = 1 - full_RSS / full_tot_var

    # perform leave-one-out cross validation
    cv_results = {'masked_R2': [], 'full_R2': []}
    cv_betas = []

    if crossval:
        for sess_id in sess_ids:
            cv_test_rows = sess_rows[sess_id]
            if len(cv_test_rows) == 0:
                continue

            cv_train_rows = np.concatenate([sess_rows[s] for s in sess_ids if s != sess_id])
            cv_row_mask_test = all_event_mask[cv_test_rows]

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
        'R2_masked': mask_R2,
        'RSS_masked': masked_RSS,
        'R2_full': full_R2,
        'RSS_full': full_RSS,
        'var_events': all_event_tot_var/all_event_T,
        'mean_events': all_event_mean,
        'var_non_events': non_event_tot_var/non_event_T,
        'mean_non_events': non_event_mean,
        'var_full': full_tot_var/tot_T,
        'mean_full': full_mean,
        'var_ratio' : var_ratio,
        'mean_shift' : mean_shift,
        'second_moment': var_ratio + mean_shift**2,
        'n_preds': P,
        'T_full': tot_T,
        'T_events': all_event_T,
        'T_non_events': non_event_T,
        'cv_results': cv_results
    }

    kernel_stats = {}
    event_stats = {}

    # compute stats on coefficients using Newey-West covariance
    if compute_param_stats:
        df = all_event_T-P
        autocorr_k = _compute_resid_autocorr_lag(full_resid, sess_rows)
        max_lag = min(max_k, autocorr_k)
        cov_beta = _hac_covariance(predictor_mat, full_resid, sess_rows, max_lag)
        cov_beta *= all_event_T/df
        se_betas = np.sqrt(np.diag(cov_beta))

        t_stats = betas / se_betas
        p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df))

        ci_alpha = 0.05
        c = t.ppf(1-ci_alpha/2, df)
        cis_low = betas - c * se_betas
        cis_high = betas + c * se_betas
    else:
        cov_beta = np.full((P,P), np.nan)
        se_betas = np.full(P, np.nan)
        t_stats = np.full(P, np.nan)
        p_vals = np.full(P, np.nan)
        cis_low = np.full(P, np.nan)
        cis_high = np.full(P, np.nan)

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

        red_masked_RSS = np.sum((all_event_signals - red_pred_signals[all_event_mask])**2)
        red_full_RSS = np.sum((stacked_signals - red_pred_signals)**2)

        # also compute only during event kernel
        event_signals = stacked_signals[event_row_sel]
        event_mean = np.mean(event_signals)
        red_event_RSS = np.sum((event_signals - red_pred_signals[event_row_sel])**2)
        full_event_RSS = np.sum((event_signals - pred_signals[event_row_sel])**2)
        event_tot_var = np.sum((event_signals - event_mean)**2)

        # ΔR² (fraction of total variance explained by event)- effect size
        delta_R2_masked = (red_masked_RSS - masked_RSS) / all_event_tot_var
        delta_R2_full = (red_full_RSS - full_RSS) / full_tot_var

        event_R2_full = 1 - full_event_RSS / event_tot_var
        event_R2_reduced = 1 - red_event_RSS / event_tot_var
        delta_R2_event = event_R2_full - event_R2_reduced

        # compute overall variance/power ratios associated with this event
        event_T = sum(event_row_sel)
        event_var_ratio = (event_tot_var/event_T)/(non_event_tot_var/non_event_T)
        event_mean_shift = (event_mean - non_event_mean)/non_event_std

        # Wald Statistic
        beta_e = betas[event_cols]
        cov_e = cov_beta[np.ix_(event_cols, event_cols)]

        if compute_param_stats:
            W = beta_e.T @ np.linalg.inv(cov_e) @ beta_e
            p_event = 1 - chi2.cdf(W, len(event_cols))
        else:
            W = np.nan
            p_event = np.nan

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
            perm_masked_RSS = np.sum((all_event_signals - perm_pred_signals[all_event_mask])**2)
            perm_full_RSS = np.sum((stacked_signals - perm_pred_signals)**2)

            perm_delta_R2_masked = (red_masked_RSS - perm_masked_RSS) / all_event_tot_var
            perm_delta_R2_full = (red_full_RSS - perm_full_RSS) / full_tot_var

            perm_results['delta_R2_masked'].append(perm_delta_R2_masked)
            perm_results['delta_R2_full'].append(perm_delta_R2_full)

        event_stats[e] = {
            'delta_R2_masked': delta_R2_masked,
            'RSS_reduced_masked': red_masked_RSS,
            'delta_R2_full': delta_R2_full,
            'RSS_reduced_full': red_full_RSS,
            'delta_R2_event': delta_R2_event,
            'R2_event_full': event_R2_full,
            'R2_event_reduced': event_R2_reduced,
            'RSS_event_reduced': red_event_RSS,
            'RSS_event_full': full_event_RSS,
            'var_event': event_tot_var/event_T,
            'mean_event': event_mean,
            'var_ratio' : event_var_ratio,
            'mean_shift' : event_mean_shift,
            'second_moment': event_var_ratio + event_mean_shift**2,
            'T_event': event_T,
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

        cols = event_col_idxs[e]
        delays = kernel_delays[e]

        P += len(cols)

        if len(event_idxs) == 0:
            continue

        # get rows and columns of predictor matrix that will have values of 1
        delay_idxs = event_idxs[:, None] + delays[None, :]
        row_idxs = delay_idxs.ravel()
        col_idxs = np.tile(cols, len(event_idxs))

        valid = (row_idxs >= 0) & (row_idxs < T)

        pred_rows.append(row_idxs[valid])
        pred_cols.append(col_idxs[valid])

    if not pred_rows:
        return sp.csr_matrix((T, P), dtype=np.float32)

    pred_rows = np.concatenate(pred_rows)
    pred_cols = np.concatenate(pred_cols)

    data = np.ones(len(pred_rows), dtype=np.float32)

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

# %% Event-aligned regression

def regress_over_time(signals, predictors, group_col='subjid', add_intercept=False):
    """
    Fit a regression independently at each timepoint in the event-aligned signals.
    If there is one group, fit OLS, if multiple groups fit a weighted least squares with equal weights to each subject

    Parameters
    ----------
    signals : stacked signal matrix (n_trials, T)
    predictors : dataframe of predictors for each trial
    group_col : The random effects group column
    add_intercept : Whether to add an intercept to the regression

    Returns
    -------
    a dictionary with the following elements:
        params: the best fit parameter values over time for each predictor
        se: the standard errors for the parameters
        ci_lower: lower bound of the parameters 95% confidence intervals
        ci_upper: upper bound of the parameters 95% confidence intervals
        p_vals: p values of the parameters
        t_vals: t values of the parameters
        n_obs: number of valid observations at each timepoint
        n_groups: number of fit groups
        model_type: the model type, either OLS or WLS

    """

    signals = np.asarray(signals)

    if signals.ndim != 2:
        raise ValueError('signals must have shape (n_trials, n_timepoints)')

    if len(predictors) != signals.shape[0]:
        raise ValueError('predictors must have one row for every trial in signals')

    if group_col not in predictors.columns:
        raise ValueError('{} not found in predictors'.format(group_col))

    predictor_cols = [col for col in predictors.columns if col != group_col]

    if not predictor_cols:
        raise ValueError('No predictor columns found')

    n_trials, T = signals.shape
    n_groups = predictors[group_col].nunique()

    if n_groups == 1:
        model_type = 'OLS'
    else:
        model_type = 'WLS'

    # Quote column names so names containing spaces/special characters work.
    def q(name):
        return f"Q('{name}')"

    if add_intercept:
        fixed_formula = " + ".join(q(col) for col in predictor_cols)
        param_names = ['Intercept'] + predictor_cols
    else:
        fixed_formula = "0 + " + " + ".join(q(col) for col in predictor_cols)
        param_names = predictor_cols

    results = {
        'params': {name: np.full(T, np.nan) for name in param_names},
        'se': {name: np.full(T, np.nan) for name in param_names},
        'ci_lower': {name: np.full(T, np.nan) for name in param_names},
        'ci_upper': {name: np.full(T, np.nan) for name in param_names},
        'p_vals': {name: np.full(T, np.nan) for name in param_names},
        't_vals': {name: np.full(T, np.nan) for name in param_names},
        'R2': np.full(T, np.nan),
        'n_obs': np.zeros(T, dtype=int),
        'n_groups': n_groups,
        'model_type': model_type,
    }

    for i in range(T):

        df = predictors.copy()
        df['_signal'] = signals[:, i]

        df = df.dropna()

        if df.empty:
            continue

        try:
            if model_type == 'OLS':
                fit = sm.OLS.from_formula(f"_signal ~ {fixed_formula}", data=df).fit()
            else:
                weights = 1 / df.groupby(group_col)[group_col].transform('size')
                fit = sm.WLS.from_formula(formula=f"_signal ~ {fixed_formula}", data=df, weights=weights).fit(
                    cov_type='cluster', cov_kwds={'groups': df[group_col]})

            results['n_obs'][i] = len(df)
            results['R2'][i] = fit.rsquared

            conf_int = fit.conf_int(alpha=0.05)

            for name in param_names:
                qname = q(name)
                results['params'][name][i] = fit.params[qname]
                results['se'][name][i] = fit.bse[qname]
                results['p_vals'][name][i] = fit.pvalues[qname]
                results['t_vals'][name][i] = fit.tvalues[qname]
                results['ci_lower'][name][i] = conf_int.loc[qname, 0]
                results['ci_upper'][name][i] = conf_int.loc[qname, 1]

        except Exception as e:
            print(f"An error occurred during the regression at timestep {i}: {e}\ncontinuing...\n")
            continue

    return results


# def regress_over_time(signals, predictors, group_col='subjid', random_intercept=True, random_slopes=None, add_intercept=False):
#     """
#     Fit a regression independently at each timepoint in the event-aligned signals.
#     If there is one group, fit OLS, if multiple groups fit a mixed effects model

#     Parameters
#     ----------
#     signals : stacked signal matrix (n_trials, T)
#     predictors : dataframe of predictors for each trial
#     group_col : The random effects group column
#     random_intercept : (Optional) Whether to have a random intercept
#     random_slopes : (Optional) The predictors that should get random slopes, or group-level random effects on the predictor coefficient.
#                     None (default) is all predictors, [] is no predictors wherein the only random effect could be the intercept
#     add_intercept : Whether to add an intercept to the regression

#     Returns
#     -------
#     a dictionary with the following elements:
#         params: the best fit parameter values over time for each predictor
#         se: the standard errors for the parameters
#         ci_lower: lower bound of the parameters 95% confidence intervals
#         ci_upper: upper bound of the parameters 95% confidence intervals
#         p_vals: p values of the parameters
#         t_vals: t values of the parameters
#         converged: whether the regression converged
#         n_obs: number of valid observations at each timepoint
#         n_groups: number of fit groups
#         model_type: the model type, either OLS or MixedLM

#     """

#     signals = np.asarray(signals)

#     if signals.ndim != 2:
#         raise ValueError('signals must have shape (n_trials, n_timepoints)')

#     if len(predictors) != signals.shape[0]:
#         raise ValueError('predictors must have one row for every trial in signals')

#     if group_col not in predictors.columns:
#         raise ValueError('{} not found in predictors'.format(group_col))

#     predictor_cols = [col for col in predictors.columns if col != group_col]

#     if not predictor_cols:
#         raise ValueError('No predictor columns found')

#     if random_slopes is None:
#         random_slopes = predictor_cols.copy()

#     invalid_slopes = set(random_slopes) - set(predictor_cols)
#     if invalid_slopes:
#         raise ValueError('Random slope predictors do not exist in predictor table: {}'.format(', '.join(invalid_slopes)))

#     n_trials, T = signals.shape
#     n_groups = predictors[group_col].nunique()

#     if n_groups == 1:
#         model_type = 'OLS'
#     else:
#         model_type = 'MixedLM'

#     # Quote column names so names containing spaces/special characters work.
#     def q(name):
#         return f"Q('{name}')"

#     if add_intercept:
#         fixed_formula = " + ".join(q(col) for col in predictor_cols)
#         param_names = ['Intercept'] + predictor_cols
#     else:
#         fixed_formula = "0 + " + " + ".join(q(col) for col in predictor_cols)
#         param_names = predictor_cols

#     random_formula = "1" if random_intercept else "0"

#     if len(random_slopes) > 0:
#         random_formula += " + " + " + ".join(q(col) for col in random_slopes)

#     results = {
#         'params': {name: np.full(T, np.nan) for name in param_names},
#         'se': {name: np.full(T, np.nan) for name in param_names},
#         'ci_lower': {name: np.full(T, np.nan) for name in param_names},
#         'ci_upper': {name: np.full(T, np.nan) for name in param_names},
#         'p_vals': {name: np.full(T, np.nan) for name in param_names},
#         't_vals': {name: np.full(T, np.nan) for name in param_names},
#         'converged': np.zeros(T, dtype=bool),
#         'n_obs': np.zeros(T, dtype=int),
#         'n_groups': n_groups,
#         'model_type': model_type,
#     }

#     for i in range(T):

#         df = predictors.copy()
#         df['_signal'] = signals[:, i]

#         df = df.dropna()

#         if df.empty:
#             continue

#         try:
#             if model_type == 'OLS':
#                 model = sm.OLS.from_formula(f"_signal ~ {fixed_formula}", data=df)
#             else:
#                 model = sm.MixedLM.from_formula(formula=f"_signal ~ {fixed_formula}", data=df,
#                                                 groups=df[group_col], re_formula=random_formula)

#             fit = model.fit(method='lbfgs')

#             results['converged'][i] = True if model_type == 'OLS' else fit.converged
#             results['n_obs'][i] = len(df)

#             conf_int = fit.conf_int(alpha=0.05)

#             for name in param_names:
#                 qname = q(name)
#                 results['params'][name][i] = fit.params[qname]
#                 results['se'][name][i] = fit.bse[qname]
#                 results['p_vals'][name][i] = fit.pvalues[qname]
#                 results['t_vals'][name][i] = fit.tvalues[qname]
#                 results['ci_lower'][name][i] = conf_int.loc[qname, 0]
#                 results['ci_upper'][name][i] = conf_int.loc[qname, 1]

#         except Exception as e:
#             print(f"An error occurred during the regression at timestep {i}: {e}\ncontinuing...\n")
#             continue

#     return results
