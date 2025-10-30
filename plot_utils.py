# -*- coding: utf-8 -*-
"""
Set of functions to create common plots

@author: tanner stevenson
"""

import matplotlib.pyplot as plt
import numpy as np
import pyutils.utils as utils
import seaborn as sb
import pandas as pd

def plot_shaded_error(x, y, y_err=None, ax=None, **kwargs):

    if ax is None:
        ax = get_axes()

    if not y_err is None:
        y_err = np.array(y_err)
        if y_err.ndim == 1:
            y_err_up = y_err
            y_err_low = y_err
        elif y_err.ndim == 2:
            # make sure only 2 rows
            si = np.argwhere(np.array(y_err.shape) == 2)
            if len(si) > 0:
                if si[0,0] == 0:
                    y_err_low = y_err[0,:]
                    y_err_up = y_err[1,:]
                else:
                    y_err_low = y_err[:,0]
                    y_err_up = y_err[:,1]
            else:
                raise ValueError('Invalid error format. Must either be 1xN or 2xN where N is number of data points in x & y')
        else:
            raise ValueError('Invalid error format. Must either be 1xN or 2xN where N is number of data points in x & y')

        line = ax.plot(x, y, zorder=2, **kwargs)
        
        # make sure the fill is the same color as the signal line
        # don't include the error in the legend
        tmp_kwargs = kwargs.copy()
        if 'color' in tmp_kwargs:
            _ = tmp_kwargs.pop('color')

        # make sure the fill doesn't show up in the legend
        tmp_kwargs['label'] = '_'

        upper = y + y_err_up
        lower = y - y_err_low
        
        ax.fill_between(x, upper, lower, alpha=0.2, color=line[0].get_color(), zorder=1, **tmp_kwargs)

    else:
        # just plot signal
        line = ax.plot(x, y, **kwargs)

    return line[0], ax

def plot_psth(time, signal, error=None, ax=None, plot_x0=True, **kwargs):

    if ax is None:
        ax = get_axes()

    # plot line at x=0
    if plot_x0:
        plot_x0line(ax=ax)

    return plot_shaded_error(time, signal, y_err=error, ax=ax, **kwargs)


def plot_psth_dict(psth_dict, ax=None, plot_x0=True, **kwargs):

    return plot_psth(psth_dict['time'], psth_dict['signal_avg'], psth_dict['signal_se'], ax, plot_x0, **kwargs)


def plot_raster(spike_times, ax=None, plot_x0=True):

    # if no axes are passed in, create an axis
    if ax is None:
        ax = get_axes()

    # plot line at x=0
    if plot_x0:
        plot_x0line(ax=ax)

    # determine if there are multiple trials to stack or not
    if len(spike_times) > 0 and utils.is_scalar(spike_times[0]):
        spike_times = [spike_times]

    # iterate through trials and plot the spikes stacked one on top of the other
    for i, trial_spike_times in enumerate(spike_times):
        y_min = [i] * len(trial_spike_times)
        y_max = [i+1] * len(trial_spike_times)
        lines = ax.vlines(trial_spike_times, y_min, y_max)

    # this doesn't work correctly... not sure why
    #ax.eventplot(spike_times, **kwargs)

    return lines, ax


def plot_stacked_bar(values_list, value_labels=None, x_labels=None, orientation='h', ax=None, err=None, x_label_rot=None):

    if orientation != 'h' and orientation != 'v':
        raise ValueError('The orientation can only be \'h\' for horizontal groups or \'v\' for vertically stacked bars')

    if ax is None:
        ax = get_axes()

    # get number of data categories to plot
    n_cats = len(values_list)

    # process inputs
    if not all([len(values_list[0]) == len(values_list[i]) for i in range(1, n_cats)]):
        raise ValueError('The number of values to be plotted in each series are not equal')

    if value_labels is None:
        value_labels = ['_' for i in range(n_cats)]
    else:
        if len(value_labels) != n_cats:
            raise ValueError('The number of value labels ({0}) does not match the number of plot categories ({1})'.format(
                len(value_labels), n_cats))

    if x_labels is None:
        x_labels = [i for i in range(values_list[0])]
    else:
        if len(x_labels) != len(values_list[0]):
            raise ValueError('The number of x labels ({0}) does not match the number of elements being plotted ({1})'.format(
                len(x_labels), len(values_list[0])))

    if err is None:
        err = np.full(n_cats, None)
    else:
        if len(err) != n_cats:
            raise ValueError('The number of y errors ({0}) does not match the number of plot categories ({1})'.format(
                len(err), n_cats))


    x = np.arange(len(x_labels))

    # group bars horizontally
    if orientation == 'h':
        # determine width of the bars
        width = 1/(n_cats+1)

        for i in range(n_cats):
            ax.bar(x + width*i, values_list[i], width=width, label=value_labels[i], yerr=err[i])

        ax.set_xticks(x + width*(n_cats-1)/2, x_labels)
        
        ax.set_xlim(-width, x[-1]+width*n_cats)
    # else stack bars vertically
    else:
        y_start = np.zeros_like(x)
        width = 0.75

        for i in range(n_cats):
            vals = np.array(values_list[i])
            vals[np.isnan(vals)] = 0
            ax.bar(x, vals, width=width, label=value_labels[i], bottom=y_start, yerr=err[i])
            y_start = y_start + vals

        ax.set_xticks(x, x_labels)
        ax.set_xlim(-width, x[-1]+width)
        
    if not x_label_rot is None:
        ax.tick_params(axis='x', labelrotation=x_label_rot)

    if any([not l.startswith('_') for l in value_labels]):
        ax.legend()


def plot_grouped_error_bar(values_list, error_list, value_labels=None, x_labels=None, ax=None):

    if ax is None:
        ax = get_axes()

    # get number of data categories to plot
    n_cats = len(values_list)

    # process inputs
    if not all([len(values_list[0]) == len(values_list[i]) for i in range(1, n_cats)]):
        raise ValueError('The number of values to be plotted in each series are not equal')

    if not all([len(error_list[0]) == len(error_list[i]) for i in range(1, n_cats)]):
        raise ValueError('The number of errors to be plotted in each series are not equal')

    if not len(values_list[0]) == len(error_list[0]):
        raise ValueError('The number of values and errors to be plotted in each series are not equal')

    if value_labels is None:
        value_labels = ['' for i in range(n_cats)]
    else:
        if len(value_labels) != n_cats:
            raise ValueError('The number of value labels ({0}) does not match the number of plot categories ({1})'.format(
                len(value_labels), n_cats))

    if x_labels is None:
        x_labels = [i for i in range(values_list[0])]
    else:
        if len(x_labels) != len(values_list[0]):
            raise ValueError('The number of x labels ({0}) does not match the number of elements being plotted ({1})'.format(
                len(x_labels), len(values_list[0])))

    x = np.arange(len(x_labels))

    # determine offset of each grouped point
    x_offset = 1/(n_cats+1)

    for i in range(n_cats):
        ax.errorbar(x + x_offset*i, values_list[i], yerr=error_list[i], label=value_labels[i], fmt='o', capsize=4)

    ax.set_xticks(x + x_offset*(n_cats-1)/2, x_labels)

    ax.legend()


def plot_value_matrix(values, ax=None, x_rot=0, y_rot=0, fmt='.3f', cbar=True, **kwargs):
    ''' Plot a heatmap of the given values '''
    if ax is None:
        ax = get_axes()

    # make sure any pandas tables have numeric values
    if type(values) is pd.DataFrame:
        values = values.infer_objects()

    # plot the heatmap
    hm = sb.heatmap(values, annot=(fmt != None), fmt=fmt, ax=ax, linewidth=1, vmin=0, vmax=1, cbar=cbar, **kwargs)
    # move x axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # adjust label orientation
    ax.xaxis.set_tick_params(rotation=x_rot)
    ax.yaxis.set_tick_params(rotation=y_rot)
    # remove tick marks
    ax.tick_params(axis='both', which='both', length=0)

    return hm, ax

def plot_stacked_heatmap_avg(data_mat, t, heatmap_ax=None, avg_ax=None, x_label='', y_label='', title='',
                             show_cbar=True, error_type='std', cmap=None, vmax=None, vmin=None, **kwargs):
    ''' Plots a heatmap of the data matrix along with an average signal trace '''

    # first plot the average activity to get the appropriate x axis labels for the heatmap
    match error_type:
        case 'std':
            err = np.nanstd(data_mat, axis=0)
        case 'se':
            err = utils.stderr(data_mat, axis=0)
        case 'none' | None:
            err = None
        case _:
            raise ValueError('Incorrect error type specified: {}. Possible error types: std, se, none.'.format(error_type))

    if heatmap_ax is None or avg_ax is None:
        _, axs = plt.subplots(2, 1, layout='constrained', height_ratios=(1.5,1))
        heatmap_ax = axs[0]
        avg_ax = axs[1]

    line = plot_psth(np.nanmean(data_mat, axis=0), t, err, avg_ax, **kwargs)
    avg_ax.set_xlabel(x_label)
    avg_ax.set_ylabel(y_label)

    # get heatmap tick locations
    xticks = np.array(avg_ax.get_xticks())
    # curate to limits of plot
    xticks = xticks[(xticks >= t[0]) & (xticks <= t[-1])]
    tick_start_idx = np.argmin(abs(t-xticks[0]))
    tick_end_idx = np.argmin(abs(t-xticks[-1]))
    tick_labels = [str(x) for x in xticks]
    tick_idxs = np.linspace(tick_start_idx, tick_end_idx, len(tick_labels))

    im = heatmap_ax.imshow(data_mat, interpolation=None, aspect='auto', cmap=cmap,
                           vmax=vmax, vmin=vmin, **kwargs)
    zero_idx = np.argmin(abs(t))
    plot_x0line(x0=zero_idx, ax=heatmap_ax)
    heatmap_ax.set_xticks(tick_idxs, labels=tick_labels)
    heatmap_ax.set_ylabel('Trials')
    heatmap_ax.set_title(title)

    if show_cbar:
        fig = heatmap_ax.get_figure()
        fig.colorbar(im, ax=heatmap_ax, label=y_label)

    return im, line

## General Helper Methods
def get_axes():
    # if there are no figures, create one first
    if len(plt.get_fignums()) == 0:
        _, ax = plt.subplots(1, 1)
    else:
        ax = plt.gca()

    return ax

def plot_x0line(x0=None, ax=None, **kwargs):
    if ax is None:
        ax = get_axes()

    return plot_dashlines(vals=x0, dir='v', ax=ax, **kwargs)

def plot_unity_line(ax, **kwargs):
    ''' Plots the unity line y=x '''
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Define the range for y = x (taking min/max to avoid clipping)
    line_min = min(xmin, ymin)
    line_max = max(xmax, ymax)
    
    default_vals = dict(dashes=[4,4], c='grey', lw=1)
    for k,v in default_vals.items():
        if not k in kwargs:
            kwargs[k] = v
    
    ax.plot([line_min, line_max], [line_min, line_max], **kwargs)

def plot_dashlines(vals=None, dir='v', ax=None, **kwargs):
    if ax is None:
        ax = get_axes()

    if vals is None:
        vals = [0]
    elif utils.is_scalar(vals):
        vals = [vals]

    default_vals = dict(dashes=[4,4], c='grey', lw=1)
    for k,v in default_vals.items():
        if not k in kwargs:
            kwargs[k] = v

    for val in vals:
        match dir:
            case 'v':
                ax.axvline(val, **kwargs)
            case 'h':
                ax.axhline(val, **kwargs)

    return ax


def show_axis_labels(ax, axis='y'):
    if axis == 'y':
        ax.yaxis.set_tick_params(which='both', labelleft=True)
    else:
        ax.xaxis.set_tick_params(which='both', labelleft=True)
        
        
def transform_axes_to_data(x, y, ax):
    return ax.transLimits.inverted().transform((x,y))

def transform_data_to_axes(x, y, ax):
    return ax.transLimits.transform((x,y))

# Methods to autoscale an axis to a sub-view from stack overflow: https://stackoverflow.com/questions/29461608/fixing-x-axis-scale-and-autoscale-y-axis
def autoscale(ax=None, axis='y', margin=0.1):
    '''Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    '''
    if ax is None:
        ax = plt.gca()
    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x,y = get_xy(artist)
        if axis == 'y':
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = low if low < newlow else newlow
        newhigh = high if high > newhigh else newhigh

    margin = margin*(newhigh - newlow)

    setlim(newlow-margin, newhigh+margin)

def calculate_new_limit(fixed, dependent, limit):
    '''Calculates the min/max of the dependent axis given
    a fixed axis with limits
    '''
    if len(fixed) > 2:
        mask = (fixed>limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            # This is a axhline in the autoscale direction
            low = np.inf
            high = -np.inf
    return low, high

def get_xy(artist):
    '''Gets the xy coordinates of a given artist
    '''
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y
