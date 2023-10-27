# -*- coding: utf-8 -*-
"""
Set of functions to create common plots

@author: tanner stevenson
"""

import matplotlib.pyplot as plt
import numpy as np
import pyutils.utils as utils

def plot_shaded_error(x, y, y_err=None, ax=None, **kwargs):

    # if no axes are passed in, create an axis
    if ax is None:
        # if there are no figures, create one first
        if len(plt.get_fignums()) == 0:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plt.gca()

    if not y_err is None:
        # plot error first
        upper = y + y_err
        lower = y - y_err
        # don't include the error in the legend
        tmp_kwargs = kwargs.copy()
        tmp_kwargs['label'] = '_'

        fill = ax.fill_between(x, upper, lower, alpha=0.2, **tmp_kwargs)
        # make sure the fill is the same color as the signal line
        c = fill.get_facecolor()
        ax.plot(x, y, color=c[:, 0:3], **kwargs)
    else:
        # just plot signal
        ax.plot(x, y, **kwargs)

    return ax

def plot_psth(signal, time, error=None, ax=None, plot_x0=True, **kwargs):

    # if no axes are passed in, create an axis
    if ax is None:
        # if there are no figures, create one first
        if len(plt.get_fignums()) == 0:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plt.gca()

    # plot line at x=0
    if plot_x0:
        ax.axvline(dashes=[4, 4], c='k', lw=1)

    plot_shaded_error(time, signal, y_err=error, ax=ax, **kwargs)

    return ax


def plot_psth_dict(psth_dict, ax=None, plot_x0=True, **kwargs):

    return plot_psth(psth_dict['signal_avg'], psth_dict['time'], psth_dict['signal_se'], ax, plot_x0, **kwargs)


def plot_raster(spike_times, ax=None, plot_x0=True):

    # if no axes are passed in, create an axis
    if ax is None:
        # if there are no figures, create one first
        if len(plt.get_fignums()) == 0:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plt.gca()

    # plot line at x=0
    if plot_x0:
        ax.axvline(dashes=[4, 4], c='k', lw=1)

    # determine if there are multiple trials to stack or not
    if len(spike_times) > 0 and utils.is_scalar(spike_times[0]):
        spike_times = [spike_times]

    # iterate through trials and plot the spikes stacked one on top of the other
    for i, trial_spike_times in enumerate(spike_times):
        y_min = [i] * len(trial_spike_times)
        y_max = [i+1] * len(trial_spike_times)
        ax.vlines(trial_spike_times, y_min, y_max)

    # this doesn't work correctly... not sure why
    #ax.eventplot(spike_times, **kwargs)

    return ax


def plot_stacked_bar(values_list, value_labels=None, x_labels=None, orientation='h', ax=None):

    if orientation != 'h' and orientation != 'v':
        raise ValueError('The orientation can only be \'h\' for horizontal groups or \'v\' for vertically stacked bars')

    # if no axes are passed in, create an axis
    if ax is None:
        # if there are no figures, create one first
        if len(plt.get_fignums()) == 0:
            fig, ax = plt.subplots(1, 1)
        else:
            ax = plt.gca()

    # get number of data categories to plot
    n_cats = len(values_list)

    # process inputs
    if not all([len(values_list[0]) == len(values_list[i]) for i in range(1, n_cats)]):
        raise ValueError('The number of values to be plotted in each series are not equal')

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

    # group bars horizontally
    if orientation == 'h':
        # determine width of the bars
        width = 1/(n_cats+1)

        for i in range(n_cats):
            ax.bar(x + width*i, values_list[i], width=width, label=value_labels[i])

        ax.set_xticks(x + width*(n_cats-1)/2, x_labels)
    # else stack bars vertically
    else:
        y_start = np.zeros_like(x)

        for i in range(n_cats):
            ax.bar(x, values_list[i], width=0.75, label=value_labels[i], bottom=y_start)
            y_start = y_start + values_list[i]

        ax.set_xticks(x, x_labels)

    ax.legend()
