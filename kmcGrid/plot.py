"""
Static Plotting for print-friendly visualization of KMC simulation data
Written By: Amro Dodin
"""

import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from kmcGrid import sample as smp
from kmcGrid.observables import window_slope
from _plot_defaults import fig_spec, font_spec, axis_rect

# Set Default Font Parameters
mpl.rc('font', **font_spec)


def plot_2d_lattice(lattice, line_increment=None, axis_scale=(1, 1), axis_units='nm',
                    figkwargs=fig_spec, axis_rect=axis_rect,
                    linekwargs={'color': 'w', 'linewidth': 3.}, imkwargs={}, fname=None):
    assert lattice.dim == 2
    sx, sy = lattice.size
    sx = sx * axis_scale[0]
    sy = sy * axis_scale[1]

    fig = plt.figure(**figkwargs)
    ax = fig.add_axes(axis_rect)
    im = ax.imshow(lattice.energy, origin='lower', extent=[0, sx, 0, sy], **imkwargs)
    ax.set_xlabel('x (' + axis_units + ')')
    ax.set_xlim(0., sx)
    ax.set_ylabel('y (' + axis_units + ')')
    ax.set_ylim(0., sy)

    if line_increment is None:
        if fname is not None:
            fig.savefig(fname)
        plt.show()
        return ax

    if type(line_increment) is tuple:
        assert len(line_increment) == 2
        ix, iy = line_increment
    else:
        ix = line_increment
        iy = line_increment
    x_lines = np.arange(ix, sx, ix)
    y_lines = np.arange(iy, sy, iy)

    for x in x_lines:
        ax.plot([x, x], [0, sy], **linekwargs)
    for y in y_lines:
        ax.plot([0, sx], [y, y], **linekwargs)

    if fname is not None:
        fig.savefig(fname)
    plt.show()
    return ax


def plot_rmsd_trajectories(s0, state_trajectories, time_trajectories, target_times, num_plotted=3,
                           figkwargs=fig_spec, axis_rect=axis_rect,
                           linekwargs={'linewidth': 3.}, fname=None):
    timed_samples = smp.sample_msd(s0, state_trajectories, time_trajectories, target_times)
    fig = plt.figure(**figkwargs)
    ax = fig.add_axes(axis_rect)
    t_space = int(len(target_times)/num_plotted)
    rmsd_max = 1.5*np.std(timed_samples[-1])
    rmsd_hists = []
    for t in range(0, len(timed_samples), t_space):
        hist, bins = np.histogram(timed_samples[t], bins=10, range=(0, rmsd_max))
        rmsd_hists.append(hist)
    bin_mids = []
    for i in range(len(bins)-1):
        bin_mids.append(0.5*(bins[i] + bins[i+1]))
    for h in rmsd_hists:
        ax.plot(bin_mids, h, **linekwargs)
    if fname is not None:
        fig.savefig(fname)
    plt.show()
    return ax


def plot_mean_msd(s0, state_trajectories, time_trajectories, target_times, ave_window=None,
                  figkwargs=fig_spec, axis_rect=axis_rect, fname=None, slope_fname=None, target_axes=None):
    timed_samples = smp.sample_msd(s0, state_trajectories, time_trajectories, target_times)
    mean_samples = np.mean(timed_samples, axis=1)

    if target_axes is None:
        ax1 = None
        ax2 = None
    elif type(target_axes) is tuple:
        ax1 = target_axes[0]
        ax2 = target_axes[1]

    if ax1 is None:
        fig1 = plt.figure(**figkwargs)
        ax1 = fig1.add_axes(axis_rect)
    else:
        fig1 = ax1.figure
    ax1.plot(target_times, mean_samples)

    if fname is not None:
        fig1.savefig(fname)

    if ave_window is None:
        plt.show()
        return ax1

    if ax2 is None:
        fig2 = plt.figure(**figkwargs)
        ax2 = fig2.add_axes(axis_rect)
    else:
        fig2 = ax2.figure
    slopes = window_slope(target_times, mean_samples, ave_window)
    ax2.plot(target_times[0:-ave_window], slopes)

    if slope_fname is not None:
        fig2.savefig(slope_fname)

    plt.show()
    return ax1, ax2
