"""
Static Plotting for print-friendly visualization of KMC simulation data
Written By: Amro Dodin
"""

import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

from kmcGrid import sample as smp
from kmcGrid.observables import window_slope

# Set Default Font Parameters
font = {'family': 'serif', 'size': 30}
mpl.rc('font', **font)


def plot_2d_lattice(lattice, line_increment=None, axis_scale=(1, 1), axis_units='nm',
                    figkwargs={'figsize': [10, 10]}, axis_rect = [0.15, 0.15, 0.8, 0.8],
                    linekwargs={'color': 'w', 'linewidth': 3.},  imkwargs={}, fname=None):
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


def plot_mean_msd(s0, state_trajectories, time_trajectories, target_times, ave_window=None):
    timed_samples = smp.sample_msd(s0, state_trajectories, time_trajectories, target_times)
    mean_samples = np.mean(timed_samples, axis=1)
    plt.plot(target_times, mean_samples)
    if ave_window is not None:
        fig = plt.figure()
        slopes = window_slope(target_times, mean_samples, ave_window)
        plt.plot(target_times[0:-ave_window], slopes)
    plt.show()
