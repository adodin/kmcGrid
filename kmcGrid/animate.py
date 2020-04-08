"""
Animation Functions for Visualizing Time-Dependent KMC Simulations
Written By: Amro Dodin
"""

import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from kmcGrid import sample as smp
from kmcGrid.observables import calculate_rmsd


def animate_populations(state_trajectories, time_trajectories, target_times, lattice, cmap='viridis'):
    """ Generates an animation showing the time-dependent population on a 2D Lattice"""
    num_traj = len(state_trajectories)
    norm = mpl.colors.LogNorm(vmin=1/num_traj, vmax=1.0)

    counts = smp.sample_trajectories(state_trajectories, time_trajectories, target_times, lattice)
    counts = counts
    fig, ax = plt.subplots()
    im = ax.imshow(counts[0]/num_traj, cmap=cmap, norm=norm, origin='lower')
    fig.colorbar(im, ax=ax)

    def update(frame):
        im.set_array(counts[frame]/num_traj)
        return [im]

    ani = FuncAnimation(fig, update, frames=range(len(target_times)))

    plt.show()
    return ani


def animate_observable_histograms(state_trajectories, time_trajectories, target_times,
                                  observable_function, observable_kwargs, n_bins=100, bin_max=None, ylim=None):
    # Calculate Timed MSD distribution
    timed_samples = smp.sample_observable(state_trajectories, time_trajectories, target_times,
                                          observable_function, observable_kwargs)

    # Find Max Width From largest MSD
    if bin_max is None:
        bin_max = np.max(timed_samples)

    # Construct Histogram Bins
    bins = np.linspace(0, bin_max, n_bins)

    n, bins = np.histogram(timed_samples[0], bins, density=True)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)

    # Builds a List of Vertices that need to be moved
    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * mpl.path.Path.LINETO
    codes[0::5] = mpl.path.Path.MOVETO
    codes[4::5] = mpl.path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    patch = None

    def animate(i):
        # simulate new data coming in
        data = timed_samples[i]
        n, bin = np.histogram(data, bins, density=True)
        top = bottom + n
        verts[1::5, 1] = top
        verts[2::5, 1] = top
        return [patch, ]

    fig, ax = plt.subplots()
    ax.set_xlim(0, bin_max)
    barpath = mpl.path.Path(verts, codes)
    patch = mpl.patches.PathPatch(
        barpath, facecolor='pink', edgecolor='red')
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    if ylim is None:
        ax.set_ylim(bottom.min(), top.max())
    else:
        ax.set_ylim(*ylim)

    ani = mpl.animation.FuncAnimation(fig, animate, len(target_times), repeat=False, blit=True)
    plt.show()
    return ani


def animate_rmsd_histogram(s0, state_trajectories, time_trajectories, target_times, nbins=100, bin_max=None, ylim=None):
    return animate_observable_histograms(state_trajectories, time_trajectories, target_times,
                                         calculate_rmsd, {'s0': s0}, nbins, bin_max, ylim)
