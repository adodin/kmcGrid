"""
Static Plotting for print-friendly visualization of KMC simulation data
Written By: Amro Dodin
"""

import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt

import sample as smp
from observables import window_slope

# Set Default Font Parameters
font = {'family': 'serif', 'size': 30}
mpl.rc('font', **font)


def plot_mean_msd(s0, state_trajectories, time_trajectories, target_times, ave_window=None):
    timed_samples = smp.sample_msd(s0, state_trajectories, time_trajectories, target_times)
    mean_samples = np.mean(timed_samples, axis=1)
    plt.plot(target_times, mean_samples)
    if ave_window is not None:
        fig = plt.figure()
        slopes = window_slope(target_times, mean_samples, ave_window)
        plt.plot(target_times[0:-ave_window], slopes)
    plt.show()
