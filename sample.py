"""
Functions for Sampling Trajectories at specific target times
Required since KMC produces trajectories with random time steps
Written By: Amro Dodin
"""

import numpy as np
from observables import calculate_msd, calculate_rmsd


def sample_trajectories(state_trajectories, time_trajectories, target_times, lattice):
    """ Samples Trajectories contained in lists state_trajectories, time_trajectories in lattice at target_times """
    counts = []
    for t in target_times:
        count = np.zeros(lattice.size)
        for ss, ts in zip(state_trajectories, time_trajectories):
            index = np.argmax(t <= ts)
            count[ss[index]] += 1
        counts.append(count)
    return counts


def sample_observable(state_trajectories, time_trajectories, target_times, calculate_observable, observable_kwargs={}):
    """
    Samples Observables along state_trajectories, time_trajectories at target_times
    calculate_observable is a function that calculates the desired observable and observable_kwargs are its kwargs
    """
    # Calculate MSD along each trajectory
    observable_trajectories = []
    for s_traj in state_trajectories:
        observable_trajectories.append(calculate_observable(state_trajectory=s_traj, **observable_kwargs))

    # Find MSD distribution at each target time
    observable_samples = []
    for t in target_times:
        timed_samples = []
        for ms, ts in zip(observable_trajectories, time_trajectories):
            index = np.argmax(t <= ts)
            timed_samples.append(ms[index])
        observable_samples.append(timed_samples)
    return observable_samples


# Sample Functions to show how to use sample_observable
def sample_msd(s0, state_trajectories, time_trajectories, target_times):
    return sample_observable(state_trajectories, time_trajectories, target_times, calculate_msd, {'s0': s0})


def sample_rmsd(s0, state_trajectories, time_trajectories, target_times):
    return sample_observable(state_trajectories, time_trajectories, target_times, calculate_rmsd, {'s0': s0})