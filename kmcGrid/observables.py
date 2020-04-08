"""
Utility Functions for Calculating Observables in KMC Simulations
Written By: Amro Dodin
"""

import numpy as np
import scipy.stats as stats

def calculate_msd(s0, state_trajectory):
    """ Calculates MSD from initial state s0 along trajectory s_traj """
    s_0 = np.array(s0)
    st = np.asarray(state_trajectory)
    msd = np.sum((st-s_0)**2, axis=-1)
    return msd


def calculate_rmsd(s0, state_trajectory):
    """ Calculates RMSD from initial state s0 along trajectory state_trajectory """
    return np.sqrt(calculate_msd(s0, state_trajectory))


def window_slope(x, y, window):
    """ Time-dependent slope of data (x, y) using fixed width window """
    assert len(x) == len(y)
    length = len(x)
    slopes = []
    for i in range(length-window):
        lim = min(length, i + window)
        m, c, r, p, s_err = stats.linregress(x[i:lim], y[i:lim])
        slopes.append(m)
    return slopes
