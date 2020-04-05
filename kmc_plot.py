import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def histogram_trajectories(state_trajectories, time_trajectories, target_times, lattice):
    counts = []
    for t in target_times:
        count = np.zeros(lattice.size)
        for ss, ts in zip(state_trajectories, time_trajectories):
            index = np.argmax(t <= ts)
            count[ss[index]] += 1
        counts.append(count)
    return counts
