import numpy as np
import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def histogram_trajectories(state_trajectories, time_trajectories, target_times, lattice):
    counts = []
    for t in target_times:
        count = np.zeros(lattice.size)
        for ss, ts in zip(state_trajectories, time_trajectories):
            index = np.argmax(t <= ts)
            count[ss[index]] += 1
        counts.append(count)
    return counts


def imshow_animate(state_trajectories, time_trajectories, target_times, lattice):
    counts = histogram_trajectories(state_trajectories, time_trajectories, target_times, lattice)
    fig, ax = plt.subplots()
    im = ax.imshow(counts[0])

    def update(frame):
        im.set_array(counts[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames= range(len(target_times)))

    plt.show()
    return ani

