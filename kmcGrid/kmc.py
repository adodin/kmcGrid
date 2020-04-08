""" Simple Rejection Free KMC Engine for grids
Written by: Amro Dodin
"""

import numpy as np


def step(s0, lattice):
    neighbors = lattice.get_neighbors(s0)
    ks = lattice.get_rates(s0)
    k_tot = np.sum(ks)
    k_cum = np.cumsum(ks)
    assert len(k_cum) == len(neighbors)
    assert k_cum[-1] == k_tot
    u = np.random.random()
    s_next = neighbors[np.argmax( k_tot * u <= k_cum)]
    u_time = np.random.random()
    dt = -np.log(u_time)/k_tot
    return s_next, dt


def run(s0, lattice, tau):
    states = []
    times = []
    curr = s0
    t = 0.
    states.append(curr)
    times.append(t)
    while t < tau:
        curr, dt = step(curr, lattice)
        t += dt
        if t > tau:
            t = tau
        states.append(curr)
        times.append(t)
    return states, np.array(times)


if __name__ == '__main__':
    from kmcGrid import plot
    from kmcGrid.lattice import SCLattice

    T = 300
    lattice = SCLattice((100, 100), 0., 1.0, T)
    num_traj = 1000
    t_max = 30
    n_plot = 40
    s0 = (50, 50)
    s_traj = []
    t_traj = []
    target_times = np.linspace(0, t_max, n_plot)
    for i in range(num_traj):
        print(str(i + 1) + '/' + str(num_traj))
        lattice.shuffle()
        states, times = run(s0, lattice, t_max)
        s_traj.append(states)
        t_traj.append(times)
    plt = plot.plot_mean_msd(s0, s_traj, t_traj, target_times)
