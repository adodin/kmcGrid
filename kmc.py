""" Simple Rejection Free KMC Engine for grids
Written by: Amro Dodin
"""

import numpy as np

k_B = 8.617333E-5


class SCLattice:
    """ Simple Cube Lattice with Nearest Neighbor Coupling """
    def __init__(self, size, std_energy, rate, temp):
        self.rate = rate
        self.size = size
        self.dim = len(size)
        self.energy = np.random.normal(0., std_energy, size)
        self.temp = temp
        self.__min = tuple(np.zeros(self.dim))

    def get_neighbors(self, state):
        mut_state = list(state)
        neighbors = []
        size = self.size
        for d in range(self.dim):
            mut_state[d] -= 1
            s_new = tuple(mut_state)
            condition = all(s>= mn for s, mn in zip(s_new, self.__min))
            if condition:
                neighbors.append(s_new)
            mut_state[d] += 2
            s_new = tuple(mut_state)
            condition = all(s < mx for s, mx in zip(s_new, size))
            if condition:
                neighbors.append(s_new)
            mut_state[d] -= 1
        return neighbors

    def get_rates(self, state):
        neighbors = self.get_neighbors(state)
        rates = []
        for n in neighbors:
            delta = (self.energy[state] - self.energy[n])/(k_B * self.temp)
            rates.append(self.rate * np.exp(delta))
        return np.array(rates)


def kmc_step(s0, lattice):
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


def kmc_run(s0, lattice, tau):
    states = []
    times = []
    curr = s0
    t = 0.
    states.append(curr)
    while t < tau:
        curr, dt = kmc_step(curr, lattice)
        t += dt
        if t > tau:
            t = tau
        states.append(curr)
        times.append(t)
    return states, np.array(times)


if __name__ == '__main__':
    import kmc_plot
    import matplotlib.pyplot as plt
    T = 300
    lattice = SCLattice((5, 5), 0.5*k_B*300, 1.0, 300.)
    s0 = (2, 2)
    s_traj = []
    t_traj = []
    for i in range(10000):
        print(i+1/100)
        states, times = kmc_run(s0, lattice, 10.)
        s_traj.append(states)
        t_traj.append(times)
    hist = kmc_plot.imshow_animate(s_traj, t_traj, np.linspace(0., 3., 100), lattice)
