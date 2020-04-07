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
        self.std_energy = std_energy
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

    def shuffle(self):
        self.energy=np.random.normal(0., self.std_energy, self.size)

    def two_dim_coarse_grain(self, new_size):
        assert len(new_size) == 2 == self.dim
        window = (int(self.size[0]/new_size[0]), int(self.size[1]/new_size[1]))
        fe = np.zeros(new_size)
        Z = np.exp(-self.energy/(k_B*T))
        for i in range(new_size[0]):
            for j in range(new_size[1]):
                fe[i,j] = np.sum(Z[i*window[0]:(i+1)*window[0], j*window[1]:(j+1)*window[1]])
        fe = -k_B * self.temp * np.log(fe)
        self.energy = fe - np.mean(fe)
        self.size = new_size
        self.std_energy = np.std(fe)


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
    times.append(t)
    while t < tau:
        curr, dt = kmc_step(curr, lattice)
        t += dt
        if t > tau:
            t = tau
        states.append(curr)
        times.append(t)
    return states, np.array(times)


if __name__ == '__main__':
    import plot
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
        states, times = kmc_run(s0, lattice, t_max)
        s_traj.append(states)
        t_traj.append(times)
    plt = plot.plot_mean_msd(s0, s_traj, t_traj, target_times)
