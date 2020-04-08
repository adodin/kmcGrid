"""
Runs KMC Simulations on 2D Grid then visualizes population dynamics
Written by: Amro Dodin
"""

import kmcGrid.lattice
import kmcGrid.kmc
import kmcGrid.animate
import numpy as np

# Grid Parameters
T = 300
grid_size = (100, 100)
s0 = (50, 50)

# Simulation Parameters
num_traj = 1000
t_max = 30
n_plot = 40

# Build Lattice
lattice = kmcGrid.lattice.SCLattice((100, 100), 0., 1.0, T)

# Build Simulation Arrays
s_traj = []
t_traj = []
target_times = np.linspace(0, t_max, n_plot)

# Run Simulations
for i in range(num_traj):
    print(str(i + 1) + '/' + str(num_traj))
    lattice.shuffle()
    states, times = kmcGrid.kmc.run(s0, lattice, t_max)
    s_traj.append(states)
    t_traj.append(times)

# Animate Populations
plt = kmcGrid.animate.animate_populations(s_traj, t_traj, target_times, lattice)
