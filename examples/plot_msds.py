"""
Generates <MSD> vs time plots and (optionally) calculates instantaneous diffusion constants
"""

import kmcGrid.lattice
import kmcGrid.kmc
import kmcGrid.plot as plot
import kmcGrid.animate as animate
import numpy as np

# Grid Parameters
T = 300
grid_size = (100, 100)
s0 = (50, 50)

# Simulation Parameters
num_traj = 50000
t_max = 3.
n_plot = 2000

sig = 0.01

# Build Lattice
lattice = kmcGrid.lattice.SCLattice((100, 100), sig * kmcGrid.lattice.k_B* T, 1.0, T)

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

ax_mean, ax_slope = plot.plot_mean_msd(s0, s_traj, t_traj, target_times, 50)

sig = .5

# Build Lattice
lattice = kmcGrid.lattice.SCLattice((100, 100), sig * kmcGrid.lattice.k_B* T, 1.0, T)

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

ax_mean, ax_slope = plot.plot_mean_msd(s0, s_traj, t_traj, target_times, 50, target_axes=(ax_mean, ax_slope))

sig = 1.

# Build Lattice
lattice = kmcGrid.lattice.SCLattice((100, 100), sig * kmcGrid.lattice.k_B* T, 1.0, T)

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

ax_mean, ax_slope = plot.plot_mean_msd(s0, s_traj, t_traj, target_times, 50, target_axes=(ax_mean, ax_slope))

sig = 2.

# Build Lattice
lattice = kmcGrid.lattice.SCLattice((100, 100), sig * kmcGrid.lattice.k_B* T, 1.0, T)

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

ax_mean, ax_slope = plot.plot_mean_msd(s0, s_traj, t_traj, target_times, 50, target_axes=(ax_mean, ax_slope))

sig = 3.

# Build Lattice
lattice = kmcGrid.lattice.SCLattice((100, 100), sig * kmcGrid.lattice.k_B* T, 1.0, T)

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

ax_mean, ax_slope = plot.plot_mean_msd(s0, s_traj, t_traj, target_times, 50, target_axes=(ax_mean, ax_slope))
