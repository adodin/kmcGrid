"""
Script Showing Plotting of 2D Lattice Energy Landscapes at different scales
Written by: Amro Dodin
"""

import kmcGrid.lattice as latt
import kmcGrid.plot as plot
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

T = 300
sig_E = 1
lattice_size = (1000, 1000)
lattice = latt.SCLattice(lattice_size, sig_E * latt.k_B*T, 1.0, T)
hist1000, bin_edges = lattice.histogram_energy(bins=50, density=True, range=(-3., 3.))
lattice.two_dim_coarse_grain((500, 500))
hist500, bin_edges = lattice.histogram_energy(bins=bin_edges, density=True)
lattice.two_dim_coarse_grain((100, 100))
hist100, bin_edges = lattice.histogram_energy(bins=bin_edges, density=True)
norm = Normalize(vmin=-1.5*lattice.std_energy, vmax=1.5*lattice.std_energy)

plot.plot_2d_lattice(lattice, 1, axis_units=r'$\mu$m', axis_scale=(.1, .1), axis_rect=[0.15, 0.15, 0.8, 0.8],
                     imkwargs={'norm': norm}, fname='../FIG/grid-100nm.eps')

lattice.two_dim_coarse_grain((10, 10))
hist10, bin_edges = lattice.histogram_energy(bins=bin_edges, density=True)
plot.plot_2d_lattice(lattice, 1, axis_units=r'$\mu$m', axis_scale=(1., 1.), axis_rect=[0.15, 0.15, 0.8, 0.8],
                     imkwargs={'norm': norm}, fname='../FIG/grid-1um.eps')

# Code T
bin_mids = []
for i in range(len(bin_edges)-1):
    bin_mids.append(0.5*(bin_edges[i+1] + bin_edges[i]))

fig = plt.figure(figsize=[10., 10.])
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.plot(bin_mids, hist1000, linewidth=3.0)
ax.plot(bin_mids, hist500, linewidth=3.0)
ax.plot(bin_mids, hist100, linewidth=3.0)
ax.plot(bin_mids, hist10, linewidth=3.0)
ax.set_xlabel(r'$F/k_B T$')
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-0.01, 4.5)
ax.set_ylabel(r'$P(F)$')
ax.legend(['10 nm', '20 nm', '100 nm', r'1 $\mu$m'])
fig.savefig(fname='../FIG/energy-hists.eps')
plt.show()
