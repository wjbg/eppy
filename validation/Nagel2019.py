"""Generate figures 5 and 6 in Nagel (2019).

This scripts reproduces figures 5 and 6 in the paper by Nagel as a
check for code implementation. The original publication can be found
via: https://ieeexplore.ieee.org/document/8902282

Nagel, J. (2019). Finite-Difference Simulation of Eddy Currents in
Nonmagnetic Sheets via Electric Vector Potential. IEEE Transactions on
Magnetics, 55, 1-8.

"""

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

import eppy


# --------------------------------------------------------
# Plate
#

# dimensions
Lx = .02
Ly = .01
t = 0.001

# cell size and number of cells in x- and y-direction
dx = dy = .0005
Nx = int(np.ceil(Lx/dx + 1))
Ny = int(np.ceil(Ly/dy + 1))

# position vector for points on XY plane
X = np.linspace(-Lx/2, Lx/2, Nx)
Y = np.linspace(-Ly/2, Ly/2, Ny)
pos = np.array([np.array([x, y, 0]) for y in Y for x in X])

# conductivity and resistivity
sigma = 5E6  # S/m
rho = 1/sigma

# coil excitation frequency and current
f = 10E3
omega = 2*np.pi*f

# magnetic field
B_ext = -0.1
B = np.ones(Nx*Ny)*B_ext

# system matrix
M = eppy.system_matrix(rho, dx, dy, Nx, Ny)
N = eppy.biot_savart_matrix(X, Y, t)
Cx, Cy = eppy.contour_matrices(dx, dy, Nx, Ny, omega)
Dx, Dy = eppy.derivative_matrices(dx, dy, Nx, Ny)
K = M + Cx@N@Dy - Cy@N@Dx

# unknown electric vector potential
T = np.zeros(Nx*Ny, dtype=complex)

# boundary condition mask
mask = eppy.mask_bc(Nx, Ny)


# --------------------------------------------------------
# Solve
#

# calculate flux
flux = eppy.rhs(B, omega, dx, dy)

# solve system
T[mask] = np.linalg.solve(K[:, mask][mask, :], flux[mask])


# --------------------------------------------------------
# Figure 5 from Nagel2019
#

# calculate currents
Jx = np.dot(Dy, T)
Jy = -np.dot(Dx, T)

# apply a phase shift
Jx = eppy.phase_shift(Jx, np.pi/2)
Jy = eppy.phase_shift(Jy, np.pi/2)

# scale and shift dimensions and current to match figure in paper
Xp = X*100 + 1
Yp = Y*100 + 0.5
Jx = Jx/1E6
Jy = Jy/1E6

# plot
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(9, 5))
_, pc = eppy.plot_current_density(Xp, Yp, Jx, Jy, d='mag', ax=ax)
_, sp = eppy.plot_current_streamlines(Xp, Yp, Jx, Jy, ax=ax)

# labels
ax.set_title('Current density [MA/m^2]\n Compare to Fig. 5 in Nagel2019')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

# add color bar
pc.set_clim(0, 100)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.87, 0.2, 0.05, 0.6])
fig.colorbar(pc, cax=cbar_ax)

# change colormap and line collor to match figure in Nagel2019
plt.set_cmap('jet')
sp.lines.set_color('black')

# for some reason it is difficult to change the arrow color...
# this seems to work:
for art in ax.get_children():
    if not isinstance(art, patches.FancyArrowPatch):
        continue
    art.set_edgecolor([0, 0, 0, 0])
    art.set_facecolor([0, 0, 0, 1]),
    art.set_mutation_scale(20)
    art.set_zorder(10)

ax.set_xlim([0, 2])
ax.set_ylim([0, 1])
plt.show()


# --------------------------------------------------------
# Figure 6 from Nagel2019
#

# calculate currents (this time there is no shift)
Jx = np.dot(Dy, T)
Jy = -np.dot(Dx, T)

# data from simulation
Jym = eppy.vector2matrix(Jy, Nx-1, Ny-1)
Jyline = (Jym[9,:]+Jym[10,:])/2
xc = (X[:-1]+X[1:])/2

# data from Fig. 6 in Nagel2019
fn = "Nagel2019_fig6_data.csv"
data = np.genfromtxt(fn, delimiter=',')
data_real = data[:,:2]
data_real= data_real[~np.isnan(data_real[:,0])]
data_imag = data[:,2:]

# scale units to match figure
xc = xc*100 + 1.0
Jyline = Jyline/1E6

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(7, 5))
ax.plot(data_real[:,0], data_real[:,1], 'xk')
ax.plot(data_imag[:,0], data_imag[:,1], 'vk')
ax.plot(xc, Jyline.real)
ax.plot(xc, Jyline.imag)

ax.set_xlabel('x-distance [m]')
ax.set_ylabel('current density [MA/m^2]')
ax.set_title('Compare to Fig. 6 in Nagel2019 (markers)')
ax.set_xlim(0, 2)
plt.show()
