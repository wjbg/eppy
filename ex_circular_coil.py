import numpy as np
import matplotlib.pyplot as plt
import time

import mf

# init timer
start_time = time.perf_counter()

# --------------------------------------------------------
# Plate
#

# dimensions
Lx = .3
Ly = .3
t = 1E-3

# cell size and number of cells in x- and y-direction
dx = dy = .01
Nx = int(np.ceil(Lx/dx + 1))
Ny = int(np.ceil(Ly/dy + 1))

# position vector for points on XY plane
X = np.linspace(-Lx/2, Lx/2, Nx)
Y = np.linspace(-Ly/2, Ly/2, Ny)
pos = np.array([np.array([x, y, 0]) for y in Y for x in X])

# conductivity and resistivity
sigma = 25E3  # S/m
rho = 1/sigma

# coil excitation frequency and current
f = 250E3
omega = 2*np.pi*f
current = 1.0

# time after init
init_time = time.perf_counter()
print("Initiation: {:.2f} seconds".format(init_time-start_time))

# system matrix
M = mf.system_matrix(rho, dx, dy, Nx, Ny)
N = mf.biot_savart_matrix(X, Y, t)
Cx, Cy = mf.contour_matrices(dx, dy, Nx, Ny, omega)
Dx, Dy = mf.derivative_matrices(dx, dy, Nx, Ny)
K = M + Cx@N@Dy - Cy@N@Dx

# unknown electric vector potential
T = np.zeros(Nx*Ny, dtype=complex)


# boundary condition mask
mask = mf.mask_bc(Nx, Ny)

# time after init
matrix_time = time.perf_counter()
print("System matrix: {:.2f} seconds".format(matrix_time-init_time))


# --------------------------------------------------------
# Coil geometry and magnetic field
#

#  points
radius = 25E-3
height = 10E-3
points = np.array([[radius, 0, height],
                   [0, radius, height],
                   [-radius, 0, height]])

# generate coil segments
esize = radius*np.pi*2/40
R, dl = mf.circle_segments_3p(points[0],
                              points[1],
                              points[2],
                              esize)

# magnetic field
B = mf.biot_savart(dl, R, pos, current)
Bz = B[:, 2]

# time after coil generation
coil_time = time.perf_counter()
print("Magnetic field: {:.2f} seconds".format(coil_time-matrix_time))


# --------------------------------------------------------
# Solve
#

# calculate flux
flux = mf.rhs(Bz, omega, dx, dy)

# solve system
T[mask] = np.linalg.solve(K[:, mask][mask, :], flux[mask])

# calculate currents
Jx = np.dot(Dy, T)
Jy = -np.dot(Dx, T)

# calculate induced magnetic field
B_ind = np.dot(Cx@N@Dy - Cy@N@Dx, T)  # this still needs some thinking...
                                      # we take the Re()

# currents calculated
solve_time = time.perf_counter()
print("Problem solved: {:.2f} seconds".format(solve_time-coil_time))


# --------------------------------------------------------
# Plot magnetic field and current density
#

# plot Z-component of coil magnetic field and eddy current distr.
fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(12, 6))
_, cs_b = mf.plot_mf(X, Y, Bz, d='z', levels=10, ax=ax[0])
_, cs_I = mf.plot_current_density(X, Y, Jx, Jy, d='mag', ax=ax[1])
_, sp_I = mf.plot_current_streamlines(X, Y, Jx, Jy, ax=ax[1])

# labels
ax[0].set_title('Z-component of magnetic field (coil)')
ax[0].set_xlabel('x [m]')
ax[0].set_ylabel('y [m]')
ax[1].set_title('Eddy current distribution [A/m^2]')
ax[1].set_xlabel('x [m]')
ax[1].set_ylabel('y [m]')

# add color bar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
fig.colorbar(cs_I, cax=cbar_ax)

# limits
ax[0].set_xlim([-.14, .14])
ax[0].set_ylim([-.14, .14])
ax[1].set_xlim([-.14, .14])
ax[1].set_ylim([-.14, .14])

# plot time
plot_time = time.perf_counter()
print("Data plotted: {:.2f} seconds".format(plot_time-solve_time))
