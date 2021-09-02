"""Eppy example: square coil parallel to plate.

This example illustrates how to use eppy to calculate the eddy
currents in a plate due to a flat square coil.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
import time
import sys
import os
sys.path.append("..")

import eppy
from coil_geom import coil_segments

# init timer
start_time = time.perf_counter()


# --------------------------------------------------------
# Plate
#

# dimensions
Lx = .25
Ly = .25
t = 1E-3

# cell size and number of cells in x- and y-direction
dx = dy = .005
Nx = int(np.ceil(Lx/dx + 1))
Ny = int(np.ceil(Ly/dy + 1))

# position vector for points on XY plane
X = np.linspace(-Lx/2, Lx/2, Nx)
Y = np.linspace(-Ly/2, Ly/2, Ny)
pos = np.array([np.array([x, y, 0]) for y in Y for x in X])

# position vector for center of cells where the current is defined
xc, yc = eppy.current_coordinates(X, Y)

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
M = eppy.system_matrix(rho, dx, dy, Nx, Ny)
N = eppy.biot_savart_matrix(X, Y, t)
Cx, Cy = eppy.contour_matrices(dx, dy, Nx, Ny, omega)
Dx, Dy = eppy.derivative_matrices(dx, dy, Nx, Ny)
K = M + Cx@N@Dy - Cy@N@Dx

# unknown electric vector potential
T = np.zeros(Nx*Ny, dtype=complex)


# boundary condition mask
mask = eppy.mask_bc(Nx, Ny)

# time after init
matrix_time = time.perf_counter()
print("System matrix: {:.2f} seconds".format(matrix_time-init_time))


# --------------------------------------------------------
# Coil geometry and magnetic field
#

# parameters
height = 10E-3
radius = 5E-3
width = 25E-3
d_rad = (1-np.sqrt(2)/2)*radius

#  points
points = np.array([[width, width-radius, height],        # 0
                   [width-d_rad, width-d_rad, height],   # 1
                   [width-radius, width, height],        # 2
                   [-width+radius, width, height],       # 3
                   [-width+d_rad, width-d_rad, height],  # 4
                   [-width, width-radius, height],       # 5
                   [-width, -width+radius, height],      # 6
                   [-width+d_rad, -width+d_rad, height], # 7
                   [-width+radius, -width, height],      # 8
                   [width-radius, -width, height],       # 9
                   [width-d_rad, -width+d_rad, height],  # 10
                   [width, -width+radius, height]])      # 11

lines = np.array([[2, 3],
                  [5, 6],
                  [8, 9],
                  [11, 0]])

arcs = np.array([[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8],
                 [9, 10, 11]])

esize = 2*np.pi*radius/16
R, dl = coil_segments(points, esize, lines=lines, arcs=arcs)

# magnetic field
B = eppy.biot_savart(dl, R, pos, current)
Bz = B[:, 2]

# time after coil generation
coil_time = time.perf_counter()
print("Magnetic field: {:.2f} seconds".format(coil_time-matrix_time))


# --------------------------------------------------------
# Solve
#

# calculate flux
flux = eppy.rhs(Bz, omega, dx, dy)

# solve system
T[mask] = np.linalg.solve(K[:, mask][mask, :], flux[mask])

# currents calculated
solve_time = time.perf_counter()
print("Problem solved: {:.2f} seconds".format(solve_time-coil_time))


# --------------------------------------------------------
# Plot
#

# The functions below are sloppy in the sense that they use variables
# available in the global namespace. Some care is advised.

def plot_data(T, flux, theta=0):
    """Plots overview of data at time theta.

    Parameters
    ----------
    T : ndarray(dtype=float, dim=1)
        Electric vector potential {T}.
    flux : ndarray(dtype=float, dim=1)
        Magnetic flux vector {b} due to the coil.
    theta : float from 0 to 2*pi
        Angle for which to plot the data.

    """

    # shift data and calculate currents and magnetic field
    T_shifted = eppy.phase_shift(T, theta)
    flux_shifted = eppy.phase_shift(flux, theta)
    Jx = np.dot(Dy, T_shifted)
    Jy = -np.dot(Dx, T_shifted)
    J_mag = np.sqrt(Jx.real**2 + Jy.real**2)
    flux_eddy = np.dot(Cx@N@Dy - Cy@N@Dx, T_shifted) # magnetic flux due to eddy currents
    B_total = (np.real(flux_shifted) + np.real(flux_eddy))/omega/dx/dy

    # generate figure
    plt.figure(figsize=(8, 3))

    # plot coil
    ax_coil = plt.axes([0.06, 0.5, 0.2, 0.4],
                        xticks=[], yticks=[])
    plate = Rectangle((-Lx/2, -Ly/2), width=Lx, height=Ly,
                   facecolor="lightgray", edgecolor="black",
                   linewidth=0.5, zorder=1)
    ax_coil.add_patch(plate)
    ax_coil.text(-Lx/2.1, -Ly/2.1, "$\sigma$ = 25 kS/m \n t = 1 mm", fontsize=8)
    ax_coil.text(-Lx/1.9, 0, "250 mm", fontsize=8,
                 horizontalalignment="right",
                 verticalalignment="center",
                 rotation="vertical")
    ax_coil.text(0, -Ly/1.9, "250 mm", fontsize=8,
                 horizontalalignment="center",
                 verticalalignment="top")
    ax_coil.text(0, Ly/1.9, "Square coil \n100 mm above plate", fontsize=8,
                 horizontalalignment="center",
                 verticalalignment="bottom")
    ax_coil.plot(R[:, 0], R[:, 1], marker=".", markersize=1, zorder=10)
    ax_coil.axis("off")
    ax_coil.set_aspect("equal")
    ax_coil.set_xlim(-Lx/1.9, Lx/1.9)
    ax_coil.set_ylim(-Ly/1.9, Ly/1.9)

    # plot current
    ax_I = plt.axes([0.06, 0.12, 0.2, 0.25], xticks=[], yticks=[])
    t1 = np.linspace(0, 2*np.pi)
    t2 = np.linspace(-.5, 2*np.pi+0.5)
    ax_I.plot(t1, np.cos(t1))
    ax_I.plot(t2, np.cos(t2), ":", color="tab:blue")
    ax_I.plot([-.5, 2*np.pi+0.5], [0, 0], 'k', linewidth=0.5)
    ax_I.plot(theta, np.cos(theta), 'o', mfc="black", mec="black")
    ax_I.text(np.pi, 1.10, "Coil current", fontsize=8,
                 horizontalalignment="center",
                 verticalalignment="bottom")
    ax_I.axis("off")

    # plot magnetic field
    ax_mf = plt.axes([0.32, 0.1, 0.3, 0.79],
                     aspect="equal",
                     xticks=[], yticks=[])
    ax_mf.pcolormesh(X, Y, eppy.vector2matrix(B_total, Nx, Ny),
                     shading="nearest",
                     vmin=-2E-5, vmax=2E-5)
    ax_mf.text(0, Ly/1.9, "Z-component of magnetic field [T]", fontsize=8,
               horizontalalignment="center",
               verticalalignment="baseline")
    ax_mf.set_xlim(-Lx/2, Lx/2)
    ax_mf.set_ylim(-Ly/2, Ly/2)

    # plot eddy currents
    ax_ec = plt.axes([0.65, 0.1, 0.3, 0.79],
                     aspect="equal",
                     xticks=[], yticks=[])
    ax_ec.pcolormesh(X, Y, eppy.vector2matrix(J_mag, Nx-1, Ny-1),
                     vmin=0, vmax=6600)
    ax_ec.streamplot(xc, yc,
                     eppy.vector2matrix(Jx.real, Nx-1, Ny-1),
                     eppy.vector2matrix(Jy.real, Nx-1, Ny-1),
                     density=0.4, linewidth=1, color="white")
    ax_ec.text(0, Ly/1.92, "Eddy current density [A/m$^2$]", fontsize=8,
               horizontalalignment="center",
               verticalalignment="baseline")
    ax_ec.set_xlim(-Lx/2, Lx/2)
    ax_ec.set_ylim(-Ly/2, Ly/2)
    plt.show()

def generate_pngs(thetas, folder="img/"):
    for i, theta in enumerate(thetas):
        plot_data(T, flux, theta)
        plt.savefig(folder + "image{:03d}.png".format(i))
        plt.close()

def generate_gif(folder="img/"):
    images = []
    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith(".png"):
            file_path = os.path.join(folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimwrite(folder + "movie.gif", images, fps=20)
