"""
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import nquad
import cmath


# ----------------------------------------------------------------------
# Magnetic field
#

def dB_biot_savart(dl, R, cur=1.0, mu_0=4*np.pi*1E-7):
    """Returns the magnetic field at R generated by wire element dl.

    Parameters
    ----------
    dl : ndarray(dtype=float, dim=1)
        Vector with length dl, pointing in the direction of the wire
        element (x, y, z).
    R : ndarray(dtype=float, dim=1)
        Position vector from the wire element (dl) to the
        point where the field is evaluated (x, y, z).
    cur : float
        The current through the wire in Ampere (defaults to 1).
    mu_0 : float
        Magnatic permeability (the default is the magnetic
        permeability of free space).

    Returns
    -------
    dB : vector (float)
        Magnetic field of a wire element (dl) carrying a current (I).

    """
    return (mu_0 * cur * np.cross(dl, R)) / (4*np.pi * np.linalg.norm(R)**3)


def biot_savart(dl, R0, points, cur=1.0, mu_0=4*np.pi*1E-7):
    """Returns magnetic field at points generated by a wire.

    Parameters
    ----------
    dl : ndarray(dtype=float, dim=2)
        Array with length vectors of the wire elements carrying a current.
    R0 : ndarray(dtype=float, dim=2)
        Array with position vectors from the corresponding wire elements.
    points : ndarray(dtype=float, dim=2)
        Array with position vectors of points where the magnetic field is
        to be evaluated.
    cur : float
        Wire current in Amperes (defaults to 1 A).
    mu_0 : float
        Magnetic permeability of the surrounding medium (defaults to
        the magnetic permeability of vacuum).

    Returns
    -------
    B : ndarray(dtype=float, dim=2)
        Array of the magnetic field components at points.

    """
    R = points[:, None, :] - R0
    cross = np.cross(dl, R)
    norm = np.linalg.norm(R, axis=2)
    dB = (mu_0 * cur * cross) / (4*np.pi * norm[:, :, None]**3)
    return np.sum(dB, axis=1)


def plot_coil(R, dl, ax=None):
    """Plots coil geometry based on position vector and segment lengths.

    Parameters
    ----------
    R : ndarray(dtype=float, dim=2)
        Array with position vectors for line segments.
    dl : ndarray(dtype=float, dim=2)
        Array with length vectors for line segments.
    ax : matplotlib.axes, optional.
        Axes handle; if not provided, a new figure will be created.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    ax.quiver(R[:, 0], R[:, 1], R[:, 2],
              dl[:, 0], dl[:, 1], dl[:, 2], pivot='middle')
    return ax


def plot_mf(x, y, B, d='z', levels=10, ax=None):
    """Plots magnetic field.

    Parameters
    ----------
    x : ndarray(dtype=float, dim=1)
        Coordinates in x-direction.
    y : ndarray(dtype=float, dim=1)
        Coordinates in y-direction.
    B : ndarray(dtype=float, dim=2) or ndarray(dtype=float, dim=1)
        Can be:
        - an array of vectors with magnetic field strength in x, y, z, or
        - an array with the magnetic field strength in z-direction.
    d : {'x', 'y', 'z', 'norm'}, defaults to 'z'.
        Direction of field to plot; only relevant in case B represents
        a vector field (B.ndim == 2).
    levels : int
        Number of color levels to use in the contour plot.
    ax : matplotlib.axes, optional.
        Axes handle; if not provided, a new figure will be created.

    Returns
    -------
    ax : matplotlib.axes
        Axes handle.
    cs : matplotlib.contour.QuadContourSet
        Contour set handle.

    """
    Nx = len(x)
    Ny = len(y)
    if B.ndim == 2:
        d = {"x": 0, "y": 1, "z": 2}[d]
        B = np.linalg.norm(B, axis=1) if d == "norm" else B[:, d]
    if ax is None:
        _, ax = plt.subplots()
    cs = ax.contourf(x, y, vector2matrix(B, Ny, Nx), levels=levels)
    ax.set_aspect('equal', adjustable='box')
    return ax, cs


def vector2matrix(V, Nx, Ny):
    """Transforms vector to matrix.

    Parameters
    ----------
    V : ndarray(dtype=float, dim=1)
        Vector to be transformed of length Nx*Ny.
    Nx : int
        Number of points in x-direction.
    Ny : int
        Number of points in y-direction.

    Returns
    -------
    M : ndarray(dtype=float, dim=2)
        Matrix.

    """
    return np.reshape(V, (Ny, Nx))


def matrix2vector(M, Nx, Ny):
    """Transforms vector to matrix.

    Parameters
    ----------
    M : ndarray(dtype=float, dim=2)
        Matrix of shape (Ny, Nx)
    Nx : int
        Number of points in x-direction.
    Ny : int
        Number of points in y-direction.

    Returns
    -------
    V : ndarray(dtype=float, dim=1)
        Vector.

    """
    return np.reshape(M, Nx*Ny)


def DK_coil():
    points = np.array([[-10, 0, 100],
                       [-10, 0, 70],
                       [-60, 0, 25],
                       [-40, 0, 5],
                       [40, 0, 5],
                       [60, 0, 25],
                       [10, 0, 70],
                       [10, 0, 100]])/1000.0
    lines = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [4, 5],
                      [5, 6],
                      [6, 7]])
    return points, lines


def wire_coil(h, L):
    """Simple wire parallel to x-axis.

    Parameters
    ----------
    h : float
        Distance between the wire and xy-plane.
    L : float
        Length of the wire.

    Returns
    -------
    points : nd.array(dtype=float, dim=2)
        Coordinates.
    lines : nd.array(dtype=int, dim=2)
        Connectivity matrix.

    """
    points = np.array([[-L/2, 0, h],
                       [L/2, 0, h]])
    lines = np.array([[0, 1]])
    return points, lines


def system_matrix(rho, dx, dy, Nx, Ny):
    """Returns system matrix in case self-inductance is negligible.

    Returns the system matrix [M] required to calculate the electric
    vector potential {T} as a function of the magnetic flux vector
    {b}:

        [M]{T} = {b},

    following the paper of Nagel [cite:Nagel2019]. Please note that
    self-inductance is excluded in this case.

    Parameters
    ----------
    rho : float
        Resistivity of the flat plate.
    dx : float
        X-coordinate spacing.
    dy : float
        Y-coordinate spacing.
    Nx : int
        Number of grid points in X-direction.
    Ny : int
        Number of grid points in Y-direction.

    Returns
    -------
    M : nd.array(dtype=float, dim=2)
        System matrix.

    Nagel2019: Nagel, J. Finite-difference simulation of eddy currents
    in nonmagnetic sheets via electric vector potential. IEEE
    Transactions on Magnetics. DOI: 10.1109/TMAG.2019.2940204

    """
    dxdy = dx/dy
    dydx = dy/dx
    diagonals = [-rho*dxdy*np.ones(Nx*Ny-Nx),
                 -rho*dydx*np.ones(Nx*Ny-1),
                 2*rho*(dxdy+dydx)*np.ones(Nx*Ny),
                 -rho*dydx*np.ones(Nx*Ny-1),
                 -rho*dxdy*np.ones(Nx*Ny-Nx)]
    offsets = [-Nx, -1, 0, 1, Nx]
    M = diags(diagonals, offsets).toarray()
    return M


def rhs(Bz, omega, dx, dy):
    """Returns magnetic flux vector.

    Returns the magnetic flux vector {b} required to calculate the
    electric vector potential {T} according to:

        [M]{T} = {b},

    following the work of Nagel [cite:Nagel2019]. Please note that {b}
    consists of complex numbers.

    Parameters
    ----------
    Bz : nd.array(dtype=float, dim=1)
        Z-compnent of magnetic field.
    omega : float
        Angular frequency.
    dx : float
        X-coordinate spacing.
    dy : float
        Y-coordinate spacing.

    Returns
    -------
    b : nd.array(dtype=complex, dim=1)
        Magnetic flux vector.

    Nagel2019: Nagel, J. Finite-difference simulation of eddy currents
    in nonmagnetic sheets via electric vector potential. IEEE
    Transactions on Magnetics. DOI: 10.1109/TMAG.2019.2940204

    """
    return omega*Bz*dx*dy*1j


def current_coordinates(x, y):
    """Returns the coordinates for the currents.

    Considering the illustration below, the magnetic vector potential
    is defined at the circles (O), while the currents are defined at
    the crosses (x).

         O-------O-------O-------O
         |       |       |       |
         |   x   |   x   |   x   |
         |       |       |       |
         O-------O-------O-------O
         |       |       |       |
         |   x   |   x   |   x   |
    ^ y  |       |       |       |
    |    O-------O-------O-------O
    |
    0----> x

    Parameters
    ----------
    x : nd.array(dtype=float, dim=1)
        Array X-coordinates where the electric vector potentials are
        defined.
    y : nd.array(dtype=float, dim=1)
        Array Y-coordinates where the electric vector potentials are
        defined.

    Returns
    -------
    xc : nd.array(dtype=float, dim=1)
        Array X-coordinates where the currents are defined.
    yc : nd.array(dtype=float, dim=1)
        Array Y-coordinates where the currents are defined.

    """
    xc = (x[:-1] + x[1:])/2
    yc = (y[:-1] + y[1:])/2
    return xc, yc


def derivative_matrices(dx, dy, Nx, Ny):
    """Returns derivative matrices along X and Y.

    The derivative matrices are used to determine the current in X-
    and Y-direction from the electric vector potential {T} as:

        {Jx} = +[Dy]{T}, and: {Jy} = -[Dx]{T},

    following the work of Nagel [cite:Nagel2019].

    In more detail, and considering the illustration below, the
    derivatives at a point along X and Y are defined as:

        d/dx = (P2+P4-P1-P3)/(2 dx),

        d/dy = (P1+P2-P3-P4)/(2 dy).

    The derivative matrices are simply used to express these relations
    for all points in conveniently.

          |       |
        --P1------P2--
          |       |
          |   x   |
          |       |
        --P3------P4--
          |       |

    Nagel2019: Nagel, J. Finite-difference simulation of eddy currents
    in nonmagnetic sheets via electric vector potential. IEEE
    Transactions on Magnetics. DOI: 10.1109/TMAG.2019.2940204

    """
    N_col = Nx*Ny
    N_row = (Nx-1)*(Ny-1)
    Dx = np.zeros((N_row, N_col))
    Dy = np.zeros((N_row, N_col))
    for j in range(0, Ny-1):
        for i in range(0, Nx-1):
            Dx[j*(Nx-1)+i, j*Nx+i] = -1
            Dx[j*(Nx-1)+i, j*Nx+i+Nx] = -1
            Dx[j*(Nx-1)+i, j*Nx+i+1] = 1
            Dx[j*(Nx-1)+i, j*Nx+i+1+Nx] = 1
            Dy[j*(Nx-1)+i, j*Nx+i] = -1
            Dy[j*(Nx-1)+i, j*Nx+i+Nx] = 1
            Dy[j*(Nx-1)+i, j*Nx+i+1] = -1
            Dy[j*(Nx-1)+i, j*Nx+i+1+Nx] = 1
    return Dx/2/dx, Dy/2/dy


def mask_bc(Nx, Ny):
    """Returns a mask with False values for the domain boundaries."""
    bot = np.arange(0, Nx)
    right = np.arange(Nx-1, Nx*Ny, Nx)
    top = np.arange(Nx*Ny-Nx, Nx*Ny)
    left = np.arange(0, Nx*Ny, Nx)
    ind = np.unique(np.concatenate((bot, right, top, left), axis=None))
    mask = np.ones(Nx*Ny, dtype=bool)
    mask[ind] = False
    return mask


def plot_current_density_cf(x, y, Jx, Jy, d='mag', ax=None):
    """Plots current density using contourf.

    Parameters
    ----------
    x : ndarray(dtype=float, dim=1)
        Coordinates in x-direction where electric vector potential T
        is defined.
    y : ndarray(dtype=float, dim=1)
        Coordinates in y-direction where electric vector potential T
        is defined.
    Jx : ndarray(dtype=float, dim=1)
        Current density in x-direction.
    Jy : ndarray(dtype=float, dim=1)
        Current density in y-direction.
    d : {'x', 'y', 'mag'}, defaults to 'mag'
        Direction of current to plot.
    ax : matplotlib.axes, optional.
        Axes handle; if not provided, a new figure will be created.

    Returns
    -------
    ax : matplotlib.axes
        Axes handle.
    cs : matplotlib.contour.QuadContourSet
        Contour set handle.


    """
    Nx, Ny = len(x), len(y)
    xc, yc = current_coordinates(x, y)
    if ax is None:
        _, ax = plt.subplots()
    if d == 'mag':
        J_mag = np.sqrt(Jx.real**2 + Jy.real**2)
        cs = ax.contourf(xc, yc, vector2matrix(J_mag, Nx-1, Ny-1), levels=10)
    elif d == 'x':
        cs = ax.contourf(xc, yc, vector2matrix(Jx.real, Nx-1, Ny-1), levels=10)
    elif d == 'y':
        cs = ax.contourf(xc, yc, vector2matrix(Jy.real, Nx-1, Ny-1), levels=10)
    else:
        cs = None
    ax.set_aspect('equal', adjustable='box')
    return ax, cs


def plot_current_density(x, y, Jx, Jy, d='mag', ax=None):
    """Plots current density using pcolormesh.

    Parameters
    ----------
    x : ndarray(dtype=float, dim=1)
        Coordinates in x-direction where electric vector potential T
        is defined.
    y : ndarray(dtype=float, dim=1)
        Coordinates in y-direction where electric vector potential T
        is defined.
    Jx : ndarray(dtype=float, dim=1)
        Current density in x-direction.
    Jy : ndarray(dtype=float, dim=1)
        Current density in y-direction.
    d : {'x', 'y', 'mag'}, defaults to 'mag'
        Direction of current to plot.
    ax : matplotlib.axes, optional.
        Axes handle; if not provided, a new figure will be created.

    Returns
    -------
    ax : matplotlib.axes
        Axes handle.
    cs : matplotlib.contour.QuadContourSet
        Contour set handle.


    """
    Nx, Ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)
    if ax is None:
        _, ax = plt.subplots()
    if d == 'mag':
        J_mag = np.sqrt(Jx.real**2 + Jy.real**2)
        mg = ax.pcolormesh(X, Y, vector2matrix(J_mag, Nx-1, Ny-1))
    elif d == 'x':
        mg = ax.pcolormesh(X, Y, vector2matrix(Jx.real, Nx-1, Ny-1))
    elif d == 'y':
        mg = ax.pcolormesh(X, Y, vector2matrix(Jy.real, Nx-1, Ny-1))
    else:
        mg = None
    ax.set_aspect('equal', adjustable='box')
    return ax, mg


def plot_current_streamlines(X, Y, Jx, Jy, ax=None):
    """Plots current streamlines.

    Parameters
    ----------
    x : ndarray(dtype=float, dim=1)
        Coordinates in x-direction where electric vector potential T
        is defined.
    y : ndarray(dtype=float, dim=1)
        Coordinates in y-direction where electric vector potential T
        is defined.
    Jx : ndarray(dtype=float, dim=1)
        Current density in x-direction.
    Jy : ndarray(dtype=float, dim=1)
        Current density in y-direction.
    ax : matplotlib.axes, optional.
        Axes handle; if not provided, a new figure will be created.

    Returns
    -------
    ax : matplotlib.axes, optional.
        Axes handle.
    sp : matplotlib.streamplot.StreamplotSet
        Streamplot handle.

    """
    Nx, Ny = len(X), len(Y)
    xc, yc = current_coordinates(X, Y)
    if ax is None:
        _, ax = plt.subplots()
    sp = ax.streamplot(xc, yc,
                       vector2matrix(Jx.real, Nx-1, Ny-1),
                       vector2matrix(Jy.real, Nx-1, Ny-1),
                       density=0.6, linewidth=1, color='white')
    ax.set_aspect('equal', adjustable='box')
    return ax, sp


def contour_matrices(dx, dy, Nx, Ny, omega):
    """Returns countour matrices.

    The contour matrices Cx and Cy relate the magnetic vector
    potential A to the self inductance potential p as:

        {p} = [Cx]{a_x} + [Cy]{a_y}

    with a_x and a_y denoting the x- and y-components of the magnetic
    vector potential A, following the work of Nagel [cite:Nagel2019].

    Parameters
    ----------
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    Nx : int
        Number of grid points in x-direction.
    Ny : int
        Number of grid points in y-direction.
    omega : float
        Excitation frequency.

    Returns
    -------
    Cx : nd.array(dtype=float, ndim=2)
        Contour matrix in 'x-direction'.
    Cy : nd.array(dtype=float, ndim=2)
        Contour matrix in 'y-direction'.

    Nagel2019: Nagel, J. Finite-difference simulation of eddy currents
    in nonmagnetic sheets via electric vector potential. IEEE
    Transactions on Magnetics. DOI: 10.1109/TMAG.2019.2940204

    """
    N_col = (Nx-1)*(Ny-1)
    N_row = Nx*Ny
    Cx = np.zeros((N_row, N_col), dtype='cfloat')
    Cy = np.zeros((N_row, N_col), dtype='cfloat')
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            Cx[j*Nx+i, (j-1)*(Nx-1)+i] = 1j
            Cx[j*Nx+i, j*(Nx-1)+i] = -1j
            Cx[j*Nx+i, j*(Nx-1)+i-1] = -1j
            Cx[j*Nx+i, (j-1)*(Nx-1)+i-1] = 1j
            Cy[j*Nx+i, (j-1)*(Nx-1)+i] = 1j
            Cy[j*Nx+i, j*(Nx-1)+i] = 1j
            Cy[j*Nx+i, j*(Nx-1)+i-1] = -1j
            Cy[j*Nx+i, (j-1)*(Nx-1)+i-1] = -1j
    return omega*dx*Cx/2, omega*dy*Cy/2


def volume_int_I(dx, dy, t):
    """Returns volume integral needed for Biot Savart matrices.

    Following the work of Nagel [cite:Nagel2019], the volume integral
    is defined as:

        I = 8 * Int 1/sqrt(x^2 + y^2 + z^2) dxdydz,

    with the bounds:
        x : (0, dx/2)
        y : (0, dy/2)
        z : (0, t/2)

    The value of I is obtained through numerical integration.

    Parameters
    ----------
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    t : float
        Plate thickness.

    Returns
    -------
    I : float
        Volume integral.
    err : float
        Estimate of absolute error.

    Nagel2019: Nagel, J. Finite-difference simulation of eddy currents
    in nonmagnetic sheets via electric vector potential. IEEE
    Transactions on Magnetics. DOI: 10.1109/TMAG.2019.2940204

    """
    def f(x, y, z):
        return 1/np.sqrt(x**2 + y**2 + z**2)
    res = nquad(f, [[0, dx/2],
                    [0, dy/2],
                    [0, t/2]])
    I = res[0]
    err = res[1]
    return 8*I, err


def biot_savart_matrix(x, y, t, mu_0=4*np.pi*1E-7):
    """Returns Biot-Savart matrix.

    The Biot-Savart matrix relates the eddy current density J to the
    magnetic vector A as:

        {a_x} = [N]{J_x},

        {a_y} = [N]{J_y},

    with a_x and a_y denoting the x- and y-components of the magnetic
    vector potential A, and J_x and J_y denoting the x- and
    y-components of the eddy current density.

    Parameters
    ----------
    x : nd.array(dtype=float, dim=1)
        Array X-coordinates where the electric vector potentials are
        defined.
    y : nd.array(dtype=float, dim=1)
        Array Y-coordinates where the electric vector potentials are
        defined.
    t : float
        Plate thickness.
    mu_0 : float, optional
        Magnatic permeability (the default is the magnetic
        permeability of free space).


    Returns
    -------
    I : float
        Volume integral.
    err : float
        Estimate of absolute error.

    Nagel2019: Nagel, J. Finite-difference simulation of eddy currents
    in nonmagnetic sheets via electric vector potential. IEEE
    Transactions on Magnetics. DOI: 10.1109/TMAG.2019.2940204

    """
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    xc, yc = current_coordinates(x, y)
    pos = np.array([np.array([x, y, 0]) for y in yc for x in xc])
    N = np.zeros((len(xc)*len(yc), len(xc)*len(yc)))
    dV = dx*dy*t
    I, _ = volume_int_I(dx, dy, t)
    for i, loc in enumerate(pos):
        with np.errstate(divide='ignore'):
            N[i] = dV/np.linalg.norm(loc-pos, axis=1)
        N[i, i] = I
    return (mu_0/4/np.pi)*N


def phase_shift(J, phi):
    """Applies phase shift to array with complex numbers.

    Parameters
    ----------
    J : nd.array(dtype=complex, dim=1)
        Array with complex numbers to which the phase shift is applied.
    phi : float
        Phase shift angle (in radians).

    Returns
    -------
    Js : nd.array(dtype=complex, dim=1)
        Phase shifted numbers.

    """
    r = np.abs(J)
    theta = np.angle(J) + phi
    Js = np.array([cmath.rect(r[i], theta[i]) for i in range(len(r))])
    return Js
