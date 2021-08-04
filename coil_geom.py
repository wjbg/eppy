"""

"""

import numpy as np

# ----------------------------------------------------------------------
# Coil geometry
#

def coil_segments(points, esize, **kwargs):
    """Return position vectors and line segments for a coil.

    ~~~The coil is generated based on lines between coordinate points,
    and a connectivity matrix.

    Parameters
    ----------
    points : ndarray(dtype=float, dim=2)
        List of coordinates (x, y, z).
    esize : float
        Desired element length.
    **kwargs : keyword arguments
        See below.

    Keyword arguments
    -----------------
    **lines : ndarray(dtype=float, dim=2), optional
        Connectivity matrix (N, 2) with the start and end point of a
        line on each row.
    **arcs : ndarray(dtype=float, dim=2), optional
        Arc definition matrix (N, 3). The arc is defined by three
        points on its radius, and the current running from P1 via P2
        to P3. The first element [i, 0] corresponds to P1, the second
        element to P2 and the third element to P3 of that particular
        arc.

    Returns
    -------
    R : ndarray(dtype=float, dim=2)
        Array of position vectors for each small line segment.
    dl : ndarray(dtype=float, dim=2)
        Array of line segment vectors.

    """
    R = np.empty((0, 3), float)
    dl = np.empty((0, 3), float)
    if 'lines' in kwargs:
        lines = kwargs.get("lines")
        for line in lines:
            dR, ddl = line_segments(points[line[0]], points[line[1]], esize)
            R = np.vstack((R, dR))
            dl = np.vstack((dl, ddl))
    if 'arcs' in kwargs:
        arcs = kwargs.get("arcs")
        for arc in arcs:
            dR, ddl = circle_segments_3p(points[arc[0]], points[arc[1]],
                                         points[arc[2]], esize)
            R = np.vstack((R, dR))
            dl = np.vstack((dl, ddl))
    return R, dl


def line_segments(p1, p2, esize):
    """Return position vectors and line segments for straight line.

    Parameters
    ----------
    p1 : ndarray(dtype=float, dim=1)
        Coordinates of start point (x, y, z).
    p2 : ndarray(dtype=float, dim=1)
        Coordinates of end point (x, y, z).
    esize : float
        Desired element length.

    Returns
    -------
    R : ndarray(float, dim=2)
        Array with position vectors for all line segment.
    dl : ndarray(float dim=2)
        Array of line segment length vectors.

    """
    L = np.linalg.norm(p2-p1)
    nel = int(np.ceil(L/esize))
    esize = L/nel
    points = np.linspace(p1, p2, nel+1)
    R = (points[:-1, :] + points[1:, :])/2
    dl = np.tile(esize*(p2-p1)/L, (nel, 1))
    return R, dl


def circle_segments_3p(p1, p2, p3, esize, is_arc=False):
    """Return position vectors and line segments for an arc.

    The arc or circle is defined three points in three dimensions. The
    current is defined to run from p1 via p2 to p3.

    Parameters
    ----------
    p1 : ndarray(dtype=float, dim=1)
        Coordinates of the first point (x, y, z)
    p2 : ndarray(dtype=float, dim=1)
        Coordinates of the first point (x, y, z)
    p3 : ndarray(dtype=float, dim=1)
        Coordinates of the first point (x, y, z)
    esize : float
        Desired element length.
    is_arc : boolean, optional
        Indicate whether this is an arc running from p1 to p3 (True)
        or a full circle (False).

    Returns
    -------
    R : ndarray(dtype=float, dim=2)
        Array of position vectors for each line segment.
    dL : ndarray(dtype=float, dim=2)
        Array of line segment vectors.

    """
    center, radius = circle_radius_center_3p(p1, p2, p3)
    v1 = p1 - center
    v2 = p2 - center
    v3 = p3 - center

    normal, _, theta = rotation_direction_and_angle(v1, v2, v3)
    n = int(radius*theta/esize) if radius*theta/esize > 10 else 10
    theta = theta if is_arc else 2*np.pi
    alphas = np.linspace(0, theta, n, endpoint=False) + theta/n/2
    R = [np.dot(rotation_matrix_3d(normal, alpha), v1) for alpha in alphas]
    R = np.array(R) + center
    dl = [np.dot(rotation_matrix_3d(normal, alpha+np.pi/2), v1)
          for alpha in alphas]  # is the length of this thing correct?
    dl = (2*np.pi*radius/n) * np.array(dl)
    return R, dl


def spiral_segments(center, R_in, R_out, theta, h, esize):
    """Return position vectors and line segments for a spiral.

    The spiral is oriented to turn around the normal of the XY-plane.

    Parameters
    ----------
    center : ndarray(dtype=float, dim=1)
        Coordinates of the origin point (x, y, z)
    R_in : float
        Inner radius.
    R_out : float.
        Outer radius.
    theta : float
        Total winding angle.
    h : float
        Spiroal height.
    esize : float
        Desired element length.

    Returns
    -------
    R : ndarray(dtype=float, dim=2)
        Array of position vectors for each line segment.
    dL : ndarray(dtype=float, dim=2)
        Array of line segment vectors.

    """
    n = int(theta*R_out/esize)
    alpha = np.linspace(0, theta, n, endpoint=False) + theta/n/2
    radii = np.linspace(R_in, R_out, n)

    # Position vector.
    x = center[0] + radii*np.cos(alpha)
    y = center[1] + radii*np.sin(alpha)
    z = center[2] + np.linspace(0, h, n)
    R = np.array([x, y, z]).T

    # Element segment
    K = (R_out-R_in)/theta
    dx = -(R_in + K*alpha)*np.sin(alpha) + K*np.cos(alpha)
    dy = (R_in + K*alpha)*np.cos(alpha) + K*np.sin(alpha)
    dz = h/theta*np.ones(len(dx))
    dtheta = theta/n
    dL = dtheta*np.array([dx, dy, dz]).T
    return R, dL


def circle_radius_center_3p(p1, p2, p3):
    """Return radius and center of a circle that fits through three points.

    Parameters
    ----------
    p1 : ndarray(dtype=float, dim=1)
        Coordinates of first point (x, y, z).
    p2 : ndarray(dtype=float, dim=1)
        Coordinates of second point (x, y, z).
    p3 : ndarray(dtype=float, dim=1)
        Coordinates of third point (x, y, z).

    Returns
    -------
    P : ndarray(dtype=float, dim=1)
        Coordinates of center (x, y, z).
    R : float
        Radius.

    """
    # triangle edges
    t = np.linalg.norm(p3-p2)
    u = np.linalg.norm(p3-p1)
    v = np.linalg.norm(p2-p1)

    # semi-perimiter & triangle area
    s = (t + u + v)/2
    A = np.sqrt(s*(s-t)*(s-u)*(s-v))

    # circumradius
    R = t*u*v/(4*A)

    # barcyntric coordinates of center
    b1 = t**2 * (u**2 + v**2 - t**2)
    b2 = u**2 * (v**2 + t**2 - u**2)
    b3 = v**2 * (t**2 + u**2 - v**2)

    # cartesian coordinate of center
    P = np.column_stack((p1, p2, p3)).dot(np.hstack((b1, b2, b3)))
    P = P/(b1 + b2 + b3)
    return P, R


def rotation_direction_and_angle(v1, v2, v3, eps=1E-10):
    """Return outward normal and angles for an arc.

    The direction of the current in a cricle or arc segment is defined
    by means of three position vectors v1, v2, and v3, which share the
    origin in the arc center. The current flows from v1 via v2 to v3.
    This function returns the corresponding normal vector around which
    to rotate. In addition, the angle between the vectors (in
    direction of rotation) is returned as well.

    Parameters
    ----------
    v1 : ndarray(dtype=float, dim=1)
        Position vector of the first point w.r.t. the center.
    v2 : ndarray(dtype=float, dim=1)
        Position vector of the second point w.r.t. the center.
    v3 : ndarray(dtype=float, dim=1)
        Position vector of the third point w.r.t. the center.
    eps : float, optional
        Precision (defaults to 1E-10)

    Returns
    -------
    normal : ndarray(dtype=float, dim=1)
        Normal vector that determines current direction in arc.
    phi : float
        Angle between first and second vector in direction of rotation.
    theta : float.
        Angle between first and third vector in direction of rotation.

    """
    phi = np.arccos(np.dot(v1, v2))
    theta = np.arccos(np.dot(v1, v3))
    if (phi < eps) or (theta < eps):
        raise ValueError("Circle points coincide.")
    if abs(phi-np.pi) < eps:  # v1 and v2 are aligned
        # print("v1 and v2 are aligned")
        normal = -np.cross(v1, v3)
        theta = 2*np.pi - theta
    elif abs(theta-np.pi) < eps:  # v1 and v3 are aligned
        # print("v1 and v3 are aligned")
        normal = np.cross(v1, v2)
    else:  # v2 & v3 are not aligned with v1
        # print("vectors are not aligned")
        N12 = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))
        N13 = np.cross(v1, v3)/np.linalg.norm(np.cross(v1, v3))
        if np.linalg.norm(N12 - N13) < eps:  # v2 &  v3 lie in same circle half
            if theta - phi < eps:
                raise ValueError("Circle points coincide.")
            elif theta > phi:
                normal = np.cross(v1, v2)
            elif phi > theta:
                normal = -np.cross(v1, v2)
                theta = 2*np.pi - theta
                phi = 2*np.pi - phi
        else:
            normal = np.cross(v1, v2)
            theta = 2*np.pi-theta
    return normal, phi, theta


def tilt_and_rotate_coil():
    pass


def tilt_and_rotate(points, origin, new_z, theta, eps=1E-10):
    """Rotate points around a given origin.

    Parameters
    ----------
    points : ndarray(dtype=float, dim=2)
        Points to rotate from old CS to new CS.
    origin : ndarray(dtype=float, dim=1)
        Point around which to rotate.
    new_z : ndarray(dtype=float, dim=1)
        Direction of new_z axis in terms of old CS.
    theta : float
        Rotation angle around new z-axis.
    eps : float, optional
        Precision (defaults to 1E-10)

    Returns
    -------
    new_points : ndarray(dtype=float, dim=2)
        Points in new CS.

    """
    old_z = np.array([0.0, 0.0, 1.0])
    new_z = new_z/np.linalg.norm(new_z)
    phi = np.arccos(np.dot(old_z, new_z))

    # rotate from old to new z-axis
    if (phi < eps):
        # do nothing as old and new z-axis are equal
        R = points
    elif (abs(phi-np.pi) < eps):
        # rotate pi around x as z_new = -z_old
        axis = np.array([1.0, 0, 0])
        rot = rotation_matrix_3d(axis, phi)
        R = np.array([np.dot(rot, p - origin) for p in points])
    else:
        axis = np.cross(old_z, new_z)
        rot = rotation_matrix_3d(axis, phi)
        R = np.array([np.dot(rot, p - origin) for p in points])

    # rotate theta around new z axis
    rot = rotation_matrix_3d(new_z, theta)
    R = np.array([np.dot(rot, p) for p in R])
    return R + origin


def rotation_matrix_3d(axis, theta):
    """Return rotation matrix to rotate a vector in three dimensions.

    Makes use of the Euler-Rodrigues formula.

    Parameters
    ----------
    axis : ndarray(dtype=float, dim=1)
        Normal axis around which to rotate.
    theta : float
        Rotation angle (in radians).

    Returns
    -------
    rot : ndarray(dtype=float, dim=3)
        Rotation matrices for each angle in theta.

    """
    axis = axis/np.linalg.norm(axis)
    a = np.cos(theta/2)
    b, c, d = axis*np.sin(theta/2)
    rot = np.array([[a*a+b*b-c*c-d*d,     2*(b*c-a*d),     2*(b*d+a*c)],
                    [    2*(b*c+a*d), a*a+c*c-b*b-d*d,     2*(c*d-a*b)],
                    [    2*(b*d-a*c),     2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    return rot