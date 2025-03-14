#!/usr/bin/env python3
# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""A math toolbox for MARILib.

:author: DRUOT Thierry, original Scilab implementation

:author: ROCHES Pascal, portage to Python
"""

# ruff: noqa D205, E501, N806, D103, N803, F821

import warnings

import numpy as np
from numpy.linalg import solve
from numpy.linalg.linalg import LinAlgError


def great_circle_distance(p1, p2):
    """Compute the great circle distance between 2 points
    p : (lat, long).
    """
    r = 6371000.0
    v1 = np.array([
        np.cos(p1[0]) * np.sin(p1[1]),
        np.cos(p1[0]) * np.cos(p1[1]),
        np.sin(p1[0]),
    ])
    v2 = np.array([
        np.cos(p2[0]) * np.sin(p2[1]),
        np.cos(p2[0]) * np.cos(p2[1]),
        np.sin(p2[0]),
    ])
    s = angle(
        np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2), typ=1
    )  # Get an angle between -pi and pi
    return r * abs(s)


def angle(sin_a, cos_a, typ=2):
    """Calculate an angle from its sine and cosine coordinates
    if typ=1 then -180<a_<180, if typ=2 then 0<a_<360.
    """
    epsilon = 1.0e-15
    if abs(cos_a - 1) < epsilon:
        a = 0.0
    elif abs(cos_a + 1) < epsilon:
        a = np.pi
    elif abs(sin_a - 1) < epsilon:
        a = 0.5 * np.pi
    elif abs(sin_a + 1) < epsilon:
        a = -0.5 * np.pi
    else:
        a = np.sign(sin_a) * abs(np.arccos(cos_a))
    if typ == 2:
        a = a + (1 - np.sign(a)) * np.pi
    if a > 2 * np.pi:
        a = a - 2.0 * np.pi
    return a


def rotate_vector(vec, axe, cos_a, sin_a):
    """Retrieve the vector vec rotated by the angle a around axe, using Olinde-Rodrigues formula
    v_star = vec + sin_a * np.cross(axe, vec) + (1 - cos_a) * np.cross(axe, np.cross(axe, vec)).
    """
    return (
        cos_a * vec + (1 - cos_a) * np.dot(vec, axe) * axe + sin_a * np.cross(axe, vec)
    )


def rotate_point(pivot, axe, angle, p0):
    """Retrieve the point coming from the rotation of p0 of angle, around the axis defined by pivot and axe."""
    vec = p0 - pivot
    r0 = vec - np.dot(vec, axe) * axe
    Q = np.array([[0, -axe[2], axe[1]], [axe[2], 0, -axe[0]], [-axe[1], axe[0], 0]])
    RmI = np.sin(angle) * Q + (1 - np.cos(angle)) * np.matmul(Q, Q)
    return p0 + np.matmul(RmI, r0)


def renorm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)


def cross_op(v):
    """Build the skew-symmetric matrix of vector v."""
    return np.array([[0, v[2], -v[1]], [-v[2], 0, v[0]], [v[1], -v[0], 0]])


def lin_interp_1d(x, X, Y):
    """Linear interpolation without any control.

    :param x: current position
    :param X: array of the abscissa of the known points
    :param Y: array of the known values at given abscissa
    :return: y the interpolated value of Y at x

    """
    n = np.size(X)
    for j in range(1, n):
        if x < X[j]:
            return Y[j - 1] + (Y[j] - Y[j - 1]) * (x - X[j - 1]) / (X[j] - X[j - 1])
    return Y[n - 2] + (Y[n - 1] - Y[n - 2]) * (x - X[n - 2]) / (X[n - 1] - X[n - 2])


def vander3(X):
    """Return the vandermonde matrix of a dim 3 array A = [X^2, X, 1]."""
    return np.array([
        [X[0] ** 2, X[0], 1.0],
        [X[1] ** 2, X[1], 1.0],
        [X[2] ** 2, X[2], 1.0],
    ])


def trinome(A, Y):
    """Calculate trinome coefficients from 3 given points
    A = [X2, X, 1] (Vandermonde matrix).
    """
    X = np.array([A[0][1], A[1][1], A[2][1]])
    X2 = np.array([A[0][0], A[1][0], A[2][0]])

    det = X2[0] * (X[1] - X[2]) - X2[1] * (X[0] - X[2]) + X2[2] * (X[0] - X[1])

    adet = Y[0] * (X[1] - X[2]) - Y[1] * (X[0] - X[2]) + Y[2] * (X[0] - X[1])

    bdet = X2[0] * (Y[1] - Y[2]) - X2[1] * (Y[0] - Y[2]) + X2[2] * (Y[0] - Y[1])

    cdet = (
        X2[0] * (X[1] * Y[2] - X[2] * Y[1])
        - X2[1] * (X[0] * Y[2] - X[2] * Y[0])
        + X2[2] * (X[0] * Y[1] - X[1] * Y[0])
    )

    if det != 0:
        C = np.array([adet / det, bdet / det, cdet / det])
    elif X[0] != X[2]:
        C = np.array([0.0, Y[0] - Y[2], Y[2] * X[0] - Y[0] * X[2] / (X[0] - X[2])])
    else:
        C = np.array([0.0, 0.0, (Y[0] + Y[1] + Y[2]) / 3.0])

    return C


def maximize_1d(xini, dx, *fct):
    """Optimize 1 single variable, no constraint.

    :param xini: initial value of the variable.
    :param dx: fixed search step.
    :param fct: function with the signature : ['function_name',a1,a2,a3,...,an] and function_name(x,a1,a2,a3,...,an).

    """
    n = len(fct[0])

    X0 = xini
    Y0 = fct[0][0](X0, *fct[0][1:n])

    X1 = X0 + dx
    Y1 = fct[0][0](X1, *fct[0][1:n])

    if Y0 > Y1:
        dx = -dx
        X0, X1 = X1, X0

    X2 = X1 + dx
    Y2 = fct[0][0](X2, *fct[0][1:n])

    while Y1 < Y2:
        X0 = X1
        X1 = X2
        X2 = X2 + dx
        Y0 = Y1
        Y1 = Y2
        Y2 = fct[0][0](X2, *fct[0][1:n])

    X = np.array([X0, X1, X2])
    Y = np.array([Y0, Y1, Y2])

    A = vander3(X)  # [X**2, X, numpy.ones(3)]
    C = trinome(A, Y)

    xres = -C[1] / (2.0 * C[0])
    yres = fct[0][0](xres, *fct[0][1:n])

    rc = 1

    return (xres, yres, rc)


def look_ahead_4_zero(xini, dx, *fct):
    """Look ahead for a zero of the function.

    :param xini: initial value of the variable.
    :param dx: fixed search step.
    :param fct: function with the signature : ['function_name',a1,a2,a3,...,an] and function_name(x,a1,a2,a3,...,an).

    """
    n = len(fct[0])

    X0 = xini
    Y0 = fct[0][0](X0, *fct[0][1:n])

    X1 = X0 + dx
    Y1 = fct[0][0](X1, *fct[0][1:n])

    while Y0 * Y1 > 0:
        X0 = X1
        Y0 = Y1
        X1 = X1 + dx
        Y1 = fct[0][0](X1, *fct[0][1:n])

    xres = X0 + (X1 - X0) * (-Y0) / (Y1 - Y0)
    yres = 0

    rc = 1

    return (xres, yres, rc)


def newton_solve(
    res_func,
    x0,
    dres_dy=None,
    args=(),
    res_max=1e-12,
    relax=0.995,
    max_iter=50,
    full_output=False,
):
    if res_max <= 0:
        msg = f"res_max too small ({res_max:g} <= 0)"
        raise ValueError(msg)
    if max_iter < 1:
        msg = "max_iter must be greater than 0"
        raise ValueError(msg)
    if dres_dy is None:
        res_func, dres_dy = get_jac_func(res_func, args)

    msg = ""
    ier = 1
    k = 0
    y_curr = np.atleast_1d(1.0 * x0)
    myargs = (y_curr, *args)
    curr_res = res_func(*myargs)
    n0 = norm(curr_res)
    if n0 == 0.0:
        return y_curr, n0, 1
    if not np.any(dres_dy(*myargs)):
        msg = "jacobian was zero."
        if marilib.is_driven_by_gems:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        if full_output:
            infodict = {"stop_crit": n0, "niter": 1}
            return y_curr, infodict, ier, msg
        return y_curr

    stop_crit = norm(curr_res) / n0
    while k < max_iter and stop_crit > res_max:
        drdy = dres_dy(*myargs)
        if marilib.is_using_autograd:
            from autograd.np.numpy_boxes import ArrayBox

            if isinstance(drdy, ArrayBox):
                drdy = drdy._value
            if isinstance(curr_res, ArrayBox):
                curr_res = curr_res._value
        newt_step = solve(
            np.atleast_2d(drdy).astype("float64"), curr_res.astype("float64")
        )
        y_curr -= relax * newt_step
        myargs = (y_curr, *args)
        curr_res = res_func(*myargs)
        stop_crit = norm(curr_res) / n0
        k += 1
    if k == max_iter and stop_crit > res_max:
        ier = 0
        msg = (
            "Failed to converge Newton solver within " + str(max_iter) + " iterations "
        )
        if marilib.is_driven_by_gems:
            raise LinAlgError(msg)

    if full_output:
        infodict = {"stop_crit": stop_crit, "niter": k}
        return y_curr, infodict, ier, msg
    return y_curr


def norm(x):
    return np.dot(x, x.T) ** 0.5


def approx_jac(res, y, args=(), step=1e-7):
    if not hasattr(y, "__len__"):
        y = np.array([y])
        n = 1
    else:
        n = len(y)
    jac = []
    myargs = (y, *args)
    res_ref = res(*myargs)
    pert = np.zeros_like(y)
    for i in range(n):
        yi = y[i]
        if hasattr(yi, "_value"):
            yi = yi._value
        pert[i] = step * 10 ** int(np.log10(abs(yi)))
        y += pert
        myargs_loc = (y, *args)
        res_pert = res(*myargs_loc)
        jac.append((res_pert - res_ref) / pert[i])
        y -= pert
        pert[i] = 0.0
    if len(jac) == 1:
        return jac[0]
    return np.concatenate(jac)


def get_jac_func(res, args=(), step=1e-6):
    if marilib.is_using_autograd:
        from autograd import jacobian
        from autograd.np.numpy_boxes import ArrayBox

        jac = jacobian(res, argnum=0)

        def j_func(*args):
            val = jac(*args)
            if isinstance(val, np.ndarray):
                return val
            if isinstance(val, ArrayBox):
                return val._value
            return None

        def my_res(*args):
            val = res(*args)
            if isinstance(val, np.ndarray):
                return val
            if isinstance(val, ArrayBox):
                return val._value
            return None

        return my_res, j_func

    def apprx(y, *args):
        return approx_jac(res, y, args, step)

    return res, apprx
