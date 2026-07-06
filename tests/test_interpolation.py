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

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from noads.core.models.interpolation import InterpolatedUnivariateSpline
from noads.core.models.interpolation import interpolate_data


@pytest.fixture
def x_data():
    return np.array([0.0, 1.0, 2.5, 4.0, 6.0])


@pytest.fixture
def y_data():
    return np.array([1.0, 3.0, 2.0, 5.0, 4.5])


def test_interpolate_data_linear(x_data, y_data):
    x = np.array([-1.0, 0.0, 0.5, 1.75, 3.0, 6.0, 8.0])
    result = interpolate_data(x, x_data, y_data)
    expected = np.interp(x, x_data, y_data)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_interpolate_data_cubic(x_data, y_data):
    # The spline passes through the knots.
    np.testing.assert_allclose(
        interpolate_data(x_data, x_data, y_data, cubic=True), y_data, atol=1e-8
    )
    # Interior values match scipy's not-a-knot cubic spline.
    x = np.linspace(0.0, 6.0, 25)
    result = interpolate_data(x, x_data, y_data, cubic=True)
    expected = CubicSpline(x_data, y_data, bc_type="not-a-knot")(x)
    np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_spline_orders(k, x_data, y_data):
    spline = InterpolatedUnivariateSpline(x_data, y_data, k=k)
    np.testing.assert_allclose(spline(x_data), y_data, atol=1e-8)


def test_spline_natural_endpoints(x_data, y_data):
    spline = InterpolatedUnivariateSpline(x_data, y_data, k=3, endpoints="natural")
    assert float(spline.derivative(x_data[0], n=2)) == pytest.approx(0.0, abs=1e-8)
    # The right-end condition of this implementation equates the last two
    # second-derivative coefficients, so the last piece has no cubic term.
    assert float(spline.derivative(x_data[-1], n=3)) == pytest.approx(0.0, abs=1e-8)
    # An unknown endpoint condition falls back to natural.
    fallback = InterpolatedUnivariateSpline(x_data, y_data, k=3, endpoints="unknown")
    x = np.linspace(0.0, 6.0, 25)
    np.testing.assert_allclose(fallback(x), spline(x), atol=1e-10)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_spline_derivative(k, x_data, y_data):
    spline = InterpolatedUnivariateSpline(x_data, y_data, k=k)
    # Sample away from the knots, where the k=1 derivative is discontinuous.
    x = np.array([0.6, 1.8, 3.1, 4.9])
    np.testing.assert_allclose(spline.derivative(x, n=0), spline(x), atol=1e-12)

    step = 1e-6
    finite_differences = (spline(x + step) - spline(x - step)) / (2 * step)
    np.testing.assert_allclose(spline.derivative(x, n=1), finite_differences, atol=1e-4)

    if k >= 2:
        assert np.all(np.isfinite(spline.derivative(x, n=2)))
    if k == 3:
        assert np.all(np.isfinite(spline.derivative(x, n=3)))

    with pytest.raises(AssertionError):
        spline.derivative(x, n=k + 1)


def test_spline_antiderivative(x_data, y_data):
    spline = InterpolatedUnivariateSpline(x_data, y_data, k=1)
    # For a piecewise-linear function, the trapezoid rule is exact at the knots.
    expected = np.concatenate([
        [0.0],
        np.cumsum(np.diff(x_data) * (y_data[1:] + y_data[:-1]) / 2.0),
    ])
    np.testing.assert_allclose(spline.antiderivative(x_data), expected, atol=1e-10)

    # Only the linear spline has an antiderivative implementation.
    for k in (2, 3):
        spline = InterpolatedUnivariateSpline(x_data, y_data, k=k)
        assert spline.antiderivative(x_data) is None


def test_spline_invalid_inputs(x_data, y_data):
    with pytest.raises(AssertionError):
        InterpolatedUnivariateSpline(x_data, y_data, k=4)
    with pytest.raises(AssertionError):
        InterpolatedUnivariateSpline(x_data, y_data[:-1])
    with pytest.raises(AssertionError):
        InterpolatedUnivariateSpline(x_data[:1], y_data[:1], k=1)
    with pytest.raises(AssertionError):
        InterpolatedUnivariateSpline(x_data[:2], y_data[:2], k=2)
    with pytest.raises(AssertionError):
        InterpolatedUnivariateSpline(x_data[:3], y_data[:3], k=3)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_spline_tree_flatten_roundtrip(k, x_data, y_data):
    spline = InterpolatedUnivariateSpline(x_data, y_data, k=k)
    children, aux_data = spline.tree_flatten()
    rebuilt = InterpolatedUnivariateSpline.tree_unflatten(aux_data, children)
    x = np.linspace(0.0, 6.0, 13)
    np.testing.assert_allclose(rebuilt(x), spline(x), atol=1e-12)
