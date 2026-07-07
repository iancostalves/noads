# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Time-dependent controls and the first-order delay
=================================================
"""

from matplotlib.pyplot import show
from matplotlib.pyplot import subplots
from numpy import array
from numpy import interp
from numpy import linspace
from scipy.integrate import solve_ivp

# %%
# The first-order delay
# ---------------------
# Every time-dependent optimization variable (a *control*) enters the simulation
# through a first-order delay, which turns a possibly abrupt decision into a gradual
# response and improves the stability and dimensionality of the optimization. The
# delayed output ``o(t)`` follows the input signal ``i(t)`` with a time constant
# ``tau``:
#
# .. math::
#     \frac{d}{dt}o(t) = \frac{i(t) - o(t)}{\tau}
#
# The helper below integrates that equation (mirroring
# ``noads.core.scenarios.temporalscenario.delay1_rhs``), with the input signal given
# as ``(times, values)`` knots interpolated linearly, and an initial output of zero.


def first_order_delay(times, values, tau, t_eval):
    """Response of the first-order delay to a piecewise-linear input signal."""
    times, values = array(times, dtype=float), array(values, dtype=float)

    def rhs(t, o):
        return (interp(t, times, values) - o) / tau

    solution = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), [0.0], t_eval=t_eval, max_step=0.25
    )
    return solution.y[0]


# %%
# Response to a step, a ramped step, and a pulse, for two time constants. For a ramped
# step of ramp duration ``2*tau`` the output reaches 95 % of its target after about
# ``4*tau``, which the model interprets as one fleet-replacement lifetime.

t = linspace(0, 60, 601)
inputs = {
    "step": ([0, 10, 10.001, 60], [0, 0, 1, 1]),
    "ramped step (2τ)": ([0, 10, 30, 60], [0, 0, 1, 1]),
    "pulse": ([0, 10, 30, 50, 60], [0, 0, 1, 0, 0]),
}

fig1, axes1 = subplots(1, 2, figsize=(9, 4), layout="constrained", sharey=True)
for ax, tau in zip(axes1, (5.0, 10.0)):
    for label, (times, values) in inputs.items():
        ax.plot(t, interp(t, times, values), ":", linewidth=1)
        ax.plot(t, first_order_delay(times, values, tau, t), label=label, linewidth=2)
    ax.axhline(0.95, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"τ = {tau:.0f} yr")
    ax.set_xlabel("Time [yr]")
axes1[0].set_ylabel("Delay output")
axes1[0].legend(loc="lower right", fontsize="small")
fig1.suptitle("First-order delay response (dotted: input, solid: delayed output)")
show()

# %%
# Aircraft market share
# ---------------------
# The market penetration of a new aircraft is a *ramped pulse*: it stays at zero until
# the Entry-Into-Service (EIS), ramps up to a maximum share, holds for the aircraft
# lifetime, then ramps back down. Four of these numbers -- EIS, maximum share, lifetime
# and ramp-down duration -- are optimization variables (this mirrors the trapezoid
# built in ``AircraftDesign._control_share``). The delay then smooths the trapezoid
# into the actual deployment curve. EIS and maximum share shift when and how strongly
# the aircraft is deployed.

years = linspace(2025, 2075, 501)


def ramped_pulse(eis, max_share, ramp_up=5.0, lifetime=25.0, ramp_down=5.0):
    times = [eis, eis + ramp_up, eis + lifetime, eis + lifetime + ramp_down]
    return times, [0.0, max_share, max_share, 0.0]


fig2, ax2 = subplots(figsize=(7, 4), layout="constrained")
for eis, smax, color in [(2035, 0.8, "C0"), (2045, 0.5, "C1")]:
    times, values = ramped_pulse(eis, smax)
    ax2.plot(years, interp(years, times, values), ":", color=color, linewidth=1)
    ax2.plot(
        years,
        first_order_delay(times, values, tau=12.5, t_eval=years),
        color=color,
        linewidth=2,
        label=f"EIS {eis}, S_max {smax:.0%}",
    )
ax2.set_xlabel("Year")
ax2.set_ylabel("Market share")
ax2.set_title("Aircraft market share (dotted: ramped-pulse input, solid: delayed)")
ax2.legend()
show()

# %%
# Energy pathway shares
# ---------------------
# The share of each production pathway is a control discretised on coarse knots (every
# 2.5 years), whose values at each knot are optimization variables constrained to sum
# to one across the pathways of an energy. Here two competing pathways trade places
# over time; the delay again turns the piecewise-linear decisions into a smooth
# transition.

knots = linspace(2025, 2075, 21)
share_a = array([
    0.9,
    0.9,
    0.85,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.32,
    0.25,
    0.2,
    0.16,
    0.13,
    0.1,
    0.08,
    0.06,
    0.05,
    0.04,
    0.03,
    0.02,
    0.02,
])  # noqa: E501

fig3, ax3 = subplots(figsize=(7, 4), layout="constrained")
ax3.plot(
    knots, share_a, ":o", color="C2", markersize=3, linewidth=1, label="input knots"
)
ax3.plot(
    years,
    first_order_delay(knots, share_a, tau=5.0, t_eval=years),
    color="C2",
    linewidth=2,
    label="pathway A (delayed)",
)
ax3.plot(
    years,
    1.0 - first_order_delay(knots, share_a, tau=5.0, t_eval=years),
    color="C3",
    linewidth=2,
    label="pathway B = 1 - A",
)
ax3.set_xlabel("Year")
ax3.set_ylabel("Production share")
ax3.set_title("Energy pathway shares")
ax3.legend(fontsize="small")
show()

# %%
# Demand avoidance
# ----------------
# In the low-demand formulation, each market has a supply-shift ratio control: the
# fraction of trend supply that is avoided (0: all trend traffic is served, 1: fully
# grounded). Its knot values are optimization variables; the delayed control gives the
# realised avoidance, and ``1 - SR`` the fraction of trend traffic that is still flown.

sr_knots_years = linspace(2025, 2075, 11)
sr_values = array([0.0, 0.05, 0.12, 0.2, 0.28, 0.35, 0.4, 0.44, 0.47, 0.49, 0.5])

fig4, ax4 = subplots(figsize=(7, 4), layout="constrained")
sr = first_order_delay(sr_knots_years, sr_values, tau=10.0, t_eval=years)
ax4.plot(
    sr_knots_years,
    sr_values,
    ":o",
    color="C4",
    markersize=3,
    linewidth=1,
    label="supply-shift ratio (input knots)",
)
ax4.plot(years, sr, color="C4", linewidth=2, label="avoided share (delayed)")
ax4.plot(years, 1.0 - sr, color="C5", linewidth=2, label="served share = 1 - SR")
ax4.set_xlabel("Year")
ax4.set_ylabel("Fraction of trend supply")
ax4.set_title("Demand avoidance")
ax4.legend(fontsize="small")
show()
