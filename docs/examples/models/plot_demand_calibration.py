# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Calibration of air traffic demand models: global and regionalized."""

from gemseo import configure_logger
from gemseo_jax.auto_jax_discipline import AutoJAXDiscipline
from matplotlib.pyplot import subplots
from numpy import linspace

from noads.core.models.traffic import generalised_logistic
from noads.demand_calibration.global_rpk import run_global_rpk_calibration
from noads.demand_calibration.regional_departures import run_region_calibration

configure_logger()

# %%
# # World Bank Departures data
# At first we explore World Bank data on registered carrier departures. The intent is to
# fit a generalized logistic function to estimate traffic per capita from income per
# capita.
# The main idea of the calibration is to find the set of parameters that minimize the
# error between modeled history and history data. We use gradient-based algorithms to
# exploit the Automatic Differentiation (AD) provided by JAX. As these algorithms may be
# more prone to getting stuck on local minima (especially in calibration problems, which
# are often multi-modal), we use a Multi-Start strategy, i.e., run several optimizations
# with several starting points (chosen with a Design of Experiments algorithm).
# ## Globally aggregated
run_region_calibration("WLD", plot_calibration=True)

# %%
# ## Regional
countries = ["USA", "GBR", "EUU", "BOL", "BRA", "CHN", "IND"]
fig, ax = subplots(figsize=(7, 4), layout="constrained")
for country in countries:
    (best_fit, years_data, x_data, y_data, years_raw, x_raw, y_raw) = (
        run_region_calibration(country, plot_calibration=False)
    )
    x_ordered = linspace(min(x_data), max(x_data), 100)
    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic)
    best_fit.update({"x": x_ordered})
    results = model.execute(best_fit)
    lines = ax.plot(
        x_ordered,
        results["y"],
        "-",
        label=f"{country} - modelled",
    )
    ax.plot(
        x_data,
        y_data,
        ".",
        label=f"{country} - observed",
        color=lines[-1].get_color(),
    )
ax.set_ylabel("departures per capita")
ax.set_xlabel("GDP per capita\n[current US$/hab.]")
ax.legend(bbox_to_anchor=(1.1, 0.9))
ax.set_ylim(bottom=1e-4)
ax.set_yscale("log")
ax.set_xscale("log")
fig.show()

# %%
# # ICAO RPK data
# Now, we explore ICAO data on Revenue Passenger-Kilometers. The data here is only
# available on the global level.
run_global_rpk_calibration(plot_calibration=True)
# %%
# The parameters obtained in this calibration are set as default for the air traffic
# estimation based on socioeconomic drivers (population, income) from the AR6 scenarios.
