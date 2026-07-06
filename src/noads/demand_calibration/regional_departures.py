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
"""Calibration routine for Regional Carrier Departures."""

from gemseo import create_design_space
from gemseo import create_scenario
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo_jax.auto_jax_discipline import AutoJAXDiscipline
from gemseo_jax.jax_chain import JAXChain
from matplotlib.pyplot import show
from matplotlib.pyplot import subplots
from matplotlib.pyplot import yscale
from numpy import array as np_array
from numpy import diff
from numpy import linspace
from numpy import max as np_max
from numpy import min as np_min

from noads.core.models.traffic import generalised_logistic
from noads.demand_calibration.calibration_utils import error_measure
from noads.demand_calibration.calibration_utils import filter_nans
from noads.demand_calibration.calibration_utils import get_departures_data


def plot_calibration_result(
    opt_result, years_data, x_data, y_data, years_raw, x_raw, y_raw
):
    """Plot calibration of Regional Carrier Departures."""
    x_ordered = linspace(np_min(x_data), np_max(x_data), 100)

    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic)
    opt_result.update({"x": x_data})
    results = model.execute(opt_result)

    fig1, ax1 = subplots()
    ax1.plot(
        y_data,
        results["y"],
        "o",
        label="modeled",
    )
    ax1.plot(
        y_data,
        y_data,
        label="true",
    )
    ax1.set_ylabel("y data")
    ax1.set_xlabel("y modeled")
    ax1.legend(loc="upper left")
    fig1.show()
    # close(fig1)

    opt_result.update({"x": x_ordered})
    results = model.execute(opt_result)
    fig2, ax2 = subplots()
    ax2.plot(
        x_ordered,
        results["y"],
        "-",
        label="modeled",
    )
    ax2.plot(
        x_raw,
        y_raw,
        "x",
        label="all data",
    )
    ax2.plot(
        x_data,
        y_data,
        "o",
        label="calibration data",
    )
    ax2.set_ylabel("RPK per capita")
    ax2.set_xlabel("GDP per capita [current US$/hab.]")
    ax2.legend(loc="upper left")
    fig2.show()
    # close(fig_paper)

    opt_result.update({"x": x_raw})
    results = model.execute(opt_result)
    fig3, ax3 = subplots()
    ax3.plot(
        years_raw,
        results["y"],
        "-",
        label="modeled",
    )
    ax3.plot(
        years_raw,
        y_raw,
        "x",
        label="all data",
    )
    ax3.plot(
        years_data,
        y_data,
        "o",
        label="calibration data",
    )
    ax3.set_ylabel("departures per capita")
    ax3.set_xlabel("year")
    ax3.legend(loc="upper left")
    fig3.show()
    # close(fig3)


def run_region_calibration(region, plot_calibration=True):
    """Run calibration of Regional Carrier Departures."""
    years_raw, x_raw, y_raw = get_departures_data(region)
    filtered = filter_nans([years_raw, x_raw, y_raw])
    years_data = filtered[0]
    x_data = filtered[1]
    y_data = filtered[2]

    # Compute some stuff from data
    x_max = np_max(x_data)

    y_max = np_max(y_data)
    y_min = np_min(y_data)

    dy = diff(y_data)
    dy_max = np_max(dy)

    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic, static_args={"x": x_data})
    measure = AutoJAXDiscipline(error_measure, static_args={"y_data": y_data})

    jax_chain = JAXChain([model, measure])

    jax_chain.add_differentiated_outputs(["mse"])
    jax_chain.compile_jit(False)

    # Create the design space:
    design_space = create_design_space()
    design_space.add_variable(
        "left_asymptote", lower_bound=0.0, upper_bound=y_max, value=np_array(y_min)
    )
    design_space.add_variable(
        "capacity", lower_bound=y_min, upper_bound=10 * y_max, value=np_array(y_max)
    )
    design_space.add_variable(
        "growth_rate",
        lower_bound=0.2 * dy_max,
        upper_bound=1.8 * dy_max,
        value=np_array(dy_max),
    )
    design_space.add_variable(
        "x_lag",
        lower_bound=0.0,
        # The inflection income must lie within the observed data range: allowing
        # it beyond x_max lets high-income regions (e.g. USA) settle in a
        # degenerate, never-saturating basin.
        upper_bound=x_max,
        value=np_array(0.5 * x_max),
    )
    design_space.add_variable(
        "logistic_nu", lower_bound=0.1, upper_bound=10, value=np_array(1.0)
    )
    design_space.add_variable(
        "asymptote_coeff", lower_bound=0.1, upper_bound=10.0, value=np_array(1.0)
    )

    # Create the MDO scenario with an MDF formulation:
    scenario = create_scenario(
        jax_chain,
        "mse",
        design_space,
        formulation_name="DisciplinaryOpt",
    )

    settings = MultiStart_Settings(
        max_iter=5000,
        opt_algo_name="L-BFGS-B",
        doe_algo_name="OT_OPT_LHS",
        n_start=20,
        # multistart_file_path="multistart.hdf5",
    )
    scenario.execute(settings)

    best_fit = scenario.optimization_result.x_opt_as_dict
    if plot_calibration:
        scenario.post_process(
            post_name="BasicHistory", variable_names=["mse"], save=False, show=False
        )
        yscale("log")
        show()

        plot_calibration_result(
            best_fit, years_data, x_data, y_data, years_raw, x_raw, y_raw
        )

    return best_fit, years_data, x_data, y_data, years_raw, x_raw, y_raw
