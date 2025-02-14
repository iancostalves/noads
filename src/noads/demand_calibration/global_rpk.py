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
from numpy import max
from numpy import mean
from numpy import min
from numpy import sqrt

from noads.core.models.traffic import generalised_logistic
from noads.demand_calibration.calibration_utils import error_measure
from noads.demand_calibration.calibration_utils import filter_nans
from noads.demand_calibration.calibration_utils import get_rpk_data


def plot_calibration_result(
    opt_result, years_data, x_data, y_data, pop_data, years_raw, x_raw, y_raw, pop_raw
):
    x_ordered = linspace(min(x_data), 2.5 * max(x_data), 100)

    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic)
    opt_result.update({"x": x_data})
    results = model.execute(opt_result)

    fig, axes = subplots(1, 3, layout="constrained")
    fig.set_size_inches(9, 3)
    axes[0].plot(
        y_data * 1e6,
        results["y"] * 1e6,
        ".",
        label="modeled",
    )
    axes[0].plot(
        y_data * 1e6,
        y_data * 1e6,
        label="true",
    )
    axes[0].set_title("Data vs model")
    axes[0].set_ylabel("Model")
    axes[0].set_xlabel("Data")
    axes[0].legend(loc="upper left")
    sqrt((1 / y_data.size) * sum(((y_data - results["y"]) * 1e6) ** 2))
    1 - sum((y_data - results["y"]) ** 2) / (sum((y_data - mean(y_data)) ** 2))
    1 - sum(((y_data - results["y"]) * pop_data) ** 2) / (
        sum((y_data * pop_data - mean(y_data * pop_data)) ** 2)
    )
    # axes[0].text(1, 1, f"RMSE:{rmse:.3}\nR2:{r2:.3}")

    opt_result.update({"x": x_ordered})
    results = model.execute(opt_result)
    axes[1].plot(
        x_ordered,
        results["y"] * 1e6,
        "-",
        label="modeled",
    )
    axes[1].plot(
        x_raw,
        y_raw * 1e6,
        "x",
        label="all data",
    )
    axes[1].plot(
        x_data,
        y_data * 1e6,
        ".",
        label="calibration data",
    )
    axes[1].set_title("Logistic curve")
    axes[1].set_ylabel("pax km per capita")
    axes[1].set_xlabel("GDP per capita [current US$/hab.]")

    opt_result.update({"x": x_raw})
    results = model.execute(opt_result)
    axes[2].plot(
        years_raw,
        results["y"] * pop_raw * 1e6,
        "-",
        label="modeled",
    )
    axes[2].plot(
        years_raw,
        y_raw * pop_raw * 1e6,
        "x",
        label="all data",
    )
    axes[2].plot(
        years_data,
        y_data * pop_data * 1e6,
        ".",
        label="calibration data",
    )
    axes[2].set_title("RPK time series")
    axes[2].set_ylabel("pax km")
    axes[2].set_xlabel("Year")
    axes[2].legend(loc="upper left")
    fig.show()
    # close(fig)


def run_global_rpk_calibration(plot_calibration=True):
    years_raw, x_raw, y_raw, pop_raw = get_rpk_data(1970)
    filtered = filter_nans([years_raw, x_raw, y_raw, pop_raw])
    years_data = filtered[0]
    x_data = filtered[1]
    y_data = filtered[2]
    pop_data = filtered[3]

    # Compute some stuff from data
    x_max = max(x_data)

    y_max = max(y_data)
    y_min = min(y_data)

    dy = diff(y_data)
    dy_max = max(dy)

    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic, static_args={"x": x_data})
    measure = AutoJAXDiscipline(
        error_measure, static_args={"y_data": y_data, "weights": 1.0}
    )

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
        "x_lag", lower_bound=0.0, upper_bound=3.0 * x_max, value=np_array(0.1 * x_max)
    )
    design_space.add_variable(
        "logistic_nu", lower_bound=0.1, upper_bound=10, value=np_array(1.0)
    )
    design_space.add_variable(
        "asymptote_coeff", lower_bound=0.5, upper_bound=2.0, value=np_array(1.0)
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
        n_start=10,
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
            best_fit,
            years_data,
            x_data,
            y_data,
            pop_data,
            years_raw,
            x_raw,
            y_raw,
            pop_raw,
        )

    return (best_fit, years_data, x_data, y_data, years_raw, x_raw, y_raw)
