from gemseo import MDODiscipline
from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_scenario
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from jax import config

from jax.numpy import array
from jax.numpy import divide
from jax.numpy import exp
from jax.numpy.linalg import norm
from matplotlib.pyplot import savefig
from matplotlib.pyplot import close
from matplotlib.pyplot import subplots
from numpy import where
from pandas import read_csv

from pandas import read_excel

from gemseo_jax.auto_jax_discipline import AutoJAXDiscipline
from gemseo_jax.jax_chain import JAXChain

from numpy import diff
from numpy import max
from numpy import min
from numpy import array as np_array
from numpy import isnan
from numpy import append
from numpy import linspace

# WLD calibration on departures per capita (y) from GDP per capita (x)
from core.models.traffic import generalised_logistic


def error_measure(y, y_data):
    mse = norm(y - y_data) / norm(y_data)
    return mse


def get_raw_data(region):
    # Socio-economic indicators
    gdp_data = read_excel(
        "./data/GDP.xls",
        decimal=",",
        skiprows=[0, 1, 2],
        index_col=1,
    ).transpose()
    pop_data = read_excel(
        "./data/Population.xls",
        decimal=",",
        skiprows=[0, 1, 2],
        index_col=1,
    ).transpose()

    # Air traffic indicators
    dep_data = read_excel(
        "./data/AirTraffic-carrier-departures.xls",
        decimal=",",
        skiprows=[0, 1, 2],
        index_col=1,
    ).transpose()

    gdp = gdp_data[region][3:-1].to_numpy(dtype=float)
    pop = pop_data[region][3:-1].to_numpy(dtype=float)
    dep = dep_data[region][3:-1].to_numpy(dtype=float)
    years = dep_data.transpose().columns.values.tolist()[3:-1]

    dep_pc = dep / pop
    gdp_pc = gdp / pop

    return [int(y) for y in years], gdp_pc, dep_pc


def filter_nans(data_iterable, exclude_covid=True):
    filtered_data_iterable = []
    last_index = 2 if exclude_covid else 0  # exclude 2020, 2021, 2022
    for data in data_iterable:
        filtered_data_iterable.append(
            np_array([
                data[idx] for idx in range(len(data) - last_index)
                if not any([isnan(np_array(d))[idx] for d in data_iterable])
            ])
        )
    return filtered_data_iterable


def plot_calibration_result(
    opt_result, region, years_data, x_data, y_data, years_raw, x_raw, y_raw
):
    x_ordered = linspace(min(x_data), max(x_data), 100)

    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic)
    opt_result.update({"x": x_data})
    results = model.execute(opt_result)

    fig, ax = subplots()
    ax.plot(
        y_data,
        results["y"],
        "o",
        label="modeled",
    )
    ax.plot(
        y_data,
        y_data,
        label="true",
    )
    ax.set_ylabel("y data")
    ax.set_xlabel("y modeled")
    ax.legend(loc="upper left")
    fig.show()
    close(fig)

    opt_result.update({"x": x_ordered})
    results = model.execute(opt_result)
    fig, ax = subplots()
    ax.plot(
        x_ordered,
        results["y"],
        "-",
        label="modeled",
    )
    ax.plot(
        x_raw,
        y_raw,
        "x",
        label="all data",
    )
    ax.plot(
        x_data,
        y_data,
        "o",
        label="calibration data",
    )
    ax.set_ylabel("RPK per capita")
    ax.set_xlabel("GDP per capita [current US$/hab.]")
    ax.legend(loc="upper left")
    fig.show()
    close(fig)

    opt_result.update({"x": x_raw})
    results = model.execute(opt_result)
    fig, ax = subplots()
    ax.plot(
        years_raw,
        results["y"],
        "-",
        label="modeled",
    )
    ax.plot(
        years_raw,
        y_raw,
        "x",
        label="all data",
    )
    ax.plot(
        years_data,
        y_data,
        "o",
        label="calibration data",
    )
    ax.set_ylabel("departures per capita")
    ax.set_xlabel("year")
    ax.legend(loc="upper left")
    fig.show()
    close(fig)


def run_region_calibration(region, plot_calibration=True):
    years_raw, x_raw, y_raw = get_raw_data(region)
    filtered = filter_nans([years_raw, x_raw, y_raw])
    years_data = filtered[0]
    x_data = filtered[1]
    y_data = filtered[2]

    # Compute some stuff from data
    x_max = max(x_data)

    y_max = max(y_data)
    y_min = min(y_data)

    dy = diff(y_data)
    dy_max = max(dy)

    # Disciplines and Chain with gemseo-jax
    model = AutoJAXDiscipline(generalised_logistic)
    measure = AutoJAXDiscipline(error_measure, static_args={"y_data": y_data})

    jax_chain = JAXChain([model, measure])

    jax_chain.add_differentiated_outputs(["mse"])
    jax_chain.compile_jit(False)

    # Create the design space:
    design_space = create_design_space()
    design_space.add_variable(
        "left_asymptote", l_b=0.0, u_b=y_max, value=np_array(y_min)
    )
    design_space.add_variable(
        "capacity", l_b=y_min, u_b=10 * y_max, value=np_array(y_max)
    )
    design_space.add_variable(
        "growth_rate", l_b=0.2 * dy_max, u_b=1.8 * dy_max, value=np_array(dy_max)
    )
    design_space.add_variable(
        "x_inflection", l_b=0.0, u_b=3.0 * x_max, value=np_array(0.5 * x_max)
    )
    design_space.add_variable(
        "logistic_nu", l_b=0.1, u_b=10, value=np_array(1.0)
    )
    design_space.add_variable(
        "asymptote_coeff", l_b=0.5, u_b=2.0, value=np_array(1.0)
    )
    variable_names = design_space.variable_names

    # Create the MDO scenario with an MDF formulation:
    scenario = create_scenario(
        jax_chain,
        formulation="MDF",
        inner_mda_name="MDAGaussSeidel",
        objective_name="mse",
        design_space=design_space,
        grammar_type=MDODiscipline.GrammarType.SIMPLER,
    )

    # Embed scenario in an adapter
    scenario.default_inputs = {"max_iter": 250, "algo": "L-BFGS-B"}
    adapter = MDOScenarioAdapter(
        scenario,
        variable_names,
        ["mse", "y"],
        set_x0_before_opt=True,
    )

    # Make DOE scenario from adapter
    scenario_doe = create_scenario(
        adapter,
        formulation="DisciplinaryOpt",
        objective_name="mse",
        design_space=design_space,
        scenario_type="DOE",
        grammar_type=MDODiscipline.GrammarType.SIMPLER,
    )
    scenario_doe.execute({"n_samples": 15, "algo": "OT_OPT_LHS"})
    # print(output_data)
    cache_entries = list(jax_chain.cache)
    mse_entries = [entry.outputs["mse"] for entry in cache_entries]
    idx_opt = mse_entries.index(min(mse_entries))
    best_fit = cache_entries[idx_opt].inputs
    if plot_calibration:
        print(region, "best fit:", best_fit)
        scenario_doe.post_process(
            "BasicHistory", variable_names=["mse"], save=False, show=True
        )
        scenario.post_process(
            "ScatterPlotMatrix",
            variable_names=variable_names,
            save=False,
            show=True,
        )
        plot_calibration_result(
            best_fit, region, years_data, x_data, y_data, years_raw, x_raw, y_raw
        )

    return (
            best_fit, region, years_data, x_data, y_data, years_raw, x_raw, y_raw
        )
