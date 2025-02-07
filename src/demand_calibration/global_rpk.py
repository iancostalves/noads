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
from numpy import arange
from numpy import flip
from numpy import mean
from numpy import sqrt
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


def error_measure(y, y_data, weights=1.0):
    mse = norm(weights * (y - y_data)) / norm(weights * y_data)
    return mse


def get_raw_data(region, y_start):
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
    rpk_data = read_excel(
        "./data/Traffic_1929_to_2021.xlsx",
        decimal=",",
    )

    gdp_all = gdp_data[region][3:-1].to_numpy(dtype=float)
    pop_all = pop_data[region][3:-1].to_numpy(dtype=float)
    gdp_pop_years = np_array([int(y) for y in gdp_data.transpose().columns[3:-1]])

    rpk_all = flip(rpk_data["RPKs (mils)"].to_numpy(dtype=float))
    rpk_years = flip(rpk_data["Year"].to_numpy(dtype=float))

    # y_start = max([min(gdp_pop_years), min(rpk_years)])
    y_end = min([max(gdp_pop_years), max(rpk_years)])
    years = arange(y_start, y_end + 1)

    gdp = np_array([val for i, val in enumerate(gdp_all) if gdp_pop_years[i] in years])
    pop = np_array([val for i, val in enumerate(pop_all) if gdp_pop_years[i] in years])
    rpk = np_array([val for i, val in enumerate(rpk_all) if rpk_years[i] in years])

    rpk_pc = rpk / pop
    gdp_pc = gdp / pop

    return years, gdp_pc, rpk_pc, pop


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
    opt_result, region, years_data, x_data, y_data, pop_data, years_raw, x_raw, y_raw, pop_raw
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
    rmse = sqrt((1 / y_data.size) * sum(((y_data - results["y"]) * 1e6) ** 2))
    print(f"RMSE:{rmse}")
    r2_pc = 1 - sum((y_data - results["y"]) ** 2) / (sum((y_data - mean(y_data)) ** 2))
    print(f"R2pc:{r2_pc}")
    r2 = 1 - sum(((y_data - results["y"]) * pop_data) ** 2) / (
        sum((y_data * pop_data - mean(y_data * pop_data)) ** 2)
    )
    print(f"R2:{r2}")
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
    savefig(f"./calibration_{region}.pdf")
    close(fig)


def main():
    configure_logger()
    for region in ["WLD"]:
        years_raw, x_raw, y_raw, pop_raw = get_raw_data(region, 1970)
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
        model = AutoJAXDiscipline(
            function=generalised_logistic,
            static_args={"x": x_data},
            grammar_type=MDODiscipline.GrammarType.SIMPLER,
        )
        measure = AutoJAXDiscipline(
            function=error_measure,
            static_args={"y_data": y_data, "weights": 1.0},
            grammar_type=MDODiscipline.GrammarType.SIMPLER,
        )

        jax_chain = JAXChain(
            [model, measure], cache_type=MDODiscipline.CacheType.MEMORY_FULL
        )

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
            "exp_coeff", l_b=0.0, u_b=5.0, value=np_array(0.1)
        )
        design_space.add_variable(
            "x_inflection", l_b=0.0, u_b=3.0 * x_max, value=np_array(0.1 * x_max)
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
        scenario.default_inputs = {"max_iter": 200, "algo": "L-BFGS-B"}
        adapter = MDOScenarioAdapter(
            scenario,
            variable_names,
            ["mse", "y"],
            set_x0_before_opt=True,
            grammar_type=MDODiscipline.GrammarType.SIMPLER,
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
        output_data = scenario_doe.execute({"n_samples": 15, "algo": "OT_OPT_LHS"})
        print(output_data)

        # Post-process the DOE results
        # scenario_doe.post_process(
        #     "BasicHistory", variable_names=["mse"], save=False, show=True
        # )
        # scenario.post_process(
        #     "ScatterPlotMatrix",
        #     variable_names=variable_names,
        #     save=False,
        #     show=True,
        # )

        # Now we plot stuff for the best fit
        cache_entries = list(jax_chain.cache)
        mse_entries = [entry.outputs["mse"] for entry in cache_entries]
        idx_opt = mse_entries.index(min(mse_entries))
        best_fit = cache_entries[idx_opt].inputs
        print("Best fit:", best_fit)
        print(cache_entries[idx_opt].outputs)
        plot_calibration_result(
            best_fit, region, years_data, x_data, y_data, pop_data, years_raw, x_raw, y_raw, pop_raw
        )


if __name__ == "__main__":
    main()
    # plot_calibration_result(
    #     {'x_inflection': array([1923.88835173]), 'exp_coeff': array([0.55007268]),
    #         'asymptote_coeff': array([1.09240442]), 'logistic_nu': array([0.14623451]),
    #         'left_asymptote': array([0.00219871]), 'growth_rate': array([0.00028066]),
    #         'capacity': array([0.00795385])
    #     }
    # )
