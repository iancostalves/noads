from jax.numpy.linalg import norm
from numpy import arange
from numpy import array as np_array
from numpy import flip
from numpy import isnan
from pandas import read_excel


def error_measure(y, y_data):
    mse = norm(y - y_data) / norm(y_data)
    return mse


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


def get_rpk_data(y_start):
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

    region = "WLD"
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


def get_departures_data(region):
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
