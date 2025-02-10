"""Air traffic demand estimation from AR6 scenarios."""
from jax import vmap
from matplotlib.pyplot import subplots
from numpy import linspace
from numpy import ones

from application.background_scenario_data import get_ar6_data
from core.models.interpolation import interpolate_data
from core.models.traffic import AirTraffic

# %%
# # AR6 scenario data
# Let's start by getting the AR6 scenario data and taking a look at it.
scenario_data, years_data = get_ar6_data(start_year=2000, end_year=2100, plot_data=True)

# %%
# Not all data is required for air traffic estimation, only Population and GDP per
# capita. Now we interpolate, for each scenario, to get values yearly.
years = linspace(2000, 2100, 101)
population = {
    scenario: interpolate_data(years, years_data, scenario_data["population"][scenario])
    for scenario in scenario_data["population"].keys()
}
gdp_per_capita = {
    scenario: interpolate_data(
        years, years_data, scenario_data["gdp_per_capita"][scenario]
    )
    for scenario in scenario_data["gdp_per_capita"].keys()
}
# %%
# Also, we consider COVID effects by delaying trend traffic of a fixed amount of years,
# here we must also include the assumption of end of COVID related traffic constraints.
year_covid_end = 2024
gdp_per_capita_end_covid = {
    scenario: interpolate_data(
        year_covid_end, years_data, scenario_data["gdp_per_capita"][scenario]
    )
    for scenario in scenario_data["gdp_per_capita"].keys()
}

# %%
# # Model vectorization
# When a model must be evaluated yearly, rather than once, we must vectorize its
# function to account for vectorized inputs. When running a TemporalScenario this is
# done automatically, but to here we must to it manually.
model = AirTraffic()
model.discipline.jax_out_func = vmap(model.discipline.jax_out_func)
outputs = {}
fig, axes = subplots(1, 3, figsize=(7, 4), layout="constrained")
for scenario in population.keys():
    vones = ones(years.shape)
    model_input = {
        "year": years,
        "gdp_per_capita": gdp_per_capita[scenario],
        "population": population[scenario],
        "gdp_per_capita_covid_end": gdp_per_capita_end_covid[scenario] * vones,
        "load_factor_end_year": 92.0 * vones,
        "end_year": 2075.0 * vones,
    }
    model_output = model.discipline.execute(model_input)





