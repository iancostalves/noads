"""Calibration of air traffic demand models: global and regionalized."""
from gemseo import configure_logger

from demand_calibration.regional_departures import run_region_calibration

configure_logger()

run_region_calibration("WLD", plot_calibration=True)


# countries = [
#     "USA", "GBR", "EUU", "BOL", "BRA", "CHN", "IND"
# ]
#
# fig, ax = subplots(figsize=(7, 4), layout="constrained")
# for country in countries:
#     (
#         best_fit, region, years_data, x_data, y_data, years_raw, x_raw, y_raw
#     ) = run_region_calibration(country, plot_calibration=False)
#     x_ordered = linspace(min(x_data), max(x_data), 100)
#     # Disciplines and Chain with gemseo-jax
#     model = AutoJAXDiscipline(
#         function=generalised_logistic,
#         grammar_type=MDODiscipline.GrammarType.SIMPLER,
#         cache_type=MDODiscipline.CacheType.NONE,
#     )
#     best_fit.update({"x": x_ordered})
#     results = model.execute(best_fit)
#     lines = ax.plot(
#         x_ordered,
#         results["y"],
#         "-",
#         label=f"{country} - modelled",
#     )
#     ax.plot(
#         x_data,
#         y_data,
#         ".",
#         label=f"{country} - observed",
#         color=lines[-1].get_color(),
#     )
# ax.set_ylabel("departures per capita")
# ax.set_xlabel("GDP per capita\n[current US$/hab.]")
# ax.legend(bbox_to_anchor=(1.1, 0.9))
# ax.set_ylim(bottom=1e-4)
# ax.set_yscale('log')
# ax.set_xscale('log')
# fig.savefig("./dep_countries.pdf")
# close(fig)