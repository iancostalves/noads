from matplotlib.pyplot import show
from matplotlib.pyplot import subplots
from numpy import array
from numpy import linspace

from application.base_objects import categories_mission
from application.base_objects import tech_params_lower_upper_2020_2040_2060
from core.models.fleet.aircraft_tech_parameter import AircraftTechParameter

data_sources = {
    "NASA": {
        "battery_specific_energy": array([[2035, 2040], [500, 750]]),
        "emotor_specific_power": array([[2022, 2035, 2035], [2, 13, 16]]),
        "lh2tank_gravimetric_index": array([[], []]),
        "fuelcell_specific_power": array([[], []]),
        "fuelcell_efficiency": array([[2035], [55.2]]),
        "electronics_specific_power": array([[2035, 2040, 2050], [13, 19, 26]]),
        "struct_weight_factor": array([[], []]),
    },
    "IATA": {
        "battery_specific_energy": array([[2020, 2030, 2035], [200, 225, 400]]),
        "emotor_specific_power": array([[], []]),
        "lh2tank_gravimetric_index": array([[2025, 2035, 2035, 2050], [20, 35, 50, 70]]),
        "fuelcell_specific_power": array([[2025, 2040, 2050], [4, 8, 10]]),
        "fuelcell_efficiency": array([[], []]),
        "electronics_specific_power": array([[], []]),
        "struct_weight_factor": array([
            [2030, 2030, 2030, 2030, 2030, 2050, 2050, 2050, 2050, 2050, 2050, 2050],
            [86, 80, 88, 83, 90, 80, 72, 68, 70, 83, 73, 61]
        ]),
    },
    "ATI": {
        "battery_specific_energy": array([[2025], [220]]),
        "emotor_specific_power": array([[2026, 2035, 2050], [13, 23, 25]]),
        "lh2tank_gravimetric_index": array([
            [2035, 2035, 2040, 2040, 2040, 2050, 2050, 2050],
            [47, 58, 61, 66, 72, 64, 69, 75],
        ]),
        "fuelcell_specific_power": array([[2025, 2030, 2035, 2050], [1.5, 2.0, 2.5, 5.0]]),
        "fuelcell_efficiency": array([[2025, 2030, 2035, 2050], [60, 65, 70, 75]]),
        "electronics_specific_power": array([[2026, 2030, 2050], [8.92, 20, 30]]),
        "struct_weight_factor": array([[2025, 2030, 2035, 2050], [80, 75, 70, 60]]),
    },
    "ICCT": {
        "battery_specific_energy": array([[2025, 2030, 2050], [250, 300, 500]]),
        "emotor_specific_power": array([[], []]),
        "lh2tank_gravimetric_index": array([[2035, 2035], [20, 35]]),
        "fuelcell_specific_power": array([[2035, 2035], [1, 2]]),
        "fuelcell_efficiency": array([[2035, 2035], [45, 65]]),
        "electronics_specific_power": array([[2025], [8]]),
        "struct_weight_factor": array([[], []]),
    },
    "EASA": {
        "battery_specific_energy": array([[2022], [210]]),
        "emotor_specific_power": array([[2022, 2022], [57.6/22.7, 73.5/56.6]]),
        "lh2tank_gravimetric_index": array([[], []]),
        "fuelcell_specific_power": array([[], []]),
        "fuelcell_efficiency": array([[], []]),
        "electronics_specific_power": array([[], []]),
        "struct_weight_factor": array([[], []]),
    },
    "Adler et al. (2025)": {
        "battery_specific_energy": array([[2023, 2030], [300, 600]]),
        "emotor_specific_power": array([[2025], [8]]),
        "lh2tank_gravimetric_index": array([[2035], [50]]),
        "fuelcell_specific_power": array([[2035, 2050], [2, 6]]),
        "fuelcell_efficiency": array([[2035, 2050], [60, 75]]),
        "electronics_specific_power": array([[], []]),
        "struct_weight_factor": array([[], []]),
    }
}
markers_sources = {
    "NASA": "o",
    "IATA": "s",
    "ATI": "d",
    "ICCT": "X",
    "EASA": "*",
    "Adler et al. (2025)": "P",
}
units = {
    "battery_specific_energy": "Wh/kg",
    "emotor_specific_power": "kW/kg",
    "electronics_specific_power": "kW/kg",
    "fuelcell_specific_power": "kW/kg",
    "lh2tank_gravimetric_index": "%",
    "fuelcell_efficiency": "%",
    "struct_weight_factor": "%",
}
name_to_fullname = {
    "battery_specific_energy": "Battery Specific Energy",
    "emotor_specific_power": "E-motor Specific Power",
    "electronics_specific_power": "Power electronics Specific Power",
    "fuelcell_specific_power": "Fuel cell Specific Power",
    "lh2tank_gravimetric_index": "LH2 tank Gravimetric Index",
    "fuelcell_efficiency": "Fuel cell Efficiency",
    "struct_weight_factor": "Structural weight reduction",
}

years = linspace(2020, 2060, 41)
fig1, axes1 = subplots(
    4, 2,
    figsize=(8, 15), layout="constrained",
)
tech_params = {}
for i, (name, values) in enumerate(tech_params_lower_upper_2020_2040_2060.items()):
    lower, mid, upper = values
    lower_param = AircraftTechParameter(name, (lower[0], lower[1], lower[2]))
    mid_param = AircraftTechParameter(name, (mid[0], mid[1], mid[2]))
    upper_param = AircraftTechParameter(name, (upper[0], upper[1], upper[2]))
    tech_params[name] = (lower_param, mid_param, upper_param)

    lower_values = lower_param.value_at_entry_into_service(years)
    mid_values = mid_param.value_at_entry_into_service(years)
    upper_values = upper_param.value_at_entry_into_service(years)

    ax = axes1.flat[i]
    ax.fill_between(
        years, lower_values, upper_values,
        alpha=0.2, color="k",
        label="Upper-to-Lower",
    )
    ax.plot(years, lower_values, "k-", linewidth=2, label="Lower")
    ax.plot(years, mid_values, "k:", linewidth=2, label="Mid")
    ax.set_title(name_to_fullname[name], fontsize="medium")
    ax.set_ylabel(units[name])
    ax.set_xlabel("Year")
    for source, source_data in data_sources.items():
        x_data = source_data[name][0]
        y_data = source_data[name][1]
        ax.plot(
            x_data,
            y_data,
            markers_sources[source],
            label=source,
            markersize=8,
        )

axes1[0, 0].legend(loc="upper left", framealpha=0.4)
fig1.suptitle(
    "Aircraft Technology Parameters evolution",
    fontsize="large",
)
show()

# fig2, axes2 = subplots(3, 2, layout="constrained", figsize=(8, 12))
# categories = categories_mission.keys()
#
# for cat, category in enumerate(categories):
#     max_range = 1e-3 * categories_mission[category]["range"]
#     pax = categories_mission[category]["npax"]
#     ax = axes.flat[cat]
#
#     ax.set_ylim(ymin=0.0, ymax=ymax)
#     ax.set_title(
#         f"{category.replace('_', ' ')}\n({pax} pax, {max_range} km)",
#         fontsize="large",
#     )
#     if category in categories_resumed:
#         i_r += 1
#         ax_r = axes_r.flat[i_r]
#         ax_r.set_ylim(ymin=0.0, ymax=ymax)
#         ax_r.set_title(
#             f"{category.replace('_', ' ')}\n({pax} pax, {max_range} km)",
#             fontsize="large",
#         )
#     for prop, propulsion in enumerate(propulsion_architectures.keys()):
#         lines = lines_gen()
#         reference_ask_per_energy = 0
#         for scen, (scenario, aircraft_tech) in enumerate(tech_scenario.items()):
#             print("p", propulsion)
#             print("c", category)
#             print("s", scenario)
#             design_plane = Partial(
#                 design_prospective_airplane,
#                 category=category,
#                 propulsion=propulsion,
#                 aircraft_tech=aircraft_tech,
#             )
#             yearly_design = vmap(design_plane)
#             energy_per_ask = yearly_design(years)
#             yearly_ask_per_energy = 1.0 / energy_per_ask
#
#             if scen == 0:
#                 reference_ask_per_energy = yearly_ask_per_energy
#
#             if prop == 0:
#                 label = f"{propulsion} ({scenario})"
#             elif scen == 0:
#                 label = propulsion
#             else:
#                 label = "_"
#             if scen == 2:
#                 if prop == 0:
#                     label = f"{propulsion} (full range)"
#                 else:
#                     label = "_"
#                 ax.fill_between(
#                     years, reference_ask_per_energy, yearly_ask_per_energy,
#                     alpha=0.2, color=propulsion_colors_markers[propulsion][0],
#                     label=label,
#                 )
#                 if category in categories_resumed:
#                     ax_r.fill_between(
#                         years, reference_ask_per_energy, yearly_ask_per_energy,
#                         alpha=0.2, color=propulsion_colors_markers[propulsion][0],
#                         label=label,
#                     )
#             else:
#                 line = next(lines)
#                 ax.plot(
#                     years,
#                     yearly_ask_per_energy,
#                     label=label,
#                     color=propulsion_colors_markers[propulsion][0],
#                     linestyle=line,
#                     linewidth=3,
#                 )
#                 if category in categories_resumed:
#                     ax_r.plot(
#                         years,
#                         yearly_ask_per_energy,
#                         label=label,
#                         color=propulsion_colors_markers[propulsion][0],
#                         linestyle=line,
#                         linewidth=3,
#                     )
#
#     # Plot the category recent 2019 reference
#     ref_ask_per_energy = aeroscope_category_conso[category] ** -1
#     ax.hlines(
#         y=ref_ask_per_energy,
#         xmin=years[0], xmax=years[-1],
#         colors="k",
#         linestyles="--",
#         label="2019 reference",
#         linewidth=2,
#     )
#     if category in categories_resumed:
#         ax_r.hlines(
#             y=ref_ask_per_energy,
#             xmin=years[0], xmax=years[-1],
#             colors="k",
#             linestyles="--",
#             label="2019 reference",
#             linewidth=2,
#         )
#
# fig.suptitle("Aircraft Passenger Efficiency\n[pax km / MJ]", fontsize="x-large")
# fig_r.suptitle("Aircraft Passenger Efficiency\n[pax km / MJ]", fontsize="x-large")
#
# axes[-1, -1].clear()
# axes[-1, -1].set_axis_off()
# axes_r[-1, -1].clear()
# axes_r[-1, -1].set_axis_off()
#
# # Manually add legend
# axes[-1, -1].legend(
#     handles=[
#         Patch(color=tup_color_marker[0], label=prop_name)
#         for prop_name, tup_color_marker in propulsion_colors_markers.items()
#     ],
#     loc='center',
# )
# axes[-1, 0].set_xlabel("Entry-Into-Service")

