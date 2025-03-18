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

"""Prospective aircraft design accounting for technology maturing."""

from jax import vmap
from matplotlib.patches import Patch
from matplotlib.pyplot import show
from matplotlib.pyplot import subplots
from numpy import array
from numpy import linspace

from noads.application.base_objects import aeroscope_category_conso
from noads.application.base_objects import categories_mission
from noads.application.base_objects import propulsion_architectures
from noads.application.base_objects import propulsion_mission
from noads.application.base_objects import tech_params_lower_upper_2020_2040_2060
from noads.core.models.fleet.aircraft_design import AircraftDesign
from noads.core.models.fleet.aircraft_operation import AircraftOperation
from noads.core.models.fleet.aircraft_tech_parameter import AircraftTechParameter

# %%
# # Aircraft Technology Evolution
# Let's start by initializing the time evolution of some key aircraft technology
# components, according to several literature sources.

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
        "lh2tank_gravimetric_index": array([
            [2025, 2035, 2035, 2050],
            [20, 35, 50, 70],
        ]),
        "fuelcell_specific_power": array([[2025, 2040, 2050], [4, 8, 10]]),
        "fuelcell_efficiency": array([[], []]),
        "electronics_specific_power": array([[], []]),
        "struct_weight_factor": array([
            [2030, 2030, 2030, 2030, 2030, 2050, 2050, 2050, 2050, 2050, 2050, 2050],
            [86, 80, 88, 83, 90, 80, 72, 68, 70, 83, 73, 61],
        ]),
    },
    "ATI": {
        "battery_specific_energy": array([[2025], [220]]),
        "emotor_specific_power": array([[2026, 2035, 2050], [13, 23, 25]]),
        "lh2tank_gravimetric_index": array([
            [2035, 2035, 2040, 2040, 2040, 2050, 2050, 2050],
            [47, 58, 61, 66, 72, 64, 69, 75],
        ]),
        "fuelcell_specific_power": array([
            [2025, 2030, 2035, 2050],
            [1.5, 2.0, 2.5, 5.0],
        ]),
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
        "emotor_specific_power": array([[2022, 2022], [57.6 / 22.7, 73.5 / 56.6]]),
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
    },
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
propulsion_colors = {
    "JetA-GasTurbine": "dimgray",
    "Battery-Electric": "limegreen",
    "lH2-FuelCell": "royalblue",
    "lH2-GasTurbine": "orangered",
}

# %%
# Now let's visualize how the Technology Evolution scenarios compare with these sources.

years = linspace(2020, 2060, 41)
fig1, axes1 = subplots(
    4,
    2,
    figsize=(8, 15),
    layout="constrained",
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
        years,
        lower_values,
        upper_values,
        alpha=0.2,
        color="k",
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

# %%
# # Prospective Aircraft Design
# Now for each year of Entry-Into-Service we'll design an airplane using GAM and compare
# their overall empty weight and mission energy consumption.
#
# Unfeasible designs may yield infinite mass and energy, therefore the (mass/energy)
# metric is calculated on a per (pax-km) basis and then inverted. They are therefore a
# measure of traffic per (mass/energy). The highest the metric the lower the
# (mass/energy).

fig2, axes2 = subplots(3, 2, layout="constrained", figsize=(8, 12))
fig2.suptitle("Aircraft Energy Efficiency\n[pax km / MJ]", fontsize="x-large")

fig3, axes3 = subplots(3, 2, layout="constrained", figsize=(8, 12))
fig3.suptitle("Aircraft Empty-Mass Efficiency\n[pax km / kg]", fontsize="x-large")

categories = categories_mission.keys()
ymax = 4.0

for cat, category in enumerate(categories):
    max_range = 1e-3 * categories_mission[category]["range"]
    pax = categories_mission[category]["npax"]

    ax_e = axes2.flat[cat]
    ax_m = axes3.flat[cat]

    reference_energy_per_ask = aeroscope_category_conso[category]
    reference_aircraft = AircraftOperation(
        name=f"{category}_RECENT",
        propulsion=None,
        energy_per_ask=reference_energy_per_ask,
        recent=True,
    )
    for _prop, propulsion in enumerate(propulsion_architectures.keys()):
        ask_per_energy_low_to_up = []
        ask_per_owe_low_to_up = []
        for tech_idx in range(3):
            aircraft = AircraftDesign(
                name=f"{category}_{propulsion}",
                propulsion=None,
                mission={
                    **categories_mission[category],
                    **propulsion_mission[propulsion],
                    "category": category,
                },
                power_system=propulsion_architectures[propulsion],
                aircraft_tech_params=[
                    tech_param[tech_idx]
                    for tech_name, tech_param in tech_params.items()
                ],
                reference_aircraft=reference_aircraft,
            )
            model = aircraft.design_model()
            model.discipline.jax_out_func = vmap(model.discipline.jax_out_func)
            outputs = model.discipline.execute({
                f"{category}_{propulsion}.entry_into_service": years
            })
            ask_per_energy_low_to_up.append(
                1.0 / outputs[f"{category}_{propulsion}.energy_per_ask"]
            )
            ask_per_owe_low_to_up.append(
                pax * max_range / outputs[f"{category}_{propulsion}.owe"]
            )

        ax_e.fill_between(
            years,
            ask_per_energy_low_to_up[0],
            ask_per_energy_low_to_up[-1],
            alpha=0.2,
            color=propulsion_colors[propulsion],
        )
        ax_e.plot(
            years,
            ask_per_energy_low_to_up[0],
            color=propulsion_colors[propulsion],
            linestyle="-",
            linewidth=3,
        )
        ax_e.plot(
            years,
            ask_per_energy_low_to_up[1],
            color=propulsion_colors[propulsion],
            linestyle=":",
            linewidth=3,
        )
        ax_e.hlines(
            y=1.0 / reference_energy_per_ask,
            xmin=years[0],
            xmax=years[-1],
            colors="k",
            linestyles="--",
            linewidth=2,
        )

        ax_m.fill_between(
            years,
            ask_per_owe_low_to_up[0],
            ask_per_owe_low_to_up[-1],
            alpha=0.2,
            color=propulsion_colors[propulsion],
        )
        ax_m.plot(
            years,
            ask_per_owe_low_to_up[0],
            color=propulsion_colors[propulsion],
            linestyle="-",
            linewidth=3,
        )
        ax_m.plot(
            years,
            ask_per_owe_low_to_up[1],
            color=propulsion_colors[propulsion],
            linestyle=":",
            linewidth=3,
        )

    ax_e.set_ylim(ymin=0.0, ymax=ymax)
    ax_e.set_title(
        f"{category.replace('_', ' ')}\n({pax} pax, {max_range} km)",
        fontsize="large",
    )
    ax_m.set_ylim(ymin=0.0)
    ax_m.set_title(
        f"{category.replace('_', ' ')}\n({pax} pax, {max_range} km)",
        fontsize="large",
    )

axes2[-1, -1].clear()
axes2[-1, -1].set_axis_off()
axes3[-1, -1].clear()
axes3[-1, -1].set_axis_off()

# Manually add legend
axes2[-1, -1].legend(
    handles=[
        Patch(color=color, label=prop_name)
        for prop_name, color in propulsion_colors.items()
    ],
    loc="center",
)
axes2[-1, 0].set_xlabel("Entry-Into-Service")

axes3[-1, -1].legend(
    handles=[
        Patch(color=color, label=prop_name)
        for prop_name, color in propulsion_colors.items()
    ],
    loc="center",
)
axes3[-1, 0].set_xlabel("Entry-Into-Service")
show()
