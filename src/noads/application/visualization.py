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
"""Utilities for visualizing decarbonization scenarios."""

from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.pyplot import figure
from matplotlib.pyplot import subplots
from numpy import argwhere
from numpy import array
from numpy import ones
from numpy import sum as np_sum
from numpy import zeros
from plotly.graph_objs import Figure
from plotly.graph_objs import Sankey
from scipy.integrate import cumulative_trapezoid

from noads.application.background_scenario_data import co2_budget_1p5deg_66percent
from noads.application.background_scenario_data import co2_budget_1p8deg_66percent
from noads.application.background_scenario_data import co2_budget_2p0deg_66percent
from noads.application.background_scenario_data import co2_budget_2p2deg_66percent
from noads.application.background_scenario_data import co2_budget_2p5deg_66percent
from noads.application.background_scenario_data import get_scenario_color
from noads.application.background_scenario_data import lines_gen

propulsion_colors = {
    "Current": "dimgray",
    "JetA-GasTurbine-v1": "maroon",
    "JetA-GasTurbine-v2": "y",
    "Battery-Electric": "limegreen",
    "lH2-FuelCell": "royalblue",
    "lH2-GasTurbine": "orangered",
}

# Reference values for 2019
RPK_2019_TRIL = 8664032.0e-6
ASK_2019 = 10664.42e9
CO2_2019_MT = 844.43
KERO_CO2_INDEX = 88.7  # gCO2 / MJ
CONSUMPTION_2019_EJ = CO2_2019_MT / KERO_CO2_INDEX
MJ_PER_ASK_2019 = (CONSUMPTION_2019_EJ * 1e12) / ASK_2019
CO2_PER_ASK_2019 = KERO_CO2_INDEX * MJ_PER_ASK_2019


def _add_carbon_budget_lines(ax, year_start, year_end):
    """Add horizontal lines for carbon budget references."""
    budget_configs = [
        (co2_budget_2p5deg_66percent, "3% of 2.5°C (66% conf.)", "black"),
        (co2_budget_2p2deg_66percent, "3% of 2.2°C (66% conf.)", "dimgray"),
        (co2_budget_2p0deg_66percent, "3% of 2.0°C (66% conf.)", "gray"),
        (co2_budget_1p8deg_66percent, "3% of 1.8°C (66% conf.)", "darkgray"),
        (co2_budget_1p5deg_66percent, "3% of 1.5°C (66% conf.)", "silver"),
    ]

    for budget, label, color in budget_configs:
        ax.hlines(
            0.03 * budget * 1e-15,
            year_start,
            year_end,
            label=label,
            colors=color,
            linestyles="--",
        )


def _aggregate_fleet_asks(output_optimal, carrier_patterns):
    """Aggregate ASK values for carriers matching given patterns.

    Args:
        output_optimal: Dict containing output data with ".ask" keys
        carrier_patterns: List of string patterns to match in keys

    Returns:
        Array of aggregated ASK values
    """
    aggregated_ask = zeros(len(output_optimal["ask"]))
    for key in output_optimal:
        if ".ask" in key and any(pattern in key for pattern in carrier_patterns):
            aggregated_ask += output_optimal[key]
    return aggregated_ask


def _setup_trend_axes():
    """Create and configure axes for trend plots."""
    fig = figure()
    gs = fig.add_gridspec(2, 3)
    fig.set_size_inches(15, 6)

    axes = {
        "rpk": fig.add_subplot(gs[0, 0]),
        "co2": fig.add_subplot(gs[0, 1]),
        "energy": fig.add_subplot(gs[1, 0]),
        "carbon": fig.add_subplot(gs[1, 1]),
        "budget": fig.add_subplot(gs[:, 2]),
    }
    scales = {key: ax.twinx() for key, ax in axes.items()}

    return fig, axes, scales


def _configure_trend_axes(axes, scales, year_start, year_end):
    """Configure labels, titles, and limits for trend axes."""
    axes["rpk"].set_title("Traffic (RPK)")
    axes["rpk"].set_ylabel("trillion pax-km")
    axes["rpk"].set_ylim(bottom=0)
    scales["rpk"].set_ylim(bottom=0)
    scales["rpk"].set_ylabel("relative to 2019")

    axes["co2"].set_title("Emissions")
    axes["co2"].set_ylabel("Mt CO2 / year")
    axes["co2"].set_ylim(bottom=0)
    scales["co2"].set_ylim(bottom=0)
    scales["co2"].set_ylabel("relative to 2019")

    axes["energy"].set_title("Energy intensity")
    axes["energy"].set_ylabel("MJ / seat-km")
    axes["energy"].set_xlabel("Year")
    axes["energy"].set_ylim(bottom=0)
    scales["energy"].set_ylim(bottom=0)
    scales["energy"].set_ylabel("relative to 2019")

    axes["carbon"].set_title("Carbon intensity")
    axes["carbon"].set_ylabel("g CO2 / seat-km")
    axes["carbon"].set_xlabel("Year")
    axes["carbon"].set_ylim(bottom=0)
    scales["carbon"].set_ylim(bottom=0)
    scales["carbon"].set_ylabel("relative to 2019")

    _add_carbon_budget_lines(axes["budget"], year_start, year_end)
    axes["budget"].set_title("Carbon budget")
    axes["budget"].set_ylabel("Gt CO2")
    axes["budget"].set_xlabel("Year")
    scales["budget"].set_ylabel("% of 2°C budget")
    axes["budget"].set_ylim((0, 100))
    scales["budget"].set_ylim((0, 100.0 * 1.0e2 * 1.0e15 / co2_budget_2p0deg_66percent))
    axes["budget"].legend(loc="upper left", framealpha=0.5)


def _plot_trend_data(
    axes, scales, years, output_optimal, color, linestyle, linewidth, label=None
):
    """Plot trend data on configured axes."""
    # RPK
    axes["rpk"].plot(
        years,
        output_optimal["rpk"] * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    scales["rpk"].plot(
        years,
        output_optimal["rpk"] * 1e-12 / RPK_2019_TRIL,
        color=color,
        linewidth=0,
    )

    # CO2
    axes["co2"].plot(
        years,
        output_optimal["CO2"] * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    scales["co2"].plot(
        years,
        output_optimal["CO2"] * 1e-12 / CO2_2019_MT,
        color=color,
        linewidth=0,
    )

    # Energy intensity
    axes["energy"].plot(
        years,
        output_optimal["mean_energy_per_ask"],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    scales["energy"].plot(
        years,
        output_optimal["mean_energy_per_ask"] / MJ_PER_ASK_2019,
        color=color,
        linewidth=0,
    )

    # Carbon intensity
    co2_per_ask = (
        output_optimal["JET-A.mean_consumption_per_ask"]
        * output_optimal["JET-A.CO2_index"]
    )
    axes["carbon"].plot(
        years,
        co2_per_ask,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    scales["carbon"].plot(
        years,
        co2_per_ask / CO2_PER_ASK_2019,
        color=color,
        linewidth=0,
    )

    # Carbon budget
    integrated_co2 = (
        cumulative_trapezoid(output_optimal["CO2"], years, initial=0) * 1e-15
    )
    axes["budget"].plot(
        years,
        integrated_co2,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    scales["budget"].plot(
        years,
        integrated_co2 * 1.0e2 / co2_budget_2p0deg_66percent,
        color=color,
        linewidth=0,
    )


def _setup_jet_fuel_axes():
    """Create and configure axes for jet fuel plots."""
    fig = figure()
    gs = fig.add_gridspec(1, 3)
    fig.set_size_inches(15, 4)

    axes = {
        "kerosene": fig.add_subplot(gs[0, 0]),
        "biofuel": fig.add_subplot(gs[0, 1]),
        "efuel": fig.add_subplot(gs[0, 2]),
    }

    return fig, axes


def _configure_jet_fuel_axes(axes, zoom_efuel=False):
    """Configure labels and limits for jet fuel axes."""
    axes["kerosene"].set_title("Kerosene")
    axes["kerosene"].set_ylabel("Consumption [EJ / year]")
    axes["kerosene"].set_xlabel("Year")
    axes["kerosene"].set_ylim(bottom=0, top=20)

    axes["biofuel"].set_title("Biofuel")
    axes["biofuel"].set_xlabel("Year")
    axes["biofuel"].set_ylim(bottom=0, top=20)

    axes["efuel"].set_title("E-Fuel")
    axes["efuel"].set_xlabel("Year")
    axes["efuel"].set_ylim(bottom=0, top=20)

    if zoom_efuel:
        inset_ax = axes["efuel"].inset_axes([0.1, 0.5, 0.4, 0.45])
        inset_ax.set_xlim(2040, 2070)
        inset_ax.set_ylim(0, 3)
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        axes["efuel_inset"] = inset_ax


def _plot_jet_fuel_data(
    axes, years, output_optimal, color, linestyle, linewidth, label=None
):
    """Plot jet fuel consumption data."""
    axes["kerosene"].plot(
        years,
        output_optimal["JET-A.KEROSENE.consumption"] * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    axes["biofuel"].plot(
        years,
        output_optimal["JET-A.BIOFUEL.consumption"] * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    axes["efuel"].plot(
        years,
        output_optimal["JET-A.E-FUEL.consumption"] * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    if "efuel_inset" in axes:
        axes["efuel_inset"].plot(
            years,
            output_optimal["JET-A.E-FUEL.consumption"] * 1e-12,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def _setup_fleet_carrier_axes():
    """Create and configure axes for fleet carrier plots."""
    fig = figure()
    gs = fig.add_gridspec(1, 3)
    fig.set_size_inches(15, 4)

    axes = {
        "jet_a": fig.add_subplot(gs[0, 0]),
        "lh2": fig.add_subplot(gs[0, 1]),
        "battery": fig.add_subplot(gs[0, 2]),
    }

    return fig, axes


def _configure_fleet_carrier_axes(axes, zoom_battery=False):
    """Configure labels and limits for fleet carrier axes."""
    axes["jet_a"].set_title("Jet-A\n(Kerosene, Biofuel, E-Fuel)")
    axes["jet_a"].set_ylabel("trillion seat-km")
    axes["jet_a"].set_xlabel("Year")
    axes["jet_a"].set_ylim(bottom=0, top=30)

    axes["lh2"].set_title("Liquid H2")
    axes["lh2"].set_xlabel("Year")
    axes["lh2"].set_ylim(bottom=0, top=30)

    axes["battery"].set_title("Battery-Electric")
    axes["battery"].set_xlabel("Year")
    axes["battery"].set_ylim(bottom=0, top=30)

    if zoom_battery:
        inset_ax = axes["battery"].inset_axes([0.1, 0.5, 0.4, 0.45])
        inset_ax.set_xlim(2050, 2070)
        inset_ax.set_ylim(0, 3)
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
        axes["battery_inset"] = inset_ax


def _plot_fleet_carrier_data(
    axes, years, output_optimal, color, linestyle, linewidth, label=None
):
    """Plot fleet carrier ASK data."""
    jet_a_ask = _aggregate_fleet_asks(output_optimal, ["JetA-GasTurbine", "Current"])
    lh2_ask = _aggregate_fleet_asks(output_optimal, ["lH2-FuelCell", "lH2-GasTurbine"])
    battery_ask = _aggregate_fleet_asks(output_optimal, ["Battery-Electric"])

    axes["jet_a"].plot(
        years,
        jet_a_ask * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )

    axes["lh2"].plot(
        years,
        lh2_ask * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    axes["battery"].plot(
        years,
        battery_ask * 1e-12,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    if "battery_inset" in axes:
        axes["battery_inset"].plot(
            years,
            battery_ask * 1e-12,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def plot_single_scenario_result(
    scenario_name,
    output_optimal,
    energy_mix,
    fleet,
    low_demand,
    save_figs=False,
    directory_path=".",
):
    """Plot results for a single scenario."""
    years = output_optimal["year"]

    # Production mix and Emission Index
    ordered_energies = ["JET-A", "BIOFUEL", "GAS-H2"]
    mixed_energies = []
    for energy_name in ordered_energies:
        mixed_energies.extend(
            energy
            for energy in energy_mix.produced_energies
            if energy.name == energy_name
        )
    fig_e, axes_e = subplots(
        2, len(mixed_energies), figsize=(12, 7), layout="constrained"
    )
    for i, energy in enumerate(mixed_energies):
        axes_e[0, i].stackplot(
            years,
            [
                output_optimal[f"{pathway.name}.production"] * 1e-12
                for pathway in energy.pathways
            ],
            labels=[pathway.name for pathway in energy.pathways],
        )
        axes_e[0, i].set_ylim((0, 22))
        axes_e[0, i].legend(loc="upper left", reverse=True, framealpha=0.4)
        axes_e[0, i].set_title(energy.name)
        axes_e[1, i].set_ylim((0, 150))
        for pathway in energy.pathways:
            axes_e[1, i].plot(
                years,
                output_optimal[f"{pathway.name}.CO2_index"] * ones(years.shape)
                if output_optimal[f"{pathway.name}.CO2_index"].size == 1
                else output_optimal[f"{pathway.name}.CO2_index"],
                "-",
                linewidth=2,
            )
        axes_e[1, i].plot(
            years,
            output_optimal[f"{energy.name}.CO2_index"],
            "k:",
            linewidth=3,
            label="Mean",
        )

    axes_e[0, 0].set_ylabel("Production\n[EJ / year]")
    axes_e[1, 0].set_ylabel("Emission factor\n[gCO2 / MJ]")
    axes_e[1, 0].legend(loc="upper left")

    if save_figs:
        fig_e.savefig(f"{directory_path}/energy_{scenario_name}.pdf")
    else:
        fig_e.show()

    # Aggregated consumption and impacts
    fig_c, axes_c = subplots(
        len(energy_mix.input_streams),
        figsize=(6, 10),
        layout="constrained",
    )
    for i, stream in enumerate(energy_mix.input_streams):
        axes_c[i].plot(
            years,
            output_optimal[f"{stream.name}.consumption"] * 1e-12,
            linewidth=2,
            label="Consumed",
        )
        axes_c[i].set_title(f"{stream.name} consumption")
        axes_c[i].set_ylabel("EJ / year")
        if stream in energy_mix.constrained_inputs:
            axes_c[i].plot(
                years,
                output_optimal[f"{stream.name}.global_production"]
                * output_optimal[f"{stream.name}.fair_share"]
                * 1e-12,
                ":",
                linewidth=3,
                label="Available to aviation",
            )
            axes_c[i].legend(loc="upper left")
    axes_c[-1].set_xlabel("Year")

    if save_figs:
        fig_c.savefig(f"{directory_path}/conso_{scenario_name}.pdf")
    else:
        fig_c.show()

    # Fleet ASK composition and mean energy consumption
    fig_f, axes_f = subplots(
        2, len(fleet.fleets), figsize=(16, 7), layout="constrained"
    )
    for i, fleet_i in enumerate(fleet.fleets):
        asks = [
            output_optimal[f"{aircraft.name}.ask"] * 1e-12
            for aircraft in fleet_i.operating_aircraft
        ]
        labels = [
            aircraft.name.replace(f"_{fleet_i.name}", "").replace("_", " ")
            for aircraft in fleet_i.operating_aircraft
        ]
        colors = [
            color
            for aircraft in fleet_i.operating_aircraft
            for prop_name, color in propulsion_colors.items()
            if prop_name in aircraft.name
        ]
        hatches = ["_" for aircraft in fleet_i.operating_aircraft]

        if low_demand:
            asks.append(output_optimal[f"{fleet_i.name}.ask_avoided"] * 1e-12)
            labels.append("Avoided demand")
            colors.append("tan")
            hatches.append("xx")

        axes_f[0, i].stackplot(
            years,
            asks,
            labels=labels,
            colors=colors,
            hatch=hatches,
        )
        axes_f[0, i].set_title(fleet_i.name.replace("_", " "))

        for j, aircraft in enumerate(fleet_i.operating_aircraft):
            if np_sum(output_optimal[f"{aircraft.name}.ask"]) > 0.1e12:
                co2_per_ask = (
                    np_sum(
                        [
                            output_optimal[
                                f"{aircraft.name}.{carrier.name}.consumption"
                            ]
                            * output_optimal[f"{carrier.name}.CO2_index"]
                            for carrier in aircraft.propulsion.energy_carrier_mix
                        ],
                        axis=0,
                    )
                    / (output_optimal[f"{aircraft.name}.ask"])
                )

                axes_f[1, i].plot(years, co2_per_ask, "-", linewidth=2, color=colors[j])
        co2_per_ask = np_sum(
            [
                output_optimal[
                    f"{fleet_i.name}.{carrier.name}.mean_consumption_per_ask"
                ]
                * output_optimal[f"{carrier.name}.CO2_index"]
                for carrier in fleet_i.consumed_carriers
            ],
            axis=0,
        )
        axes_f[1, i].plot(years, co2_per_ask, "k:", linewidth=3, label="Mean")
        axes_f[1, i].set_ylim((0, 250))

    axes_f[0, 0].legend(bbox_to_anchor=(0.9, -0.1))
    axes_f[1, 0].legend(loc="upper right")
    axes_f[0, 0].set_ylabel("Available Seat Kilometers\n[trillion pax km]")
    axes_f[1, 0].set_ylabel("Carbon Intensity\n[g CO2 / ASK]")

    if save_figs:
        fig_f.savefig(f"{directory_path}/fleet_{scenario_name}.pdf")
    else:
        fig_f.show()

    if low_demand:
        fig_d, ax = subplots(layout="constrained")
        ax.plot(years, output_optimal["ask"], label="Fulfilled")
        ax.plot(years, output_optimal["ask_trend"], "k:", label="Trend")
        ax.plot(years, output_optimal["ask_avoided"], "r--", label="Avoided")
        ax.legend(loc="upper left", framealpha=0.7)
        ax.set_ylabel("Available Seat Kilometers\n[pax km]")
        ax.set_xlabel("Year")
        ax.set_title("Air traffic supply")

        if save_figs:
            fig_d.savefig(f"{directory_path}/demand_{scenario_name}.pdf")
        else:
            fig_d.show()

        fig_a, axes_a = subplots(3, 1, figsize=(5, 10), layout="constrained")
        axes_a[0].stackplot(
            years,
            [
                output_optimal[f"{fleet_i.name}.ask_avoided"] * 1e-12
                for fleet_i in fleet.fleets
            ],
            labels=[fleet_i.name for fleet_i in fleet.fleets],
        )
        axes_a[0].set_title("Avoided ASK")
        axes_a[0].set_ylabel("trillion seat km")
        axes_a[0].legend(loc="upper left", framealpha=0.7)

        for fleet_i in fleet.fleets:
            axes_a[1].plot(
                years,
                output_optimal[f"{fleet_i.name}.relative_price_change"] * 100,
                linewidth=2,
            )
            axes_a[2].plot(
                years,
                output_optimal[f"{fleet_i.name}.discounted_relative_price_change"]
                * 1.0e2,
                linewidth=2,
            )
        axes_a[1].plot(
            years,
            output_optimal["relative_price_change"] * 100,
            "k:",
            linewidth=3,
            label="Global mean",
        )
        axes_a[1].set_title("Relative ticket price increase")
        axes_a[1].set_ylabel("%")

        axes_a[2].plot(
            years,
            output_optimal["discounted_relative_price_change"] * 100,
            "k:",
            linewidth=3,
        )
        axes_a[2].set_title("Time-discounted ticket price increase")
        axes_a[2].set_ylabel("%")
        axes_a[2].set_xlabel("Year")

        if save_figs:
            fig_a.savefig(f"{directory_path}/price_{scenario_name}.pdf")
        else:
            fig_a.show()
        # close(fig)

    if save_figs:
        # Simplified energy sankey diagrams
        sankey_years = [2035, 2045, 2065]
        mockup_pathways = {
            "Fossil": "KEROSENE",
            "Biofuel": "BIOFUEL",
            "Electrofuel": "E-FUEL",
        }
        flow_threshold = 0.001

        nodes = [input_stream.name for input_stream in energy_mix.input_streams]
        nodes.extend([energy.name for energy in energy_mix.produced_energies])
        nodes.extend([
            pathway.name
            for energy in energy_mix.produced_energies
            for pathway in energy.pathways
            if pathway not in mockup_pathways
        ])
        nodes.extend([fleet_i.name for fleet_i in fleet.fleets])

        loss_name = "Conversion loss"
        nodes.append(loss_name)

        def colorname_from_emission_factor(ef):
            # kerosene -> 89
            if ef == -1:
                return "rgba(128,128,128,0.4)"
            if ef < 0:
                return "rgba(0,0,255,0.4)"
            if ef < 22.5:
                # 0-blue(n=0) , 15-cyan(n=255)
                n = 255 * ef / 22.5
                return f"rgba(0,{int(n)},255,0.4)"
            if ef < 45:
                # 15-cyan(n=255) , 30-green(n=0)
                n = 255 * (45 - ef) / 22.5
                return f"rgba(0,255,{int(n)},0.4)"
            if ef < 67.5:
                # 30-green(n=0) , 45-yellow(n=255)
                n = 255 * (ef - 45) / 22.5
                return f"rgba({int(n)},255,0,0.4)"
            if ef < 90:
                # 45-yellow(n=255), 60-red(n=0)
                n = 255 * (90 - ef) / 22.5
                return f"rgba(255,{int(n)},0,0.4)"
            if ef < 115.5:
                # 90-red(n=255), 115.5-darkred(n=153)
                n = (153 - 255) * (ef - 90) / 22.5 + 255
                return f"rgba({int(n)},0,0,0.4)"
            return "rgba(153,0,0,0.4)"
            # elif ef < 15:
            #     return "rgba(0,255,0,0.4)"
            # elif ef < 30:
            #     return "rgba(0,255,255,0.4)"
            # elif ef < 45:
            #     return "rgba(0,0,255,0.4)"
            # elif ef < 60:
            #     return "rgba(255,255,0,0.4)"
            # elif ef < 75:
            #     return "rgba(255,128,0,0.4)"
            # elif ef < 95:
            #     "rgba(255,51,51,0.4)"  # kerosene -> 90
            # return "rgba(204,0,0,0.4)"

        for year in sankey_years:
            flows = []

            node_emissions = {
                node: output_optimal[f"{node}.CO2_index"]
                if output_optimal[f"{node}.CO2_index"].size == 1
                else output_optimal[f"{node}.CO2_index"][argwhere(years == year)]
                for node in nodes
                if node not in [f.name for f in fleet.fleets] and node != loss_name
            }
            node_emissions.update({
                f.name: array(
                    np_sum(
                        array([
                            (
                                output_optimal[f"{carrier.name}.CO2_index"]
                                * output_optimal[
                                    f"{f.name}.{carrier.name}.mean_consumption_per_ask"
                                ]
                                / output_optimal[f"{f.name}.mean_energy_per_ask"]
                            )[argwhere(years == year)]
                            for carrier in f.consumed_carriers
                        ])
                    )
                )
                for f in fleet.fleets
            })
            node_emissions.update({loss_name: array(-1)})

            for energy in energy_mix.produced_energies:
                for pathway in energy.pathways:
                    if pathway.name in mockup_pathways:
                        # From energy to energies
                        source = nodes.index(mockup_pathways[pathway.name])
                        target = nodes.index(energy.name)
                        production = (
                            output_optimal[f"{pathway.name}.production"][
                                argwhere(years == year)
                            ]
                            * 1e-12
                        )
                        emissions = (
                            output_optimal[f"{pathway.name}.CO2_index"]
                            if output_optimal[f"{pathway.name}.CO2_index"].size == 1
                            else output_optimal[f"{pathway.name}.CO2_index"][
                                argwhere(years == year)
                            ]
                        )
                        flows.append((source, target, production, emissions))
                    else:
                        pathway_in = 0
                        # From inputs to pathways
                        for input_stream in pathway.input_streams:
                            if input_stream.name in mockup_pathways:
                                source = nodes.index(mockup_pathways[input_stream.name])
                            else:
                                source = nodes.index(input_stream.name)
                            target = nodes.index(pathway.name)
                            consumption = (
                                output_optimal[
                                    f"{pathway.name}.{input_stream.name}.consumption"
                                ][argwhere(years == year)]
                                * 1e-12
                            )
                            pathway_in += consumption
                            emissions = (
                                output_optimal[f"{input_stream.name}.CO2_index"]
                                if output_optimal[f"{input_stream.name}.CO2_index"].size
                                == 1
                                else output_optimal[f"{input_stream.name}.CO2_index"][
                                    argwhere(years == year)
                                ]
                            )
                            flows.append((source, target, consumption, emissions))

                        # From pathways to energies
                        source = nodes.index(pathway.name)
                        target = nodes.index(energy.name)
                        production = (
                            output_optimal[f"{pathway.name}.production"][
                                argwhere(years == year)
                            ]
                            * 1e-12
                        )

                        emissions = (
                            output_optimal[f"{pathway.name}.CO2_index"]
                            if output_optimal[f"{pathway.name}.CO2_index"].size == 1
                            else output_optimal[f"{pathway.name}.CO2_index"][
                                argwhere(years == year)
                            ]
                        )
                        flows.append((source, target, production, emissions))

                        # Energy balance and loss
                        loss = pathway_in - production
                        target = nodes.index(loss_name)
                        flows.append((source, target, loss, -1))

            for f in fleet.fleets:
                for energy in f.consumed_carriers:
                    # From energies to fleets
                    source = nodes.index(energy.name)
                    target = nodes.index(f.name)
                    production = (
                        output_optimal[f"{f.name}.{energy.name}.consumption"][
                            argwhere(years == year)
                        ]
                        * 1e-12
                    )
                    emissions = output_optimal[f"{energy.name}.CO2_index"][
                        argwhere(years == year)
                    ]
                    flows.append((source, target, production, emissions))

            p_fig = Figure(
                data=[
                    Sankey(
                        valueformat=".2f",
                        valuesuffix="EJ",
                        node={
                            "pad": 8,
                            "thickness": 10,
                            "line": {"color": "black", "width": 1.0},
                            "label": [node.replace("_", " ") for node in nodes],
                            "color": [
                                colorname_from_emission_factor(node_emissions[node])
                                for node in nodes
                            ],
                            "align": "right",
                        },
                        link={
                            "source": [
                                flow[0] for flow in flows if flow[2] > flow_threshold
                            ],
                            "target": [
                                flow[1] for flow in flows if flow[2] > flow_threshold
                            ],
                            "value": [
                                flow[2] for flow in flows if flow[2] > flow_threshold
                            ],
                            "color": [
                                colorname_from_emission_factor(flow[3])
                                for flow in flows
                                if flow[2] > flow_threshold
                            ],
                        },
                    )
                ],
            )
            p_fig.update_layout(
                title_text=f"Energy Sankey Diagram for"
                f" {year} Aviation ({scenario_name})",
                font_size=16,
                width=1500,
                height=600,
            )
            p_fig.write_html(f"{directory_path}/sankey_{scenario_name}_{year}.html")


def plot_scenario_comparison(
    scenario_outputs,
    year_endplots,
    save_fig=False,
    directory_path=".",
):
    """Plot basic scenario comparison."""
    fig = figure()
    gs = fig.add_gridspec(2, 9)

    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:6])
    ax6 = fig.add_subplot(gs[0:2, 6:9])

    ax1_scale = ax1.twinx()
    ax2_scale = ax2.twinx()
    ax6_scale = ax6.twinx()

    fig.set_size_inches(15, 8)
    lines = lines_gen()

    for scenario, output_optimal in scenario_outputs.items():
        line = next(lines)
        years = output_optimal["year"]

        ax1.plot(
            years,
            output_optimal["rpk"] * 1e-12,
            linestyle=line,
            linewidth=2,
            label=scenario,
        )
        ax1_scale.plot(
            years,
            output_optimal["rpk"] * 1e-12 / RPK_2019_TRIL,
            linewidth=0,
        )

        ax2.plot(years, output_optimal["CO2"] * 1e-12, linestyle=line, linewidth=2)
        ax2_scale.plot(
            years,
            output_optimal["CO2"] * 1e-12 / CO2_2019_MT,
            linewidth=0,
        )

        ax3.plot(years, output_optimal["OIL.consumption"], linestyle=line, linewidth=2)
        ax4.plot(
            years, output_optimal["BIOMASS.consumption"], linestyle=line, linewidth=2
        )
        ax5.plot(
            years,
            output_optimal["ELECTRICITY.consumption"],
            linestyle=line,
            linewidth=2,
        )

        integrated_co2 = (
            cumulative_trapezoid(output_optimal["CO2"], years, initial=0) * 1e-15
        )
        ax6.plot(years, integrated_co2, linestyle=line, linewidth=2)
        ax6_scale.plot(
            years,
            integrated_co2 * 1.0e2 / co2_budget_2p0deg_66percent,
            linewidth=0,
        )

    ax1.legend(loc="upper left", framealpha=0.5)
    ax1.set_title("Revenue Passenger-Kilometers")
    ax1.set_ylabel("trillion pax-km")
    ax1.set_xlabel("Year")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(right=year_endplots)
    ax1_scale.set_ylabel("relative to 2019")
    ax1_scale.set_ylim(bottom=0)

    ax2.set_title("Emissions")
    ax2.set_ylabel("Mt CO2 / year")
    ax2.set_xlabel("Year")
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(right=year_endplots)
    ax2_scale.set_ylabel("relative to 2019")
    ax2_scale.set_ylim(bottom=0)

    ax3.set_title("Oil")
    ax3.set_ylabel("Consumption [EJ / year]")
    ax3.set_xlabel("Year")
    ax3.set_ylim(bottom=0)
    ax3.set_xlim(right=year_endplots)

    ax4.set_title("Biomass")
    ax4.set_xlabel("Year")
    ax4.set_ylim(bottom=0)
    ax4.set_xlim(right=year_endplots)

    ax5.set_title("Electricity")
    ax5.set_xlabel("Year")
    ax5.set_ylim(bottom=0)
    ax5.set_xlim(right=year_endplots)

    _add_carbon_budget_lines(ax6, years[0], years[-1])
    ax6.set_title("Carbon budget")
    ax6.set_ylabel("Gt CO2")
    ax6.set_xlabel("Year")
    ax6_scale.set_ylabel("% of 2°C budget")
    ax6.legend(loc="lower right", framealpha=0.5)

    fig.canvas.draw()
    fig.tight_layout()

    if save_fig:
        fig.savefig(f"{directory_path}/comparison.pdf")
    else:
        fig.show()


def plot_tech_scenarios_trends(
    scenario_outputs,
    colors,
    save_fig=False,
    directory_filename=".",
):
    """Plot scenario comparison with lower, mid, and upper technology."""
    fig, axes, scales = _setup_trend_axes()

    for i, (scenario, output_list) in enumerate(scenario_outputs.items()):
        for j, output_optimal in enumerate(output_list):
            years = output_optimal["year"]
            linestyle = "-" if j == 0 else (":" if j == 1 else "--")
            linewidth = 3.5 - 0.5 * i
            label = scenario if j == 0 else None

            # Skip plotting the line for the last scenario (j == 2)
            if j != 2:
                _plot_trend_data(
                    axes,
                    scales,
                    years,
                    output_optimal,
                    color=colors[i],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                )

        # Add fill_between for lower to upper technology range
        if len(output_list) >= 3:
            years = output_list[0]["year"]

            # RPK fill
            axes["rpk"].fill_between(
                years,
                output_list[0]["rpk"] * 1e-12,
                output_list[2]["rpk"] * 1e-12,
                color=colors[i],
                alpha=0.2,
            )

            # CO2 fill
            axes["co2"].fill_between(
                years,
                output_list[0]["CO2"] * 1e-12,
                output_list[2]["CO2"] * 1e-12,
                color=colors[i],
                alpha=0.2,
            )

            # Energy intensity fill
            axes["energy"].fill_between(
                years,
                output_list[0]["mean_energy_per_ask"],
                output_list[2]["mean_energy_per_ask"],
                color=colors[i],
                alpha=0.2,
            )

            # Carbon intensity fill
            co2_per_ask_lower = (
                output_list[0]["JET-A.mean_consumption_per_ask"]
                * output_list[0]["JET-A.CO2_index"]
            )
            co2_per_ask_upper = (
                output_list[2]["JET-A.mean_consumption_per_ask"]
                * output_list[2]["JET-A.CO2_index"]
            )
            axes["carbon"].fill_between(
                years, co2_per_ask_lower, co2_per_ask_upper, color=colors[i], alpha=0.2
            )

            # Carbon budget fill
            integrated_co2_lower = (
                cumulative_trapezoid(output_list[0]["CO2"], years, initial=0) * 1e-15
            )
            integrated_co2_upper = (
                cumulative_trapezoid(output_list[2]["CO2"], years, initial=0) * 1e-15
            )
            axes["budget"].fill_between(
                years,
                integrated_co2_lower,
                integrated_co2_upper,
                color=colors[i],
                alpha=0.2,
            )

    tech_legend = [
        Line2D([0], [0], color="k", ls="-", lw=2, label="Lower technology"),
        Line2D([0], [0], color="k", ls=":", lw=2, label="Mid technology"),
        Patch(facecolor="k", alpha=0.3, label="Upper to lower tech"),
    ]
    axes["energy"].legend(loc="lower left", framealpha=0.5)
    axes["carbon"].legend(handles=tech_legend, loc="lower left", framealpha=0.5)

    year_start = next(iter(scenario_outputs.values()))[0]["year"][0]
    year_end = next(iter(scenario_outputs.values()))[0]["year"][-1]
    _configure_trend_axes(axes, scales, year_start, year_end)

    fig.canvas.draw()
    fig.tight_layout()

    if save_fig:
        fig.savefig(f"{directory_filename}.pdf")
    else:
        fig.show()


def plot_tech_scenario_jet_fuel(
    scenario_outputs,
    colors,
    save_fig=False,
    directory_filename=".",
    zoom_efuel=False,
):
    """Plot jet fuel breakdown comparison with lower, mid, and upper technology."""
    fig, axes = _setup_jet_fuel_axes()
    _configure_jet_fuel_axes(axes, zoom_efuel=zoom_efuel)

    for i, (scenario, output_list) in enumerate(scenario_outputs.items()):
        lines = lines_gen()
        for j, output_optimal in enumerate(output_list):
            line = next(lines)
            years = output_optimal["year"]
            label = scenario if j == 0 else None

            if j != 2:
                _plot_jet_fuel_data(
                    axes,
                    years,
                    output_optimal,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3.5 - 0.5 * i,
                    label=label,
                )

        axes["kerosene"].legend(loc="lower left", framealpha=0.5)
        # Add fill_between for lower to upper technology range
        if len(output_list) >= 3:
            lower_output = output_list[0]
            upper_output = output_list[2]
            years = lower_output["year"]

            axes["kerosene"].fill_between(
                years,
                lower_output["JET-A.KEROSENE.consumption"] * 1e-12,
                upper_output["JET-A.KEROSENE.consumption"] * 1e-12,
                alpha=0.3,
                color=colors[i],
            )
            axes["biofuel"].fill_between(
                years,
                lower_output["JET-A.BIOFUEL.consumption"] * 1e-12,
                upper_output["JET-A.BIOFUEL.consumption"] * 1e-12,
                alpha=0.3,
                color=colors[i],
            )
            axes["efuel"].fill_between(
                years,
                lower_output["JET-A.E-FUEL.consumption"] * 1e-12,
                upper_output["JET-A.E-FUEL.consumption"] * 1e-12,
                alpha=0.3,
                color=colors[i],
            )
            if "efuel_inset" in axes:
                axes["efuel_inset"].fill_between(
                    years,
                    lower_output["JET-A.E-FUEL.consumption"] * 1e-12,
                    upper_output["JET-A.E-FUEL.consumption"] * 1e-12,
                    alpha=0.3,
                    color=colors[i],
                )

    fig.canvas.draw()
    fig.tight_layout()

    if save_fig:
        fig.savefig(f"{directory_filename}.pdf")
    else:
        fig.show()


def plot_tech_scenario_fleet_carriers(
    scenario_outputs,
    colors,
    save_fig=False,
    directory_filename=".",
    zoom_battery=False,
):
    """Plot fleet carrier ASK comparison with lower, mid, and upper technology."""
    fig, axes = _setup_fleet_carrier_axes()
    _configure_fleet_carrier_axes(axes, zoom_battery=zoom_battery)

    for i, (scenario, output_list) in enumerate(scenario_outputs.items()):
        lines = lines_gen()
        for j, output_optimal in enumerate(output_list):
            line = next(lines)
            years = output_optimal["year"]
            label = scenario if j == 0 else None

            if j != 2:
                _plot_fleet_carrier_data(
                    axes,
                    years,
                    output_optimal,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3.5 - 0.5 * i,
                    label=label,
                )

        axes["jet_a"].legend(loc="upper left", framealpha=0.5)
        # Add fill_between for lower to upper technology range
        if len(output_list) >= 3:
            lower_output = output_list[0]
            upper_output = output_list[2]
            years = lower_output["year"]

            lower_jet_a = _aggregate_fleet_asks(
                lower_output, ["JetA-GasTurbine", "Current"]
            )
            upper_jet_a = _aggregate_fleet_asks(
                upper_output, ["JetA-GasTurbine", "Current"]
            )

            lower_lh2 = _aggregate_fleet_asks(
                lower_output, ["lH2-FuelCell", "lH2-GasTurbine"]
            )
            upper_lh2 = _aggregate_fleet_asks(
                upper_output, ["lH2-FuelCell", "lH2-GasTurbine"]
            )

            lower_battery = _aggregate_fleet_asks(lower_output, ["Battery-Electric"])
            upper_battery = _aggregate_fleet_asks(upper_output, ["Battery-Electric"])

            axes["jet_a"].fill_between(
                years,
                lower_jet_a * 1e-12,
                upper_jet_a * 1e-12,
                alpha=0.3,
                color=colors[i],
            )

            axes["lh2"].fill_between(
                years, lower_lh2 * 1e-12, upper_lh2 * 1e-12, alpha=0.3, color=colors[i]
            )

            axes["battery"].fill_between(
                years,
                lower_battery * 1e-12,
                upper_battery * 1e-12,
                alpha=0.3,
                color=colors[i],
            )
            if "battery_inset" in axes:
                axes["battery_inset"].fill_between(
                    years,
                    lower_battery * 1e-12,
                    upper_battery * 1e-12,
                    alpha=0.3,
                    color=colors[i],
                )

    fig.canvas.draw()
    fig.tight_layout()

    if save_fig:
        fig.savefig(f"{directory_filename}.pdf")
    else:
        fig.show()


def plot_multi_scenario_result(
    scenario_names,
    mean_outputs,
    output_optimal,
    energy_mix,
    fleet,
    year_endplots,
    low_demand,
    save_figs=False,
    directory_path=".",
):
    """Plot multi-scenario comparison and individual scenario results."""
    for mean_output in mean_outputs:
        fig, ax = subplots(layout="constrained")
        lines = lines_gen()
        last_idx = 0

        for scenario in scenario_names:
            name_split = scenario.split("SSP")
            ssp_idx = int(name_split[1][0])
            if ssp_idx != last_idx:
                lines = lines_gen()
            last_idx = ssp_idx

            if "cumulative." in mean_output:
                var_name = mean_output.replace("cumulative.", "")
                output_vector = cumulative_trapezoid(
                    output_optimal[f"{scenario}.{var_name}"],
                    output_optimal["fixed.year"],
                    initial=0,
                )
            else:
                if output_optimal[f"{scenario}.{mean_output}"].size == 1:
                    output_vector = output_optimal[f"{scenario}.{mean_output}"] * ones(
                        output_optimal["fixed.year"].shape
                    )
                else:
                    output_vector = output_optimal[f"{scenario}.{mean_output}"]

            ax.plot(
                output_optimal["fixed.year"],
                output_vector,
                label=scenario,
                color=get_scenario_color(scenario),
                linestyle=next(lines),
                linewidth=2,
            )

        if "cumulative." in mean_output:
            var_name = mean_output.replace("cumulative.", "")
            output_vector = cumulative_trapezoid(
                output_optimal[f"mean.{var_name}"],
                output_optimal["fixed.year"],
                initial=0,
            )
        else:
            if output_optimal[f"mean.{mean_output}"].size == 1:
                output_vector = output_optimal[f"mean.{mean_output}"] * ones(
                    output_optimal["fixed.year"].shape
                )
            else:
                output_vector = output_optimal[f"mean.{mean_output}"]
        ax.plot(
            output_optimal["fixed.year"],
            output_vector,
            "k-",
            label="Mean",
            linewidth=3,
        )
        ax.legend(loc="upper left", framealpha=0.5)
        ax.set_xlabel("Year")
        ax.set_title(mean_output)

        if save_figs:
            fig.savefig(f"{directory_path}/mean_{mean_output}.pdf")
        else:
            fig.show()

    scenario_comparison = {}
    for scenario in scenario_names:
        if not Path(f"{directory_path}/{scenario}").is_dir():
            Path(f"{directory_path}/{scenario}").mkdir()
        scenario_output = {
            name.replace(f"{scenario}.", ""): value
            for name, value in output_optimal.items()
            if scenario in name
        }
        scenario_output.update({
            name.replace("fixed.", ""): value
            for name, value in output_optimal.items()
            if "fixed." in name
        })
        scenario_comparison[scenario] = scenario_output

        plot_single_scenario_result(
            scenario_name=scenario,
            output_optimal=scenario_output,
            energy_mix=energy_mix,
            fleet=fleet,
            low_demand=low_demand,
            save_figs=save_figs,
            directory_path=f"{directory_path}/{scenario}",
        )

    plot_scenario_comparison(
        scenario_comparison,
        year_endplots,
        save_fig=save_figs,
        directory_path=directory_path,
    )


def _plot_ensemble_from_scenario_comparison(
    scenario_comparison, save_figs, directory_path, prefix="ensemble"
):
    """Helper function to plot ensemble from scenario comparison dict."""
    scenario_names = list(scenario_comparison.keys())
    scenario_colors = [get_scenario_color(scenario) for scenario in scenario_names]

    # Plot ensemble trends
    fig, axes, scales = _setup_trend_axes()

    for i, (scenario, output_optimal) in enumerate(scenario_comparison.items()):
        years = output_optimal["year"]
        lines = lines_gen()
        line = next(lines)

        _plot_trend_data(
            axes,
            scales,
            years,
            output_optimal,
            color=scenario_colors[i],
            linestyle=line,
            linewidth=2.0,
            label=scenario,
        )

    axes["rpk"].legend(loc="upper left", framealpha=0.5)
    _configure_trend_axes(axes, scales, years[0], years[-1])

    fig.canvas.draw()
    fig.tight_layout()

    if save_figs:
        fig.savefig(f"{directory_path}/{prefix}_trends.pdf")
    else:
        fig.show()

    # Plot jet fuel
    fig, axes = _setup_jet_fuel_axes()

    for i, (scenario, output_optimal) in enumerate(scenario_comparison.items()):
        years = output_optimal["year"]
        lines = lines_gen()
        line = next(lines)

        _plot_jet_fuel_data(
            axes,
            years,
            output_optimal,
            color=scenario_colors[i],
            linestyle=line,
            linewidth=2.0,
            label=scenario,
        )

    axes["kerosene"].legend(loc="upper left", framealpha=0.5)
    _configure_jet_fuel_axes(axes)

    fig.canvas.draw()
    fig.tight_layout()

    if save_figs:
        fig.savefig(f"{directory_path}/{prefix}_jet_fuel.pdf")
    else:
        fig.show()

    # Plot fleet carriers
    fig, axes = _setup_fleet_carrier_axes()

    for i, (scenario, output_optimal) in enumerate(scenario_comparison.items()):
        years = output_optimal["year"]
        lines = lines_gen()
        line = next(lines)

        _plot_fleet_carrier_data(
            axes,
            years,
            output_optimal,
            color=scenario_colors[i],
            linestyle=line,
            linewidth=2.0,
            label=scenario,
        )

    axes["jet_a"].legend(loc="upper left", framealpha=0.5)
    _configure_fleet_carrier_axes(axes)

    fig.canvas.draw()
    fig.tight_layout()

    if save_figs:
        fig.savefig(f"{directory_path}/{prefix}_fleet_carriers.pdf")
    else:
        fig.show()


def plot_multiple_multi_scenario_result(
    global_scenario_names,
    multi_scenario_results,
    colors,
    save_figs=False,
    directory_path=".",
):
    """Plot comparison across multiple multi-scenario optimizations.

    Args:
        global_scenario_names: List of scenario names to plot
        multi_scenario_results: Dict mapping optimization_name -> output_dict
        colors: List of colors for each optimization
        save_figs: Whether to save figures
        directory_path: Directory to save figures
    """
    # Plot ensemble trends
    fig, axes, scales = _setup_trend_axes()

    for opt_idx, (opt_name, scenario_outputs) in enumerate(
        multi_scenario_results.items()
    ):
        lines = lines_gen()
        scenario_data = {}

        for scenario_name in global_scenario_names:
            line = next(lines)
            lw = 3.5 - 0.5 * opt_idx
            years = scenario_outputs["fixed.year"]
            output_optimal = {
                name.replace(f"{scenario_name}.", ""): value
                for name, value in scenario_outputs.items()
                if scenario_name in name
            }
            scenario_data[scenario_name] = output_optimal

            label = f"{opt_name} - {scenario_name}"
            _plot_trend_data(
                axes,
                scales,
                years,
                output_optimal,
                color=colors[opt_idx],
                linestyle=line,
                linewidth=lw,
                label=label,
            )
        axes["energy"].legend(loc="lower left", framealpha=0.5)

        # Add fill_between across scenarios
        if len(global_scenario_names) >= 2:
            first_scenario = scenario_data[global_scenario_names[0]]
            last_scenario = scenario_data[global_scenario_names[-1]]

            axes["rpk"].fill_between(
                years,
                first_scenario["rpk"] * 1e-12,
                last_scenario["rpk"] * 1e-12,
                color=colors[opt_idx],
                alpha=0.2,
            )

            axes["co2"].fill_between(
                years,
                first_scenario["CO2"] * 1e-12,
                last_scenario["CO2"] * 1e-12,
                color=colors[opt_idx],
                alpha=0.2,
            )

            axes["energy"].fill_between(
                years,
                first_scenario["mean_energy_per_ask"],
                last_scenario["mean_energy_per_ask"],
                color=colors[opt_idx],
                alpha=0.2,
            )

            co2_per_ask_first = (
                first_scenario["JET-A.mean_consumption_per_ask"]
                * first_scenario["JET-A.CO2_index"]
            )
            co2_per_ask_last = (
                last_scenario["JET-A.mean_consumption_per_ask"]
                * last_scenario["JET-A.CO2_index"]
            )
            axes["carbon"].fill_between(
                years,
                co2_per_ask_first,
                co2_per_ask_last,
                color=colors[opt_idx],
                alpha=0.2,
            )

            integrated_co2_first = (
                cumulative_trapezoid(first_scenario["CO2"], years, initial=0) * 1e-15
            )
            integrated_co2_last = (
                cumulative_trapezoid(last_scenario["CO2"], years, initial=0) * 1e-15
            )
            axes["budget"].fill_between(
                years,
                integrated_co2_first,
                integrated_co2_last,
                color=colors[opt_idx],
                alpha=0.2,
            )

    # axes['rpk'].legend(loc="upper left", framealpha=0.5)
    year_start = next(iter(multi_scenario_results.values()))["fixed.year"][0]
    year_end = next(iter(multi_scenario_results.values()))["fixed.year"][-1]
    _configure_trend_axes(axes, scales, year_start, year_end)

    fig.canvas.draw()
    fig.tight_layout()

    if save_figs:
        fig.savefig(f"{directory_path}/multi_ensemble_trends.pdf")
    else:
        fig.show()

    # Plot jet fuel
    fig, axes = _setup_jet_fuel_axes()
    _configure_jet_fuel_axes(axes, zoom_efuel=True)

    for opt_idx, (opt_name, scenario_outputs) in enumerate(
        multi_scenario_results.items()
    ):
        lines = lines_gen()
        scenario_data = {}

        for scenario_name in global_scenario_names:
            line = next(lines)
            lw = 3.5 - 0.5 * opt_idx
            years = scenario_outputs["fixed.year"]
            output_optimal = {
                name.replace(f"{scenario_name}.", ""): value
                for name, value in scenario_outputs.items()
                if scenario_name in name
            }
            scenario_data[scenario_name] = output_optimal

            label = f"{opt_name}-{scenario_name}" if opt_idx == 0 else None
            _plot_jet_fuel_data(
                axes,
                years,
                output_optimal,
                color=colors[opt_idx],
                linestyle=line,
                linewidth=lw,
                label=label,
            )

        # Add fill_between across scenarios
        if len(global_scenario_names) >= 2:
            first_scenario = scenario_data[global_scenario_names[0]]
            last_scenario = scenario_data[global_scenario_names[-1]]

            axes["kerosene"].fill_between(
                years,
                first_scenario["JET-A.KEROSENE.consumption"] * 1e-12,
                last_scenario["JET-A.KEROSENE.consumption"] * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

            axes["biofuel"].fill_between(
                years,
                first_scenario["JET-A.BIOFUEL.consumption"] * 1e-12,
                last_scenario["JET-A.BIOFUEL.consumption"] * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

            axes["efuel"].fill_between(
                years,
                first_scenario["JET-A.E-FUEL.consumption"] * 1e-12,
                last_scenario["JET-A.E-FUEL.consumption"] * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

            axes["efuel_inset"].fill_between(
                years,
                first_scenario["JET-A.E-FUEL.consumption"] * 1e-12,
                last_scenario["JET-A.E-FUEL.consumption"] * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

    fig.canvas.draw()
    fig.tight_layout()

    if save_figs:
        fig.savefig(f"{directory_path}/multi_ensemble_jet_fuel.pdf")
    else:
        fig.show()

    # Plot fleet carriers
    fig, axes = _setup_fleet_carrier_axes()
    _configure_fleet_carrier_axes(axes, zoom_battery=True)

    for opt_idx, (opt_name, scenario_outputs) in enumerate(
        multi_scenario_results.items()
    ):
        lines = lines_gen()
        scenario_data = {}

        for scenario_name in global_scenario_names:
            line = next(lines)
            lw = 3.5 - 0.5 * opt_idx
            years = scenario_outputs["fixed.year"]
            output_optimal = {
                name.replace(f"{scenario_name}.", ""): value
                for name, value in scenario_outputs.items()
                if scenario_name in name
            }
            scenario_data[scenario_name] = output_optimal

            label = f"{opt_name}-{scenario_name}" if opt_idx == 0 else None
            _plot_fleet_carrier_data(
                axes,
                years,
                output_optimal,
                color=colors[opt_idx],
                linestyle=line,
                linewidth=lw,
                label=label,
            )  # Add fill_between across scenarios
        if len(global_scenario_names) >= 2:
            first_scenario = scenario_data[global_scenario_names[0]]
            last_scenario = scenario_data[global_scenario_names[-1]]

            first_jet_a = _aggregate_fleet_asks(
                first_scenario, ["JetA-GasTurbine", "Current"]
            )
            last_jet_a = _aggregate_fleet_asks(
                last_scenario, ["JetA-GasTurbine", "Current"]
            )

            first_lh2 = _aggregate_fleet_asks(
                first_scenario, ["lH2-FuelCell", "lH2-GasTurbine"]
            )
            last_lh2 = _aggregate_fleet_asks(
                last_scenario, ["lH2-FuelCell", "lH2-GasTurbine"]
            )

            first_battery = _aggregate_fleet_asks(first_scenario, ["Battery-Electric"])
            last_battery = _aggregate_fleet_asks(last_scenario, ["Battery-Electric"])

            axes["jet_a"].fill_between(
                years,
                first_jet_a * 1e-12,
                last_jet_a * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

            axes["lh2"].fill_between(
                years,
                first_lh2 * 1e-12,
                last_lh2 * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

            axes["battery"].fill_between(
                years,
                first_battery * 1e-12,
                last_battery * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

            axes["battery_inset"].fill_between(
                years,
                first_battery * 1e-12,
                last_battery * 1e-12,
                alpha=0.2,
                color=colors[opt_idx],
            )

    fig.canvas.draw()
    fig.tight_layout()

    if save_figs:
        fig.savefig(f"{directory_path}/multi_ensemble_fleet_carriers.pdf")
    else:
        fig.show()
