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

import plotly.io as pio
from matplotlib.pyplot import figure
from matplotlib.pyplot import subplots
from numpy import argwhere
from numpy import array
from numpy import ones
from numpy import sum as np_sum
from plotly.graph_objs import Figure
from plotly.graph_objs import Sankey
from scipy.integrate import cumulative_trapezoid

from noads.application.background_scenario_data import co2_budget_1p5deg_66percent
from noads.application.background_scenario_data import co2_budget_1p8deg_66percent
from noads.application.background_scenario_data import co2_budget_2p0deg_66percent
from noads.application.background_scenario_data import get_scenario_color
from noads.application.background_scenario_data import lines_gen

pio.renderers.default = "sphinx_gallery"

propulsion_colors = {
    "Current": "y",
    "JetA-GasTurbine": "dimgray",
    "Battery-Electric": "limegreen",
    "lH2-FuelCell": "royalblue",
    "lH2-GasTurbine": "orangered",
}


def plot_single_scenario_result(
    scenario_name, output_optimal, energy_mix, fleet, low_demand
):
    """Plot results for a single scenario."""
    years = output_optimal["year"]

    # if not isdir(folder_name):
    #     mkdir(folder_name)
    # if not isdir(f"{folder_name}/results"):
    #     mkdir(f"{folder_name}/results")
    # for name, values in output_optimal.items():
    #     if values.size == years.size:
    #         fig, ax = subplots()
    #         ax.plot(years, values)
    #         fig.savefig(f"{folder_name}/{name}.png")
    #         close(fig)

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
        )
    axes_e[0, 0].set_ylabel("Production\n[EJ / year]")
    axes_e[1, 0].set_ylabel("Emission factor\n[gCO2 / MJ]")
    fig_e.show()
    # close(fig)

    # Aggregated consumption and impacts
    # output_optimal.update(renewable_gap.default_inputs)
    fig_c, axes_c = subplots(
        len(energy_mix.input_streams),
        figsize=(6, 10),
        layout="constrained",
    )
    # Energy resources and their constraint
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
    fig_c.show()
    # close(fig)

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
            colors.append("gainsboro")
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
            co2_per_ask = (
                np_sum(
                    [
                        output_optimal[f"{aircraft.name}.{carrier.name}.consumption"]
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
        axes_f[1, i].set_ylim((0, 175))

    axes_f[0, 0].legend(bbox_to_anchor=(0.9, -0.1))
    axes_f[1, 0].legend(loc="upper right")

    axes_f[0, 0].set_ylabel("Available Seat Kilometers\n[trillion pax km]")
    axes_f[1, 0].set_ylabel("Carbon Intensity\n[g CO2 / ASK]")

    fig_f.show()
    # close(fig)

    if low_demand:
        fig_d, ax = subplots(layout="constrained")
        ax.plot(years, output_optimal["ask"], label="Fulfilled")
        ax.plot(years, output_optimal["ask_trend"], "k:", label="Trend")
        ax.plot(years, output_optimal["ask_avoided"], "r--", label="Avoided")
        ax.legend(loc="upper left", framealpha=0.7)
        ax.set_ylabel("Available Seat Kilometers\n[pax km]")
        ax.set_xlabel("Year")
        ax.set_title("Air traffic supply")
        fig_d.show()
        # close(fig)

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
        fig_a.show()
        # close(fig)

    # Simplified energy sankey diagrams
    sankey_years = [2030, 2045, 2060]
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
            title_text=f"Energy Sankey Diagram for {year} Aviation ({scenario_name})",
            font_size=16,
            width=1500,
            height=600,
        )
        p_fig.show()


def plot_scenario_comparison(scenario_outputs, year_endplots):
    """Plot basic scenario comparison."""
    fig = figure()
    gs = fig.add_gridspec(2, 9)

    rpk_2019_tril = 8664032.0e-6
    # https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/
    co2_2019_mt = 844.43
    # https://aeroscope.isae-supaero.fr/
    # kero_co2_index = 88.7  # gCO2 / MJ
    # consumption_2019_ej = (co2_2019_mt / kero_co2_index)

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

        ax1.plot(
            output_optimal["year"],
            output_optimal["rpk"] * 1e-12,
            label=scenario,
            linestyle=line,
        )
        ax1_scale.plot(
            output_optimal["year"],
            output_optimal["rpk"] * 1e-12 / rpk_2019_tril,
            linewidth=0,
        )

        ax2.plot(
            output_optimal["year"],
            output_optimal["CO2"] * 1e-12,
            linestyle=line,
        )
        ax2_scale.plot(
            output_optimal["year"],
            output_optimal["CO2"] * 1e-12 / co2_2019_mt,
            linewidth=0,
        )

        ax3.plot(
            output_optimal["year"],
            output_optimal["OIL.consumption"] * 1e-12,
            linestyle=line,
        )

        ax4.plot(
            output_optimal["year"],
            output_optimal["BIOMASS.consumption"] * 1e-12,
            linestyle=line,
        )

        ax5.plot(
            output_optimal["year"],
            output_optimal["ELECTRICITY.consumption"] * 1e-12,
            linestyle=line,
        )

        integrated_co2 = (
            cumulative_trapezoid(
                output_optimal["CO2"], output_optimal["year"], initial=0
            )
            * 1e-15
        )
        ax6.plot(
            output_optimal["year"],
            integrated_co2,
            linestyle=line,
        )
        ax6_scale.plot(
            output_optimal["year"],
            integrated_co2 * 1.0e2 / co2_budget_2p0deg_66percent,
            linewidth=0,
        )

    ax1.legend(loc="upper left", framealpha=0.7)
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
    # ax3_scale.set_ylabel("rel. 2019")
    # ax3_scale.set_ylim(bottom=0)

    ax4.set_title("Biomass")
    ax4.set_xlabel("Year")
    ax4.set_ylim(bottom=0)
    ax4.set_xlim(right=year_endplots)
    # ax4_scale.set_ylabel("rel. 2019")
    # ax4_scale.set_ylim(bottom=0)

    ax5.set_title("Electricity")
    ax5.set_xlabel("Year")
    ax5.set_ylim(bottom=0)
    ax5.set_xlim(right=year_endplots)
    # ax5_scale.set_ylabel("rel. 2019")
    # ax5_scale.set_ylim(bottom=0)

    ax6.hlines(
        0.03 * co2_budget_2p0deg_66percent * 1e-15,
        output_optimal["year"][0],
        output_optimal["year"][-1],
        label="2.9% of 2.0°C (66% conf.)",
        colors="dimgray",
        linestyles="--",
    )
    ax6.hlines(
        0.03 * co2_budget_1p8deg_66percent * 1e-15,
        output_optimal["year"][0],
        output_optimal["year"][-1],
        label="2.9% of 1.8°C (66% conf.)",
        colors="darkgray",
        linestyles="--",
    )
    ax6.hlines(
        0.03 * co2_budget_1p5deg_66percent * 1e-15,
        output_optimal["year"][0],
        output_optimal["year"][-1],
        label="2.9% of 1.5°C (66% conf.)",
        colors="lightgray",
        linestyles="--",
    )

    ax6.set_title("Carbon budget")
    ax6.set_ylabel("Gt CO2")
    ax6.set_xlabel("Year")
    ax6_scale.set_ylabel("% of 2°C budget")
    ax6.legend(loc="lower right", framealpha=0.7)

    fig.canvas.draw()
    fig.tight_layout()
    fig.show()
    # close(fig)


def plot_tech_scenario_comparison(scenario_outputs, colors):
    """Plot scenario comparison with lower, mid, and upper technology."""
    fig = figure()
    gs = fig.add_gridspec(2, 9)

    rpk_2019_tril = 8664032.0e-6
    # https://www.airlines.org/dataset/world-airlines-traffic-and-capacity/
    co2_2019_mt = 844.43
    # https://aeroscope.isae-supaero.fr/
    # kero_co2_index = 88.7  # gCO2 / MJ
    # consumption_2019_ej = (co2_2019_mt / kero_co2_index)

    ax1 = fig.add_subplot(gs[0, 0:3])
    ax2 = fig.add_subplot(gs[0, 3:6])
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:6])
    ax6 = fig.add_subplot(gs[0:2, 6:9])

    ax1_scale = ax1.twinx()
    ax2_scale = ax2.twinx()
    # ax3_scale = ax3.twinx()
    # ax4_scale = ax4.twinx()
    # ax5_scale = ax5.twinx()
    ax6_scale = ax6.twinx()

    fig.set_size_inches(15, 8)
    # fig.suptitle(f"Aviation Scenario Outcome ({ssp_name})")
    tech_labels = ["Lower", "Mid", "Upper-to-Lower"]
    for i, (scenario, output_list) in enumerate(scenario_outputs.items()):
        lines = lines_gen()
        for j, output_optimal in enumerate(output_list):
            line = next(lines)
            if j != len(output_list) - 1:
                ax1.plot(
                    output_optimal["year"],
                    output_optimal["rpk"] * 1e-12,
                    color=colors[i],
                    label=f"{scenario}, {tech_labels[j]}"
                    if i == 0
                    else (scenario if j == 0 else "_"),
                    linestyle=line,
                    linewidth=3 if i == 0 else 2,
                )
                ax1_scale.plot(
                    output_optimal["year"],
                    output_optimal["rpk"] * 1e-12 / rpk_2019_tril,
                    color=colors[i],
                    linewidth=0,
                )

                ax2.plot(
                    output_optimal["year"],
                    output_optimal["CO2"] * 1e-12,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3 if i == 0 else 2,
                )
                ax2_scale.plot(
                    output_optimal["year"],
                    output_optimal["CO2"] * 1e-12 / co2_2019_mt,
                    color=colors[i],
                    linewidth=0,
                )

                ax3.plot(
                    output_optimal["year"],
                    output_optimal["OIL.consumption"] * 1e-12,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3 if i == 0 else 2,
                )

                ax4.plot(
                    output_optimal["year"],
                    output_optimal["BIOMASS.consumption"] * 1e-12,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3 if i == 0 else 2,
                )

                ax5.plot(
                    output_optimal["year"],
                    output_optimal["ELECTRICITY.consumption"] * 1e-12,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3 if i == 0 else 2,
                )

                integrated_co2 = (
                    cumulative_trapezoid(
                        output_optimal["CO2"], output_optimal["year"], initial=0
                    )
                    * 1e-15
                )
                ax6.plot(
                    output_optimal["year"],
                    integrated_co2,
                    color=colors[i],
                    linestyle=line,
                    linewidth=3 if i == 0 else 2,
                )
                ax6_scale.plot(
                    output_optimal["year"],
                    integrated_co2 * 1.0e2 / co2_budget_2p0deg_66percent,
                    color=colors[i],
                    linewidth=0,
                )
            else:
                ax1.fill_between(
                    output_optimal["year"],
                    output_list[0]["rpk"] * 1e-12,
                    output_optimal["rpk"] * 1e-12,
                    alpha=0.2,
                    color=colors[i],
                )
                ax1_scale.plot(
                    output_optimal["year"],
                    output_optimal["rpk"] * 1e-12 / rpk_2019_tril,
                    color=colors[i],
                    linewidth=0,
                )
                ax2.fill_between(
                    output_optimal["year"],
                    output_list[0]["CO2"] * 1e-12,
                    output_optimal["CO2"] * 1e-12,
                    color=colors[i],
                    alpha=0.2,
                )
                ax2_scale.plot(
                    output_optimal["year"],
                    output_optimal["CO2"] * 1e-12 / co2_2019_mt,
                    color=colors[i],
                    linewidth=0,
                )

                ax3.fill_between(
                    output_optimal["year"],
                    output_list[0]["OIL.consumption"] * 1e-12,
                    output_optimal["OIL.consumption"] * 1e-12,
                    color=colors[i],
                    alpha=0.2,
                )

                ax4.fill_between(
                    output_optimal["year"],
                    output_list[0]["BIOMASS.consumption"] * 1e-12,
                    output_optimal["BIOMASS.consumption"] * 1e-12,
                    color=colors[i],
                    alpha=0.2,
                )

                ax5.fill_between(
                    output_optimal["year"],
                    output_list[0]["ELECTRICITY.consumption"] * 1e-12,
                    output_optimal["ELECTRICITY.consumption"] * 1e-12,
                    color=colors[i],
                    alpha=0.2,
                )

                integrated_co2 = (
                    cumulative_trapezoid(
                        output_optimal["CO2"], output_optimal["year"], initial=0
                    )
                    * 1e-15
                )
                integrated_co2_0 = (
                    cumulative_trapezoid(
                        output_list[0]["CO2"], output_list[0]["year"], initial=0
                    )
                    * 1e-15
                )

                ax6.fill_between(
                    output_optimal["year"],
                    integrated_co2_0,
                    integrated_co2,
                    color=colors[i],
                    alpha=0.2,
                )
                ax6_scale.plot(
                    output_optimal["year"],
                    integrated_co2 * 1.0e2 / co2_budget_2p0deg_66percent,
                    color=colors[i],
                    linewidth=0,
                )

    ax1.legend(loc="lower right", framealpha=0.7)
    ax1.set_title("Traffic (RPK)")
    ax1.set_ylabel("trillion pax-km")
    ax1.set_xlabel("Year")
    ax1.set_ylim(bottom=0)
    ax1_scale.set_ylim(bottom=0)
    ax1_scale.set_ylabel("relative to 2019")

    ax2.set_title("Emissions")
    ax2.set_ylabel("Mt CO2 / year")
    ax2.set_xlabel("Year")
    ax2.set_ylim(bottom=0)
    ax2_scale.set_ylim(bottom=0)
    ax2_scale.set_ylabel("relative to 2019")

    ax3.set_title("Oil")
    ax3.set_ylabel("Consumption [EJ / year]")
    ax3.set_xlabel("Year")
    ax3.set_ylim(bottom=0)

    ax4.set_title("Biomass")
    ax4.set_ylabel("Consumption [EJ / year]")
    ax4.set_xlabel("Year")
    ax4.set_ylim(bottom=0)

    ax5.set_title("Electricity")
    ax5.set_ylabel("Consumption [EJ / year]")
    ax5.set_xlabel("Year")
    ax5.set_ylim(bottom=0)

    ax6.hlines(
        0.03 * co2_budget_2p0deg_66percent * 1e-15,
        output_optimal["year"][0],
        output_optimal["year"][-1],
        label="3% of 2.0°C (66% conf.)",
        colors="dimgray",
        linestyles="--",
    )
    ax6.hlines(
        0.03 * co2_budget_1p8deg_66percent * 1e-15,
        output_optimal["year"][0],
        output_optimal["year"][-1],
        label="3% of 1.8°C (66% conf.)",
        colors="darkgray",
        linestyles="--",
    )
    ax6.hlines(
        0.03 * co2_budget_1p5deg_66percent * 1e-15,
        output_optimal["year"][0],
        output_optimal["year"][-1],
        label="3% of 1.5°C (66% conf.)",
        colors="lightgray",
        linestyles="--",
    )

    ax6.set_title("Carbon budget")
    ax6.set_ylabel("Gt CO2")
    ax6.set_xlabel("Year")
    ax6_scale.set_ylabel("% of 2°C budget")
    ax6.set_ylim((0, 55))
    ax6_scale.set_ylim((0, 50 * 1.0e2 / co2_budget_2p0deg_66percent))
    ax6.legend(loc="upper left", framealpha=0.7)

    fig.canvas.draw()
    fig.tight_layout()
    fig.show()
    # close(fig)


def plot_multi_scenario_result(
    scenario_names,
    mean_outputs,
    output_optimal,
    energy_mix,
    fleet,
    year_endplots,
    low_demand,
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
        ax.legend(loc="upper left", framealpha=0.7)
        ax.set_xlabel("Year")
        ax.set_title(mean_output)
        fig.show()
        # close(fig)

    scenario_comparison = {}
    for scenario in scenario_names:
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
        scenario_comparison.update({scenario: scenario_output})
        plot_single_scenario_result(
            scenario_name=scenario,
            output_optimal=scenario_output,
            energy_mix=energy_mix,
            fleet=fleet,
            low_demand=low_demand,
        )

    plot_scenario_comparison(scenario_comparison, year_endplots)


def plot_traffic_emissions_pareto_front(scenario_outputs, colors_markers, n_sub_opt):
    """Plot multi-scenario Traffic vs Emissions Pareto front."""
    fig, ax = subplots(layout="constrained")
    lines = lines_gen()
    for i, (scenario, output) in enumerate(scenario_outputs.items()):
        line = next(lines)
        color, marker = colors_markers[i]
        traffic = [
            output[f"{j}.cumulative.rpk"] * 1e-12
            for j in range(n_sub_opt + 1)
            if f"{j}.cumulative.rpk" in output
        ]
        emissions = [
            output[f"{j}.cumulative.CO2"] * 1e-15
            for j in range(n_sub_opt + 1)
            if f"{j}.cumulative.CO2" in output
        ]
        ax.plot(
            traffic,
            emissions,
            color=color,
            marker=marker,
            linestyle=line,
            label=scenario,
        )

    min_traffic = min(
        output[f"{j}.cumulative.rpk"] * 1e-12
        for output in scenario_outputs.values()
        for j in range(n_sub_opt + 1)
        if f"{j}.cumulative.rpk" in output
    )
    max_traffic = max(
        output[f"{j}.cumulative.rpk"] * 1e-12
        for output in scenario_outputs.values()
        for j in range(n_sub_opt + 1)
        if f"{j}.cumulative.rpk" in output
    )

    ax.hlines(
        0.03 * co2_budget_2p0deg_66percent * 1e-15,
        min_traffic,
        max_traffic,
        label="3% of 2.0°C (66% conf.)",
        colors="dimgray",
        linestyles="--",
    )
    ax.hlines(
        0.03 * co2_budget_1p8deg_66percent * 1e-15,
        min_traffic,
        max_traffic,
        label="3% of 1.8°C (66% conf.)",
        colors="darkgray",
        linestyles="--",
    )
    ax.hlines(
        0.03 * co2_budget_1p5deg_66percent * 1e-15,
        min_traffic,
        max_traffic,
        label="3% of 1.5°C (66% conf.)",
        colors="lightgray",
        linestyles="--",
    )

    ax.set_ylabel("Cumulative emissions\n[Gt CO2]")
    ax.set_xlabel("Cumulative RPK\n[trillion pax-km]")
    ax.legend(loc="lower right")
    fig.show()


# def plot_robustness_pareto_front(
#     target_scenario_name,
#     scenario_outputs,
#     colors_markers,
#     folder_name,
#     figure_name,
#     n_sub_opt,
#     low_demand,
# ):
#     """Plot multi-scenario Target Scenario vs Mean Scenario Pareto front."""
#     fig, ax = subplots(layout="constrained")
#     lines = lines_gen()
#     objective_name = "cumulative.rpk" if low_demand else "cumulative.CO2"
#     objective_scale = 1.0 if low_demand else 1.0e-15
#     all_names = ""
#     for i, (scenario, output) in enumerate(scenario_outputs.items()):
#         if i != 0:
#             all_names += ", "
#         all_names += scenario
#         line = next(lines)
#         color, marker = colors_markers[i]
#         target = [
#             output[f"{j}.{target_scenario_name}.{objective_name}"] * objective_scale
#             for j in range(n_sub_opt + 1)
#             if f"{j}.{target_scenario_name}.{objective_name}" in output
#         ]
#         mean = [
#             output[f"{j}.mean.{objective_name}"] * objective_scale
#             for j in range(n_sub_opt + 1)
#             if f"{j}.mean.{objective_name}" in output
#         ]
#         ax.plot(
#             target, mean, color=color, marker=marker, linestyle=line, label=scenario
#         )
#
#     label_name = (
#         "Cumulative RPK [trillion pax-km]"
#         if low_demand
#         else "Cumulative emissions [Gt CO2]"
#     )
#     ax.set_xlabel(f"{target_scenario_name}\n{label_name}")
#     ax.set_ylabel(f"Mean ({all_names})\n{label_name}")
#     ax.legend(loc="upper right")
#     fig.savefig(f"{folder_name}/{figure_name}.pdf")
