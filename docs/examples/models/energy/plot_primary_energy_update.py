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

"""
Primary energy of the energy carriers, update model
===================================================
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.pyplot import subplots
from numpy import linspace

from noads.application.primary_energy import CARRIERS
from noads.application.primary_energy import primary_energy_factors

# %%
# Primary energy per final energy
# -------------------------------
# Primary energy (oil, biomass or grid electricity, as in the paper) consumed per MJ of
# final energy delivered to the aircraft, for each carrier or biofuel pathway, versus
# the year. Electrolysis, power-to-liquid and liquefaction improve with the
# technology scenario of the update model (filled between the Upper and Lower
# scenarios, solid line for Lower and dotted line for Mid); refining, the biofuel
# pathways and battery charging are constant. All hydrogen is electrolytic.

years = linspace(2025, 2080, 56)
factors = [primary_energy_factors(index, years) for index in range(3)]
colors = {
    "Fossil kerosene": "maroon",
    "HEFA": "olivedrab",
    "ATJ": "yellowgreen",
    "FT": "darkgreen",
    "E-fuel": "darkorange",
    "LH2": "royalblue",
    "Battery": "limegreen",
}
resources = {
    "OIL": "Oil",
    "BIOMASS": "Biomass",
    "ELECTRICITY": "Electricity",
}

fig, axes = subplots(1, 3, figsize=(14, 5.5), layout="constrained", sharey=True)
fig.suptitle(
    "Primary energy per final energy of the carriers, update model\n[MJ / MJ]",
    fontsize="x-large",
)
for ax, (resource, title) in zip(axes, resources.items()):
    carriers = [name for name, primary in CARRIERS.items() if primary == resource]
    for carrier in carriers:
        lower, mid, upper = (factor[carrier] for factor in factors)
        ax.fill_between(years, upper, lower, color=colors[carrier], alpha=0.2, lw=0)
        ax.plot(years, lower, color=colors[carrier], ls="-", lw=3)
        ax.plot(years, mid, color=colors[carrier], ls=":", lw=3)
    ax.set_title(title, fontsize="large")
    ax.set_xlabel("Year")
    ax.set_ylim(0.0, 6.0)
    ax.grid(alpha=0.3)
    ax.legend(
        handles=[
            *(Patch(color=colors[carrier], label=carrier) for carrier in carriers),
            Line2D([0], [0], color="k", ls="-", lw=3, label="Lower"),
            Line2D([0], [0], color="k", ls=":", lw=3, label="Mid"),
        ],
        loc="upper right",
    )
axes[0].set_ylabel("Primary energy / final energy")
fig.savefig("./primary_energy_update.png", dpi=150)
