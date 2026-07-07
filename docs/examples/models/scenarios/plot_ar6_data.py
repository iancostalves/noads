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
Background scenario data from AR6
=================================
"""

from matplotlib.pyplot import show

from noads.application.background_scenario_data import get_ar6_input_data
from noads.application.background_scenario_data import get_ar6_output_data

# %%
# Scenario-dependent inputs
# -------------------------
# The choice of background scenario fixes the exogenous drivers of a NOADS run. From
# the AR6 database, five time series are read for every SSP-RCP scenario used in the
# paper: total population and per-capita income (which drive air-traffic demand), the
# grid electricity emission factor, and the global production of electricity and
# biomass (which set the resource availability and the carbon intensity of the
# electricity-based energy carriers). ``get_ar6_input_data`` reads and plots them.

ar6_inputs, years = get_ar6_input_data(start_year=2010, end_year=2080, plot_data=True)
show()

# %%
# The scenarios shown are SSP1 with RCP 1.9, SSP2 with RCP 1.9/2.6/3.4, and SSP5 with
# RCP 4.5. Their per-scenario time series are returned as nested dictionaries keyed by
# variable then scenario name:

for variable in ar6_inputs:
    print(variable, "->", ", ".join(ar6_inputs[variable]))

# %%
# Aviation outputs across the AR6 ensemble
# ----------------------------------------
# The same AR6 database also reports aviation final energy and CO2 emissions for the
# full inter-model ensemble. These are used to validate the trajectories produced by
# NOADS (see the literature comparison of the extended paper): the scenarios of this
# work should fall within the ensemble spread over the near to mid term.

get_ar6_output_data(start_year=2010, end_year=2080, plot_data=True)
show()
