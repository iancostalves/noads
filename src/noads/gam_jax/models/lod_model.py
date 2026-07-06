#!/usr/bin/env python3
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

"""Created on Thu Jan 20 20:20:20 2020.

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         AircraftOperation & Systems, Air Transport Department, ENAC
"""

# ruff: noqa: E501, A001

from noads.gam_jax.models.generic_airplane_model import GAM
from noads.gam_jax.utils import data_analysis as uda
from noads.gam_jax.utils import unit

if __name__ == "__main__":
    path_to_data_base = "../database/airplane_database.xlsx"

    df, un = uda.read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    # df = df[df['name']!='A380-800'].reset_index(drop=True)

    # Remove business aircraft
    df1 = df[df["airplane_type"] != "business"].reset_index(drop=True).copy()
    un1 = un.copy()

    # Design Analysis
    # -------------------------------------------------------------------------------------------------------------------
    gam = GAM()

    nap = df1.shape[0]

    abs = "mtow"
    ord1 = "lod1"
    ord2 = "lod2"

    df1[ord1] = 0
    un1[ord1] = "no_dim"

    df1[ord2] = 0
    un1[ord2] = "no_dim"

    for n in range(nap):
        # print(df1["name"][n])
        fuselage_length = df1["total_length"][n]
        fuselage_width = df1["fuselage_width"][n]
        wing_span = df1["wing_span"][n]
        wing_area = df1["wing_area"][n]
        htp_area = df1["htp_area"][n]
        vtp_area = df1["vtp_area"][n]

        cruise_speed = df1["cruise_speed"][n]
        mtow = df1["mtow"][n]

        airplane_type = df1["airplane_type"][n]
        cruise_altp = gam.flight_altitude(airplane_type)["mission"]

        mass = 0.90 * mtow

        lod = gam.lod_model(
            fuselage_length,
            fuselage_width,
            wing_span,
            wing_area,
            htp_area,
            vtp_area,
            cruise_altp,
            cruise_speed,
            mass,
        )

        df1.at[n, ord1] = lod

        df1.at[n, ord2] = gam.get_lod(mtow)

        # print(df1["name"][n], cruise_speed, mach, cz, cx, lift_to_drag)

        title = "Theoretical L/D"

    dict1 = uda.draw_reg(
        df1,
        un1,
        abs,
        ord1,
        [[200.0, 40000.0, 200000.0, 400000.0, 1.0e6], [13.0, 16, 19, 20.0, 20.0]],
        gam.colors,
        title=title,
    )

    dict2 = uda.draw_reg(df1, un1, ord1, ord2, [[0, 20], [0, 20]], gam.colors)

    # Test on Duo Discus XLT
    # ------------------------------------------------------------------------------------------------------------------
    fuselage_length = 7.5
    fuselage_width = 0.9
    wing_span = 20
    wing_area = 16.4
    htp_area = 0.5
    vtp_area = 0.9
    cruise_altp = 1000
    cruise_speed = unit.mps_kmph(85)
    mass = 485
    design = {
        "surface_quality": 1,
        "fuselage_shape": 1,
        "fuselage_surface": 0.5,
        "wing_laminar_ratio": 0.5,
    }
    dict = gam.lod_model(
        fuselage_length,
        fuselage_width,
        wing_span,
        wing_area,
        htp_area,
        vtp_area,
        cruise_altp,
        cruise_speed,
        mass,
        design,
        full_output=True,
    )

    # Test on Duo Discus XLT
    # ------------------------------------------------------------------------------------------------------------------
    fuselage_length = 7.1
    fuselage_width = 1.19
    wing_span = 8.72
    wing_area = 14.2
    htp_area = 3.03
    vtp_area = 1.4
    cruise_altp = 1000
    cruise_speed = unit.mps_kmph(240)
    mass = 1000
    design = {
        "surface_quality": 0,
        "fuselage_shape": 0,
        "fuselage_surface": 0,
        "wing_laminar_ratio": 0,
    }
    dict = gam.lod_model(
        fuselage_length,
        fuselage_width,
        wing_span,
        wing_area,
        htp_area,
        vtp_area,
        cruise_altp,
        cruise_speed,
        mass,
        design,
        full_output=True,
    )
