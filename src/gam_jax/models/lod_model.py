#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         AircraftOperation & Systems, Air Transport Department, ENAC
"""

from utils import unit
from generic_airplane_model import GAM
from utils import data_analysis as uda


if __name__ == '__main__':

    path_to_data_base = "../database/airplane_database.xlsx"

    df,un = uda.read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    # df = df[df['name']!='A380-800'].reset_index(drop=True)

    # Remove business aircraft
    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
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

        lod = gam.lod_model(fuselage_length, fuselage_width, wing_span, wing_area, htp_area, vtp_area,
              cruise_altp, cruise_speed, mass)

        df1.at[n, ord1] = lod

        df1.at[n, ord2] = gam.get_lod(mtow)

        # print(df1["name"][n], cruise_speed, mach, cz, cx, lift_to_drag)

        title = "Theoretical L/D"


    dict1 = uda.draw_reg(df1, un1, abs, ord1, [[200., 40000., 200000., 400000., 1.e6], [13., 16, 19, 20., 20.]], gam.colors, title=title)

    dict2 = uda.draw_reg(df1, un1, ord1, ord2, [[0, 20],[0, 20]], gam.colors)


    # Test on Duo Discus XLT
    #------------------------------------------------------------------------------------------------------------------
    print("")
    fuselage_length = 7.5
    fuselage_width = 0.9
    wing_span = 20
    wing_area = 16.4
    htp_area = 0.5
    vtp_area = 0.9
    cruise_altp = 1000
    cruise_speed = unit.mps_kmph(85)
    mass = 485
    design = {"surface_quality": 1,
              "fuselage_shape": 1,
              "fuselage_surface":0.5,
              "wing_laminar_ratio": 0.5}
    dict = gam.lod_model(fuselage_length, fuselage_width, wing_span, wing_area, htp_area, vtp_area,
                         cruise_altp, cruise_speed, mass, design, full_output=True)

    print(" L/D = ", "%.2f" % dict["lift_to_drag"])
    print(" cz = ", "%.2f" % dict["cz"])
    print(" cx = ", "%.4f" % dict["cx"])
    print(" cx0 = ", "%.4f" % dict["cx0"])
    print(" cxi = ", "%.4f" % dict["cxi"])
    print(" cxc = ", "%.4f" % dict["cxc"])
    print(" cxf = ", "%.4f" % dict["cxf"])
    print(" cx_par = ", "%.4f" % dict["cx_par"])
    print(" cx_tap = ", "%.4f" % dict["cx_tap"])


    # Test on Duo Discus XLT
    #------------------------------------------------------------------------------------------------------------------
    print("")
    fuselage_length = 7.1
    fuselage_width = 1.19
    wing_span = 8.72
    wing_area = 14.2
    htp_area = 3.03
    vtp_area = 1.4
    cruise_altp = 1000
    cruise_speed = unit.mps_kmph(240)
    mass = 1000
    design = {"surface_quality": 0,
              "fuselage_shape": 0,
              "fuselage_surface":0,
              "wing_laminar_ratio": 0}
    dict = gam.lod_model(fuselage_length, fuselage_width, wing_span, wing_area, htp_area, vtp_area,
                         cruise_altp, cruise_speed, mass, design, full_output=True)

    print(" L/D = ", "%.2f" % dict["lift_to_drag"])
    print(" cz = ", "%.2f" % dict["cz"])
    print(" cx = ", "%.4f" % dict["cx"])
    print(" cx0 = ", "%.4f" % dict["cx0"])
    print(" cxi = ", "%.4f" % dict["cxi"])
    print(" cxc = ", "%.4f" % dict["cxc"])
    print(" cxf = ", "%.4f" % dict["cxf"])
    print(" cx_par = ", "%.4f" % dict["cx_par"])
    print(" cx_tap = ", "%.4f" % dict["cx_tap"])

