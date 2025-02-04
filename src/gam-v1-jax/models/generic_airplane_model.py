



#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         AircraftOperation & Systems, Air Transport Department, ENAC
"""

import numpy as np
from jax import config
from jax.numpy import array
from jax.numpy import exp
from jax.numpy import interp
from jax.numpy import log
from jax.numpy import pi
from jax.numpy import sqrt
from jax.numpy import min
from chex import assert_scalar_positive

import matplotlib.pyplot as plt
from jax.numpy import where
from optimistix import RecursiveCheckpointAdjoint

from utils import physical_data as phd
from utils import unit
import utils.data_analysis as uda

from optimistix import Newton, root_find
# from optimistix import BFGS
# from optimistix import minimise

#-----------------------------------------------------------------------------------------------------------------------
#
#  Main class
#
#-----------------------------------------------------------------------------------------------------------------------

config.update("jax_enable_x64", True)

class GAM(object):  # Data Driven Modelling

    def __init__(
        self,
        battery_specific_energy=400.0,
        emotor_specific_power=4.5,
        lh2tank_gravimetric_index=40,
        fuelcell_specific_power=1,
        lift_to_drag="model",
        struct_weight_factor=84,
        fuelcell_efficiency=50,
        electronics_specific_power=10,
    ):

        # Key definitions
        # Categories
        self.general = "general"
        self.commuter = "commuter"
        self.regional = "regional"
        self.business = "business"
        self.short_medium = "short_medium"    # Single aisle
        self.long_range = "long_range"        # Twin aisle

        # Thrusters
        self.propeller = "propeller"
        self.fan = "fan"

        # Engines
        self.turbofan = "turbofan"
        self.turboprop = "turboprop"
        self.piston = "piston"
        self.emotor = "emotor"

        self.power_elec = "power_elec"
        self.fuel_cell = "fuel_cell"

        # Energy storage
        self.petrol = "petrol"
        self.kerosene = "kerosene"
        self.gasoline = "gasoline"
        self.gh2 = "compressed_h2"
        self.lh2 = "liquid_h2"
        self.lch4 = "liquid_ch4"
        self.lnh3 = "liquid_nh3"
        self.battery = "battery"

        self.colors = {self.general: "green",
                       self.commuter: "gold",
                       self.regional: "darkorange",
                       self.business: "blue",
                       self.short_medium: "darkviolet",
                       self.long_range: "red"}

        # Dictionary structures
        self.main_power_system = {"thruster_type": None, "engine_type": None, "bpr": None, "energy_type": None, "hybrid": None}
        self.scnd_power_system = {"thruster_type": None, "engine_type": None, "energy_type": None, "hybrid_ratio": None}

        # Maximum capacity and range per category
        self.category = {self.general: {"capacity": 6, "distance": unit.m_km(500), "speed": unit.convert_from("km/h", 300),
                                        "engine_type": self.turboprop, "thruster_type": self.propeller, "bpr":None},
                         self.commuter: {"capacity": 19, "distance": unit.m_km(1500), "speed": unit.convert_from("km/h", 400),
                                        "engine_type": self.turboprop, "thruster_type": self.propeller, "bpr":None},
                         self.regional: {"capacity": 120, "distance": unit.m_km(4500), "speed": 0.5,
                                        "engine_type": self.turbofan, "thruster_type": self.fan, "bpr":12},
                         self.short_medium: {"capacity": 250, "distance": unit.m_km(8000), "speed": 0.78,
                                        "engine_type": self.turbofan, "thruster_type": self.fan, "bpr":12},
                         self.long_range: {"capacity": 550, "distance": unit.m_km(15000), "speed": 0.85,
                                        "engine_type": self.turbofan, "thruster_type": self.fan, "bpr":12}}

        # Yearly utilization versus mean range
        self.util_dist_list = unit.convert_from("NM",
                         [ 100.,  500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.]) # Nautical Miles converted to meters (SI units)
        self.util_list = [2300., 2300., 1500., 1200.,  900.,  800.,  700.,  600.,  600.]

        self.disa = 0.
        self.take_off_time = unit.s_min(1)          # 30s take off + 30s initial climb
        self.full_epower_time = unit.s_min(15)      # for hybrid thermal-electrical only, allow to _impact_index the onboard energy storage

        # Flight altitudes, [cruise altitude, diversion altitude, holding altitude]
        self.flight_altitudes = {self.general: unit.convert_from("ft", [5000, 3000, 1500]),
                                 self.commuter: unit.convert_from("ft", [10000, 6000, 1500]),
                                 self.regional: unit.convert_from("ft", [20000, 10000, 1500]),
                                 self.business: unit.convert_from("ft", [35000, 25000, 1500]),
                                 self.short_medium: unit.convert_from("ft", [35000, 25000, 1500]),
                                 self.long_range: unit.convert_from("ft", [35000, 25000, 1500])
                                 }

        # Parameter for reserve computation, [flight fuel factor, diversion distance, holding time]
        self.reserve_parameters = {self.general: [0., 0., unit.s_min(30)],
                                   self.commuter: [0., 0., unit.s_min(30)],
                                   self.regional: [0., 0., unit.s_min(45)],
                                   self.business: [0.05, unit.m_NM(200), unit.s_min(30)],
                                   self.short_medium: [0.05, unit.m_NM(200), unit.s_min(30)],
                                   self.long_range: [0.03, unit.m_NM(200), unit.s_min(30)]
                                   }

        self.max_payload_factor = 1.15  # max_payload = nominal_paylod * max_payload_factor
        self.max_fuel_factor = 1.25     # max_fuel = nominal_fuel * max_fuel_factor
        self.mlw_factor = 1.07          # MLW = MZFW * mlw_factor

        # Efficiencies
        self.prop_eff = 0.80
        self.fan_eff = 0.82
        self.emotor_eff = 0.90  # MAGNIX
        self.fuel_cell_eff = 1e-2 * fuelcell_efficiency  # (Horizon Fuel Cell)
        self.fuel_energy_ratio = 2.28   # eta_fan / (eta_prop * eta_thermal), 0.8 / (0.7 * 0.5), FHV * SFC / SEC

        # For commercial passenger airplanes, L/D is linked to the size of the aircraft through the MTOW

        self.lod_mtow_list = [200., 40000., 200000., 500000., 1.e6]
        self.lod_list = [13., 16, 19, 20., 20.]

        self.psfc_piston = unit.convert_from("kg/kW/h", 0.25)   # Lycoming IO-720

        # print("============================>", 1/(self.psfc_piston * 43e6))

        # PSFC is linked to the max power of the turboshaft
        self.psfc_power_coef = [0.2, 10, 0.65]  # kg/kW/h = a + b / (kW)**c

        # The thermal efficiency used to _impact_index TSFC is fixed
        self.eff_th_turbofan = 0.474    # 0.474
        self.fuel_mixture = 0.02

        # Reference power regression function, from general to wide bodies, Constant c adjusted for very small aircraft
        self.ref_power_factors = [8.31693845e-05, 2.03027049e+02, -1.05e5]   # [a, b, c]


        self.standard_af_mwe_factors = [-3.18952359e-07, 4.22840552e-01, -30]   # [a, b, c]   # With new total_power

        #self.furnishing_range_list = unit.convert_from("km", [0, 2000, 10000, 20000])
        #self.furnishing_index_list = [18, 18, 30, 30]       # kg/passenger
        self.furnishing_dict = {self.general: 18,
                                self.commuter: 18,
                                self.regional: 22,
                                self.business: 40,
                                self.short_medium: 22,
                                self.long_range: 30
                                }

        self.operator_item_index = 5.e-6                    # kg/passenger/range

        # Passenger mass allowance depends on airplane category
        self.mpax_dict = {self.general: 95,
                          self.commuter: 105,
                          self.regional: 110,
                          self.business: 120,
                          self.short_medium: 115,
                          self.long_range: 120
                          }
        self.mpax = "model"

        # Engine power densities
        self.power_density = {self.turbofan: unit.W_kW(4.3),        # kW/kg
                              self.turboprop: unit.W_kW(4.3),
                              self.piston: unit.W_kW(1.1),          # Lycoming IO-720
                              self.emotor: unit.W_kW(emotor_specific_power),          # Magnix
                              self.propeller: unit.W_kW(10),
                              self.fan: unit.W_kW(15),
                              self.power_elec: unit.W_kW(electronics_specific_power),
                              self.fuel_cell: unit.W_kW(fuelcell_specific_power)         # HorizonFuelCell : 770 W/kg
                              }

        # Other densities
        self.battery_enrg_density = unit.J_Wh(battery_specific_energy)  # Wh/kg
        self.battery_vol_density = 2800.  # kg/m3

        self.tank_efficiency_factor = unit.convert_from("bar", 661e-3)  # 1000 bar.L/kg, Source AFHYPAC, Fiche 4.2 from CEA document

        self.initial_gh2_pressure = unit.convert_from("bar", 700)
        self.gh2_density = phd.fuel_density(self.gh2, press=self.initial_gh2_pressure)
        # kg_H2 / (kg_H2 + kg_Tank)
        self.gh2_tank_gravimetric_index = 1 / (1 + self.initial_gh2_pressure / (self.tank_efficiency_factor * self.gh2_density))
        self.gh2_tank_volumetric_index = 6.62  # kg_GH2 / m3_Tank, Source AFHYPAC, Fiche 4.2 from CEA document, Not used in this version

        self.lh2_density = phd.fuel_density(self.lh2)    # 20.39 K
        self.lh2_tank_gravimetric_index = 1e-2 * lh2tank_gravimetric_index  # kg_LH2 / (kg_LH2 + kg_Tank), Source Hypoint 2022 : 0.5
        self.lh2_tank_volumetric_index = 6.62  # kg_LH2 / m3_Tank, Source AFHYPAC, Fiche 4.2 from CEA document, Not used in this version

        self.lch4_density = phd.fuel_density(self.lch4)    # 111.63 K
        ih2 = self.lh2_tank_gravimetric_index
        ich4 = 1 / (1+(self.lh2_density/self.lch4_density)*(1/ih2 - 1))
        self.lch4_tank_gravimetric_index = ich4  # kg_LCH4 / (kg_LCH4 + kg_Tank), Extrapolation from liquid H2 tanks
        self.lch4_tank_volumetric_index = self.lh2_tank_volumetric_index * (self.lch4_density/self.lh2_density)  # kg_CH4 / m3_Tank

        self.initial_lnh3_pressure = unit.convert_from("bar", 30)   # Liquid here because NH3 is liquid up to 50°C at less than 20 bars
        self.lnh3_density = phd.fuel_density(self.lnh3, press=self.initial_lnh3_pressure)
        # kg_LNH3 / (kg_NH3 + kg_Tank)  liquid here because NH3 is liquid up to 50°C at les than 20 bars
        self.lnh3_tank_gravimetric_index = 1 / (1 + self.initial_lnh3_pressure / (self.tank_efficiency_factor * self.lnh3_density))
        self.lnh3_tank_volumetric_index = 80  # kg_LNH3 / m3_Tank, Not used in this version

        # Tuning parameters
        self.lod = lift_to_drag      # Allows to force a value for L/D
        self.lod_factor = 1.    # Tuning of aerodynamic efficiency

        self.stdm_shift = 0.    # Mass delta on standard MWE
        self.stdm_factor = 1e-2 * struct_weight_factor   # Mass factor on standard MWE

        self.delta_payload = 0. # Reference payload tuning
        self.delta_power = 0.   # Reference power tuning

        self.kvs1g_to = 1.13    # Not used in this version
        self.cl_max_to = 2

        self.kvs1g_ld = 1.23    # Not used in this version
        self.cl_max_ld = 3

    def yearly_utilization(self, mean_range):
        """Compute the yearly utilization from the average range
        """
        return interp(mean_range, self.util_dist_list, self.util_list)

    def print_model_data(self):
        print("")
        print("=======================================================================================================")
        print("Flight altitudes, [cruise altitude, diversion altitude, holding altitude]")
        print("      self.flight_altitudes[self.general] = ", unit.convert_to("ft", self.flight_altitudes[self.general]), " ft")
        print("     self.flight_altitudes[self.commuter] = ", unit.convert_to("ft", self.flight_altitudes[self.commuter]), " ft")
        print("     self.flight_altitudes[self.regional] = ", unit.convert_to("ft", self.flight_altitudes[self.regional]), " ft")
        print("    s elf.flight_altitudes[self.business] = ", unit.convert_to("ft", self.flight_altitudes[self.business]), " ft")
        print(" self.flight_altitudes[self.short_medium] = ", unit.convert_to("ft", self.flight_altitudes[self.short_medium]), " ft")
        print("   self.flight_altitudes[self.long_range] = ", unit.convert_to("ft", self.flight_altitudes[self.long_range]), " ft")
        print("")
        print("Parameter for reserve computation, [flight fuel factor, diversion distance, holding time]")
        print("      self.reserve_parameters[self.general] = ", [self.reserve_parameters[self.general][0], unit.NM_m(self.reserve_parameters[self.general][1]), unit.min_s(self.reserve_parameters[self.general][2])], " [no_dim, NM, min]")
        print("     self.reserve_parameters[self.commuter] = ", [self.reserve_parameters[self.commuter][0], unit.NM_m(self.reserve_parameters[self.commuter][1]), unit.min_s(self.reserve_parameters[self.commuter][2])], " [no_dim, NM, min]")
        print("     self.reserve_parameters[self.regional] = ", [self.reserve_parameters[self.regional][0], unit.NM_m(self.reserve_parameters[self.regional][1]), unit.min_s(self.reserve_parameters[self.regional][2])], " [no_dim, NM, min]")
        print("     self.reserve_parameters[self.business] = ", [self.reserve_parameters[self.business][0], unit.NM_m(self.reserve_parameters[self.business][1]), unit.min_s(self.reserve_parameters[self.business][2])], " [no_dim, NM, min]")
        print(" self.reserve_parameters[self.short_medium] = ", [self.reserve_parameters[self.short_medium][0], unit.NM_m(self.reserve_parameters[self.short_medium][1]), unit.min_s(self.reserve_parameters[self.short_medium][2])], " [no_dim, NM, min]")
        print("   self.reserve_parameters[self.long_range] = ", [self.reserve_parameters[self.long_range][0], unit.NM_m(self.reserve_parameters[self.long_range][1]), unit.min_s(self.reserve_parameters[self.long_range][2])], " [no_dim, NM, min]")
        print("")
        print("Yearly utilization for commercial passenger airplanes versus mean operational range")
        print(" self.util_dist_list = ", unit.convert_to("NM", self.util_dist_list), " NM, list of mean range")
        print("      self.util_list = ", self.util_list, " 1/year, list of mission per year")
        print("")
        print("Other data for performance estimation")
        print("          self.disa = ", "%.1f" % self.disa, " K, Temperature shift for all computations")
        print(" self.take_off_time = ", "%.0f" % self.take_off_time, " s, 30s take off + initial climb")
        print("")
        print("Some design factors to be able to _impact_index a payload - Range diagram")
        print("      self.max_payload_factor = ", "%.3f" % self.max_payload_factor, " max_payload = nominal_paylod * max_payload_factor")
        print("         self.max_fuel_factor = ", "%.3f" % self.max_fuel_factor, " max_fuel = nominal_fuel * max_fuel_factor")
        print("              self.mlw_factor = ", "%.3f" % self.mlw_factor, " MLW = MZFW * mlw_factor")
        print("")
        print("For commercial passenger airplanes, L/D is linked to the size of the aircraft through the MTOW")
        print(" self.lod_mtow_list = ", self.lod_mtow_list, " kg, list of MTOW")
        print("      self.lod_list = ", self.lod_list, " L/D, list of aerodynamic efficiencies")
        print("")
        print("PSFC is linked to the max power of the turboshaft")
        print("  self.psfc_power_coef = ", self.psfc_power_coef, " a,b,c : psfc = a + b / (kW)**c, retrieved in kg/kW/h")
        print("")
        print("TSFC is computed thanks to a fixed value for thermal efficiency")
        print("self.eff_th_turbofan = ", self.eff_th_turbofan)
        print("")
        print("The fuel mixture used to _impact_index TSFC is fixed")
        print("self.fuel_mixture = ", self.fuel_mixture)
        print("")
        print("Reference power regression function, from general to wide bodies, Constant c adjusted for very small aircraft")
        print(" self.ref_power_factors = ", self.ref_power_factors, " a,b,c : ref_power = a * mtow**2 + b*mtow + c, total installed power")
        print("")
        print("Standard mass regression function, from general to wide bodies, Constant c adjusted for very small aircraft")
        print("  self.standard_af_mwe_factors = ", self.standard_af_mwe_factors, " a,b,c : standard_af_mwe = a * mtow**2 + b*mtow + c")
        print("")
        print("Furnishing allowance depends on airplane category")
        print("      self.furnishing_dict[self.general] = ", self.furnishing_dict[self.general], " kg")
        print("     self.furnishing_dict[self.commuter] = ", self.furnishing_dict[self.commuter], " kg")
        print("     self.furnishing_dict[self.regional] = ", self.furnishing_dict[self.regional], " kg")
        print("     self.furnishing_dict[self.business] = ", self.furnishing_dict[self.business], " kg")
        print(" self.furnishing_dict[self.short_medium] = ", self.furnishing_dict[self.short_medium], " kg")
        print("   self.furnishing_dict[self.long_range] = ", self.furnishing_dict[self.long_range], " kg")
        print("")
        print("Furnishing are computed thanks to a fixed index")
        print(" self.operator_item_index = ", "%.1f" % (self.operator_item_index*1e6), " kg/pax/1000km")
        print("")
        print("Passenger mass allowance depends on airplane category")
        print("      self.mpax_dict[self.general] = ", self.mpax_dict[self.general], " kg")
        print("     self.mpax_dict[self.commuter] = ", self.mpax_dict[self.commuter], " kg")
        print("     self.mpax_dict[self.regional] = ", self.mpax_dict[self.regional], " kg")
        print("     self.mpax_dict[self.business] = ", self.mpax_dict[self.business], " kg")
        print(" self.mpax_dict[self.short_medium] = ", self.mpax_dict[self.short_medium], " kg")
        print("   self.mpax_dict[self.long_range] = ", self.mpax_dict[self.long_range], " kg")
        print("")
        print("Various efficiencies and performance indices used by models")
        print("      self.prop_eff = ", "%.3f" % self.prop_eff, " Propeller efficiency")
        print("       self.fan_eff = ", "%.3f" % self.fan_eff, " Fan propeller like efficiency")
        print("    self.emotor_eff = ", "%.3f" % self.emotor_eff, " Electric motor efficiency")
        print(" self.fuelcell_efficiency = ", "%.3f" % self.fuel_cell_eff, " Fuell cell efficiency")
        print("")
        print("   self.power_density[self.turbofan] = ", "%.2f" % unit.kW_W(self.power_density[self.turbofan]), " kW/kg")
        print("  self.power_density[self.turboprop] = ", "%.2f" % unit.kW_W(self.power_density[self.turboprop]), " kW/kg")
        print("     self.power_density[self.piston] = ", "%.2f" % unit.kW_W(self.power_density[self.piston]), " kW/kg")
        print("     self.power_density[self.emotor] = ", "%.2f" % unit.kW_W(self.power_density[self.emotor]), " kW/kg")
        print(" self.power_density[self.power_elec] = ", "%.2f" % unit.kW_W(self.power_density[self.power_elec]), " kW/kg")
        print("  self.power_density[self.fuel_cell] = ", "%.2f" % unit.kW_W(self.power_density[self.fuel_cell]), " kW/kg")
        print("  self.power_density[self.propeller] = ", "%.2f" % unit.kW_W(self.power_density[self.propeller]), " kW/kg")
        print("        self.power_density[self.fan] = ", "%.2f" % unit.kW_W(self.power_density[self.fan]), " kW/kg")
        print("")
        print(" self.battery_enrg_density = ", "%.0f" % unit.Wh_J(self.battery_enrg_density), " Wh/kg")
        print("  self.battery_vol_density = ", "%.0f" % self.battery_vol_density, " kg/m3")
        print("")
        print("      self.tank_efficiency_factor = ", "%.0f" % unit.convert_to("bar", self.tank_efficiency_factor*1e3), " bar.liter/kg")
        print("")
        print("        self.initial_gh2_pressure = ", "%.0f" % unit.convert_to("bar", self.initial_gh2_pressure), " bar")
        print("                 self.gh2_density = ", "%.2f" % self.gh2_density, " kg/m3")
        print("  self.gh2_tank_gravimetric_index = ", "%.3f" % self.gh2_tank_gravimetric_index, " kg_GH2 / (kg_GH2 + kg_Tank")
        print("   self.gh2_tank_volumetric_index = ", "%.1f" % self.gh2_tank_volumetric_index, " kg_GH2 / (m3_Tank_+_GH2)")
        print("")
        print("                 self.lh2_density = ", "%.2f" % self.lh2_density, " kg/m3")
        print("  self.lh2_tank_gravimetric_index = ", "%.3f" % self.lh2_tank_gravimetric_index, " kg_LH2 / (kg_LH2 + kg_Tank")
        print("   self.lh2_tank_volumetric_index = ", "%.1f" % self.lh2_tank_volumetric_index, " kg_LH2 / (m3_Tank_+_LH2)")
        print("")
        print("                self.lch4_density = ", "%.2f" % self.lch4_density, " kg/m3")
        print(" self.lch4_tank_gravimetric_index = ", "%.3f" % self.lch4_tank_gravimetric_index, " kg_LCH4 / (kg_LCH4 + kg_Tank")
        print("  self.lch4_tank_volumetric_index = ", "%.1f" % self.lch4_tank_volumetric_index, " kg_LCH4 / (m3_Tank_+_LCH4)")
        print("")
        print("       self.initial_lnh3_pressure = ", unit.convert_to("bar", self.initial_lnh3_pressure), " bar")
        print("                self.lnh3_density = ", "%.2f" % self.lnh3_density, " kg/m3")
        print(" self.lnh3_tank_gravimetric_index = ", "%.3f" % self.lnh3_tank_gravimetric_index, " kg_LNH3 / (kg_LNH3 + kg_Tank")
        print("  self.lnh3_tank_volumetric_index = ", "%.1f" % self.lnh3_tank_volumetric_index, " kg_LNH3 / (m3_Tank_+_LNH3)")

    def flight_altitude(self, airplane_type, cruise_altp=None):
        """return [cruise altitude, diversion altitude, holding altitude]
        """
        mz, dz, hz = self.flight_altitudes[airplane_type]
        if cruise_altp is not None:
            mz = cruise_altp
        return {"airplane_type": airplane_type, "mission": mz, "diversion": dz, "holding": hz}

    def reserve_data(self, airplane_type):
        """return [mission fuel factor, diversion leg, holding time]
        """
        ff, dl, ht = self.reserve_parameters[airplane_type]
        return {"airplane_type": airplane_type, "fuel_factor": ff, "diversion_leg": dl, "holding_time": ht}

    def get_lod(self, mtow):
        if self.lod == "model":
            lod_ref = interp(mtow, array(self.lod_mtow_list), array(self.lod_list))
            lod = lod_ref * self.lod_factor
        else:
            lod = self.lod
        return lod

    def get_piston_sfc(self, energy_type):
        fhv = phd.fuel_heat(energy_type)
        ksfc = phd.fuel_heat(self.gasoline) / fhv
        sfc = ksfc * self.psfc_piston
        return sfc, fhv

    def get_turboprop_sfc(self, max_power, energy_type):
        fhv = phd.fuel_heat(energy_type)
        ksfc = phd.fuel_heat(self.kerosene) / fhv
        a, b, c = self.psfc_power_coef
        sfc = ksfc * unit.convert_from("kg/kW/h", a + b / unit.kW_W(max_power)**c)
        return sfc, fhv

    def get_turbofan_sfc(self, tas, BPR, energy_type):
        fhv = phd.fuel_heat(energy_type)
        eff_th = self.eff_th_turbofan
        alpha = self.fuel_mixture
        eff_pr = 1 / (0.5 + sqrt(0.25 + ((alpha * eff_th * fhv) / (2 * (1 + BPR) * tas ** 2))))
        sfc = tas / (eff_th * eff_pr * fhv)
        return sfc, fhv

    def ref_power(self, mtow):
        """Required total power for an airplane with a given MTOW
        """
        a, b, c = self.ref_power_factors
        power = (a * mtow + b) * mtow + c + self.delta_power
        return power

    def take_off_energy(self, total_power):
        take_off_enrg = total_power * self.take_off_time    # Take off
        return take_off_enrg

    def climb_energy(self, mass, alt):
        climb_enrg = mass * alt * phd.gravity()      # Altitude elevation
        return climb_enrg

    def standard_mass(self, mtow):
        """Averaged Standard standard MWE for an airplane with a given MTOW (std_mass = MWE - furnishing - operator items)
        """
        a, b, c = self.standard_af_mwe_factors
        std_mass = (a * mtow + b) * mtow + c
        return std_mass

    def furnishing(self, npax, category):
        """Semi empirical furnishing mass
        """
        index = self.furnishing_dict[category]
        return index * npax

    def op_item(self, npax, distance):
        """Semi empirical mass for operator items
        """
        kp = self.operator_item_index
        return kp * npax * distance

    def get_pax_allowance(self, category):
        """Compute passenger mass allowance
        WARNING : continuous function to avoid convergence problem within solvings
        """
        if self.mpax=="model":
            m_pax = self.mpax_dict[category]
        else:
            m_pax = self.mpax
        return m_pax

    def get_pk_o_mass_max(self, owe):
        return (0.65e3 * abs(owe) ** 1.35) / owe

    def get_pk_o_mass_min(self, owe):
        return (11.0 * abs(owe) ** 1.6) / owe

    def propulsion_mass(self, power_system, total_power):
        """Estimates the mass of the propulsion system according to the selected architecture
        """
        energy_type = power_system["energy_type"]
        engine_type = power_system["engine_type"]
        thruster_type = power_system["thruster_type"]

        fuel_cell_system_mass = 0
        if engine_type == self.piston:
            propulsion_mass = total_power / self.power_density[engine_type]
            propulsion_mass += total_power / self.power_density[self.propeller]
        elif engine_type == self.turboprop:
            propulsion_mass = total_power / self.power_density[engine_type]
            propulsion_mass += total_power / self.power_density[self.propeller]
        elif engine_type == self.turbofan:
            propulsion_mass = total_power / self.power_density[engine_type]
        elif engine_type == self.emotor:
            propulsion_mass = total_power / self.power_density[engine_type]
            propulsion_mass += total_power / self.power_density[self.power_elec]
            if thruster_type == self.fan:
                propulsion_mass += total_power / self.power_density[self.fan]
            elif thruster_type == self.propeller:
                propulsion_mass += total_power / self.power_density[self.propeller]
            else:
                raise Exception("target power system - thruster type is unknown")

            if energy_type in [self.gh2, self.lh2]:
                fuel_cell_system_mass = (total_power / self.emotor_eff) / self.power_density[self.fuel_cell]
            elif energy_type == self.battery:
                pass
            else:
                raise Exception("target power system - energy_type is unknown")
        else:
            raise Exception("engine type is unknown")

        return propulsion_mass, fuel_cell_system_mass


    def energy_storage(self, power_system, max_fuel, max_enrg):
        """Compute fuel and or energy storage mass
        """
        energy_storage_mass = 0.
        if power_system["energy_type"] in [self.kerosene, self.gasoline, self.petrol]:
            fuel_density = phd.fuel_density(power_system["energy_type"])
        elif power_system["energy_type"] == self.gh2:
            fuel_density = phd.fuel_density(self.gh2, press=self.initial_gh2_pressure)
            self.gh2_tank_gravimetric_index = 1 / (
                    1 + self.initial_gh2_pressure / (self.tank_efficiency_factor * fuel_density))
            energy_storage_mass += max_fuel * (1. / self.gh2_tank_gravimetric_index - 1.)
        elif power_system["energy_type"] == self.lh2:
            fuel_density = phd.fuel_density(self.lh2)
            energy_storage_mass += max_fuel * (1. / self.lh2_tank_gravimetric_index - 1.)
        elif power_system["energy_type"] == self.lch4:
            fuel_density = phd.fuel_density(self.lch4)
            energy_storage_mass += max_fuel * (1. / self.lch4_tank_gravimetric_index - 1.)
        elif power_system["energy_type"] == self.lnh3:
            fuel_density = phd.fuel_density(self.lnh3)
            self.lnh3_tank_gravimetric_index = 1 / (
                        1 + self.initial_lnh3_pressure / (self.tank_efficiency_factor * fuel_density))
            energy_storage_mass += max_fuel * (1. / self.lnh3_tank_gravimetric_index - 1.)
        elif power_system["energy_type"] == self.battery:
            fuel_density = phd.fuel_density(self.battery)
            energy_storage_mass += max_enrg / self.battery_enrg_density
        elif power_system["energy_type"] == self.kerosene:
            fuel_density = phd.fuel_density(self.kerosene)
        else:
            raise Exception("energy_type is unknown")

        return energy_storage_mass, fuel_density

    def owe_structure(self, category, npax, mtow, distance, total_power, max_fuel, max_enrg, power_system):
        """Compute OWE from the point of view of structures
        """
        furnishing = self.furnishing(npax, category)
        operator_items = self.op_item(npax, distance)
        standard_mass = self.standard_mass(mtow)  # Standard MWE is MWE without furnishing
        propulsion_mass, fuel_cell_system_mass = self.propulsion_mass(power_system, total_power)
        energy_storage_mass, fuel_density = self.energy_storage(power_system, max_fuel, max_enrg)

        basic_mwe = standard_mass * self.stdm_factor + self.stdm_shift
        std_mwe = basic_mwe + propulsion_mass + energy_storage_mass + fuel_cell_system_mass
        mwe = std_mwe + furnishing
        owe = mwe + operator_items

        return {"owe": owe,
                "op_item": operator_items,
                "mwe": mwe,
                "furnishing": furnishing,
                "std_mwe": std_mwe,
                "propulsion_mass": propulsion_mass,
                "fuel_cell_system_mass": fuel_cell_system_mass,
                "energy_storage_mass": energy_storage_mass,
                "fuel_density": fuel_density,
                "basic_mwe": basic_mwe,
                "stdm_factor": self.stdm_factor,
                "stdm_shift": self.stdm_shift}

    def get_tas(self, tamb, speed, speed_type):
        vsnd = phd.sound_speed(tamb)
        if speed_type == "mach":
            tas = speed * vsnd
            return tas, speed
        elif speed_type == "tas":
            mach = speed/vsnd
            return speed, mach

    def leg_fuel(self, start_mass, distance, altp, speed, speed_type, mtow, max_power, power_system):
        """Compute fuel and or energy over a given distance
        """
        pamb, tamb, g = phd.atmosphere_g(altp, self.disa)
        tas, mach = self.get_tas(tamb, speed, speed_type)

        time = distance / tas
        lod = self.get_lod(mtow)
        if power_system["engine_type"] == self.piston:
            sfc, fhv = self.get_piston_sfc(power_system["energy_type"])
            fuel = start_mass * (1. - exp(-(sfc * g * distance) / (self.prop_eff * lod)))  # piston engine
            enrg = fuel * fhv
            eff = self.prop_eff / (sfc*fhv)
        elif power_system["engine_type"] == self.turboprop:
            sfc, fhv = self.get_turboprop_sfc(max_power, power_system["energy_type"])
            fuel = start_mass * (1. - exp(-(sfc * g * distance) / (self.prop_eff * lod)))  # turboprop
            enrg = fuel * fhv
            eff = self.prop_eff / (sfc*fhv)
        elif power_system["engine_type"] == self.turbofan:
            sfc, fhv = self.get_turbofan_sfc(tas, power_system["bpr"], power_system["energy_type"])
            fuel = start_mass * (1. - exp(-(sfc * g * distance) / (tas * lod)))  # turbofan
            enrg = fuel * fhv
            eff = tas / (sfc*fhv)
        elif power_system["engine_type"] == self.emotor:
            if power_system["thruster_type"] == self.propeller:
                if power_system["energy_type"] in [self.gh2, self.lh2]:  # electroprop + fuel cell
                    eff = self.prop_eff * self.emotor_eff * self.fuel_cell_eff
                    fhv = phd.fuel_heat(power_system["energy_type"])
                    fuel = start_mass * (1. - exp(-(g * distance) / (eff * fhv * lod)))
                    enrg = fuel * fhv
                elif power_system["energy_type"] == self.battery:  # electroprop + battery
                    eff = self.prop_eff * self.emotor_eff  # electrofan
                    enrg = start_mass * g * distance / (eff * lod)
                    fuel = 0.
            elif power_system["thruster_type"] == self.fan:
                if power_system["energy_type"] in [self.gh2, self.lh2]:
                    eff = self.fan_eff * self.emotor_eff * self.fuel_cell_eff  # electrofan + fuel cell
                    fhv = phd.fuel_heat(power_system["energy_type"])
                    fuel = start_mass * (1. - exp(-(g * distance) / (eff * fhv * lod)))
                    enrg = fuel * fhv
                elif power_system["energy_type"] == self.battery:
                    eff = self.fan_eff * self.emotor_eff  # electrofan
                    enrg = start_mass * g * distance / (eff * lod)  # electrofan + battery
                    fuel = 0.
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        return fuel, enrg, lod, eff, time

    def holding_fuel(self, start_mass, time, altp, speed, speed_type, mtow, max_power, power_system):
        """Compute the fuel for a given holding time
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb, tamb, g = phd.atmosphere_g(altp, self.disa)
        tas,mach = self.get_tas(tamb, speed, speed_type)

        lod = self.get_lod(mtow)

        if power_system["engine_type"] == self.piston:
            sfc, fhv = self.get_piston_sfc(power_system["energy_type"])
            fuel = start_mass * (1. - exp(-(sfc * g * tas * time) / (self.prop_eff * lod)))  # piston
            enrg = fuel * fhv
        elif power_system["engine_type"] == self.turboprop:
            sfc, fhv = self.get_turboprop_sfc(max_power, power_system["energy_type"])
            fuel = start_mass * (1. - exp(-(sfc * g * tas * time) / (self.prop_eff * lod)))  # turboprop
            enrg = fuel * fhv
        elif power_system["engine_type"] == self.turbofan:
            sfc, fhv = self.get_turbofan_sfc(tas, power_system["bpr"], power_system["energy_type"])
            fuel = start_mass * (1 - exp(-g * sfc * time / lod))  # turbofan
            enrg = fuel * fhv
        elif power_system["engine_type"] == self.emotor:
            if power_system["thruster_type"] == self.propeller:
                if power_system["energy_type"] in [self.gh2, self.lh2]:
                    eff = self.prop_eff * self.emotor_eff * self.fuel_cell_eff
                    fhv = phd.fuel_heat(power_system["energy_type"])
                    fuel = start_mass * (1 - exp(-(g * tas * time) / (eff * fhv * lod)))  # electroprop + fuel cell
                    enrg = fuel * fhv
                elif power_system["energy_type"] == self.battery:
                    eff = self.prop_eff * self.emotor_eff
                    enrg = start_mass * g * tas * time / (eff * lod)  # electroprop + battery
                    fuel = 0.
            elif power_system["thruster_type"] == self.fan:
                if power_system["energy_type"] in [self.gh2, self.lh2]:
                    eff = self.fan_eff * self.emotor_eff * self.fuel_cell_eff
                    fhv = phd.fuel_heat(power_system["energy_type"])
                    fuel = start_mass * (1 - exp(-(g * tas * time) / (eff * fhv * lod)))  # electrofan + fuel cell
                    enrg = fuel * fhv
                elif power_system["energy_type"] == self.battery:
                    eff = self.fan_eff * self.emotor_eff
                    enrg = start_mass * g * tas * time / (eff * lod)  # electrofan + battery
                    fuel = 0.
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        return fuel, enrg, lod

    def total_fuel(self, tow, distance, cruise_speed, mtow, total_power, power_system, altitude_data, reserve_data):
        """Compute the total fuel required for a mission
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        if cruise_speed > 1:
            speed_type = "tas"
        else:
            speed_type = "mach"

        max_power = total_power / power_system["engine_count"]
        cruise_altp = altitude_data["mission"]

        mission_fuel = 0
        mission_enrg = 0

        mission_enrg += self.take_off_energy(total_power)
        mission_enrg += self.climb_energy(tow, cruise_altp)
        if power_system["energy_type"] != self.battery:
            mission_fuel += self.fuel_energy_ratio * mission_enrg / phd.fuel_heat(power_system["energy_type"])

        fuel, enrg, mission_lod, global_eff, mission_time = self.leg_fuel(tow, distance, cruise_altp, cruise_speed, speed_type, mtow, max_power, power_system)
        mission_fuel += fuel
        mission_enrg += enrg

        if power_system["energy_type"] != self.battery:
            ldw = tow - mission_fuel
        else:
            ldw = tow

        reserve_fuel = 0.
        reserve_enrg = 0.

        if reserve_data["fuel_factor"] > 0:
            reserve_fuel += reserve_data["fuel_factor"] * mission_fuel
            reserve_enrg += reserve_data["fuel_factor"] * mission_enrg
        if reserve_data["diversion_leg"] > 0:
            leg = reserve_data["diversion_leg"]
            diversion_altp = altitude_data["diversion"]
            lf, le, lod, eff, time = self.leg_fuel(ldw, leg, diversion_altp, cruise_speed, speed_type, mtow, max_power,power_system)
            reserve_fuel += lf
            reserve_enrg += le
        if reserve_data["holding_time"] > 0:
            time = reserve_data["holding_time"]
            holding_altp = altitude_data["holding"]
            speed = 1. * cruise_speed
            hf, he, lod = self.holding_fuel(ldw, time, holding_altp, speed, speed_type, mtow, max_power, power_system)
            reserve_fuel += hf
            reserve_enrg += he

        return {"tow": tow,
                "distance": distance,
                "total_fuel": mission_fuel + reserve_fuel,
                "mission_fuel": mission_fuel,
                "reserve_fuel": reserve_fuel,
                "total_enrg": mission_enrg + reserve_enrg,
                "mission_enrg": mission_enrg,
                "reserve_enrg": reserve_enrg,
                "mission_lod": mission_lod,
                "global_eff": global_eff,
                "mission_time": mission_time}

    def owe_performance(self, payload, mtow, range, cruise_speed, total_power, power_system, altitude_data, reserve_data):
        """Compute OWE from the point of view of mission
        energy_storage_mass contains the battery weight or tank weight for GH2 or LH2 storage
        """
        dict = self.total_fuel(mtow, range, cruise_speed, mtow, total_power, power_system, altitude_data, reserve_data)

        owe = mtow - payload - dict["total_fuel"]

        max_fuel = dict["total_fuel"] * self.max_fuel_factor    # Tanks are sized for max fuel
        max_enrg = dict["total_enrg"] * self.max_fuel_factor    # Battery is sized for max fuel

        return {"owe": owe,
                "total_energy": dict["total_enrg"],
                "total_fuel": dict["total_fuel"],
                "mission_fuel": dict["mission_fuel"],
                "reserve_fuel": dict["reserve_fuel"],
                "max_fuel": max_fuel,
                "mission_enrg": dict["mission_enrg"],
                "reserve_enrg": dict["reserve_enrg"],
                "max_energy": max_enrg,
                "payload": payload,
                "aerodynamic_efficiency": dict["mission_lod"],
                "propulsion_system_efficiency": dict["global_eff"]}

    def design_airplane(self, power_system, mission):
        """Perform the design of the aircraft with a target on Range
            * power_system : a dictionnary with 5 entries:
                - energy_type : "petrol","kerosene","gasoline","compressed_h2","liquid_h2","liquid_ch4","liquid_nh3" or "battery"
                - engine_type : "turbofan","turboprop","piston" or "emotor" (eletric motor)
                - engine_count : the number of engines
                - thruster_type: "fan" or "propeller"
                - bpr : bypass ratio in the case of a fan truster

            * design_mission : a dictionnary with 5 entries
                - category : "general" (ex: TB20), "commuter" (PC-6), "regional" (ATR72), "business" (Falcon2000), "short_medium" (A320) or "long_range" (B787).
                - npax : number of passengers. Used to determine the maximum payload.
                - range : the horizontal distance to cover for the design mission
                - speed : Cruise Mach number (adim). Can be obtained from True Air Speed, `utils/physical_data.py` with `mach_from_vtas()`
                - altitude : typical cruise altitude.

        """
        if "bpr" not in power_system.keys(): power_system["bpr"] = None

        category = mission["category"]
        design_range = mission["range"]
        cruise_speed = mission["speed"]

        if "altitude" in mission.keys():
            cruise_altp = mission["altitude"]
        else:
            cruise_altp = self.flight_altitudes[category][0]

        if "npax" in mission.keys() and "payload" not in mission.keys():
            npax = mission["npax"]
            mpax = self.get_pax_allowance(category)
            payload = npax * mpax + self.delta_payload
        elif "npax" in mission.keys() and "payload" in mission.keys():
            npax = mission["npax"]
            payload = mission["payload"]
            mpax = payload / npax
            self.mpax = mpax
        elif "npax" not in mission.keys() and "payload" in mission.keys():
            npax = 0
            mpax = self.get_pax_allowance(category)
            payload = mission["payload"]
        else:
            raise Exception("Key 'npax' or/and key 'payload' must be present in mission input dictionary")

        altitude_data = self.flight_altitude(category, cruise_altp)
        reserve_data = self.reserve_data(category)

        def fct(mtow, *args):
            total_power = self.ref_power(mtow)
            dict_p = self.owe_performance(payload, mtow, design_range, cruise_speed, total_power, power_system, altitude_data, reserve_data)
            dict_s = self.owe_structure(category, npax, mtow, design_range, total_power, dict_p["max_fuel"], dict_p["max_energy"], power_system)
            return dict_p["owe"] - dict_s["owe"]

        mtow_ini = 0.9e-3 * (payload/mpax) * design_range
        sol = root_find(
            fn=fct, solver=Newton(rtol=1e-6, atol=1e-6), y0=mtow_ini, max_steps=500,
            # adjoint=RecursiveCheckpointAdjoint(),
        )
        # sol = minimise(
        #     fn=fct, solver=BFGS(rtol=1e-8, atol=1e-8), y0=mtow_ini, max_steps=1000
        # )

        mtow = sol.value
        total_power = self.ref_power(mtow)
        max_power = total_power / power_system["engine_count"]
        dict_p = self.owe_performance(payload, mtow, design_range, cruise_speed, total_power, power_system, altitude_data, reserve_data)
        dict_s = self.owe_structure(category, npax, mtow, design_range, total_power, dict_p["max_fuel"], dict_p["max_energy"], power_system)

        dict = self.total_fuel(mtow, 0., cruise_speed, mtow, total_power, power_system, altitude_data, reserve_data)

        a1 = mtow - dict_s["owe"] - dict["total_fuel"]
        a2 = dict_p["payload"] * self.max_payload_factor
        payload_max = where(a1 < a2, a1, a2)
        mzfw = dict_s["owe"] + payload_max
        mlw = mzfw * self.mlw_factor

        storage_energy_density = dict_p["total_energy"] / (dict_s["energy_storage_mass"] + dict_p["total_fuel"])

        propulsion_power_density = total_power / (dict_s["propulsion_mass"] + dict_s["fuel_cell_system_mass"])

        return self.design_dict(npax, mpax, design_range, max_power, total_power, mtow, mzfw, mlw, payload_max,
                                power_system, mission, altitude_data, reserve_data, dict_p, dict_s,
                                storage_energy_density, propulsion_power_density)

    # def tune_design(self, power_system, mission):
    #     """Computes the following parameters :
    #         self.lift_to_drag
    #         self.mpax
    #         self.max_payload_factor
    #     so that the results of the design process exactly matches with the characteristics of a given aircraft
    #     in terms of Design Range, Passenger Capacity, MTOW, OWE, MZFW and Payload
    #     Required input data must be provided in the mission dictionary :
    #         "category":
    #         "npax":
    #         "range":
    #         "speed":
    #         "altitude":
    #         "mtow":
    #         "owe":
    #         "payload":
    #         "payload_max":
    #     If only one key among "npax" and "payload" is present, self.mpax will keep the value given by the model
    #     Il "payload_max" is not present, self.max_payload_factor will keep the value given by the model
    #     "category", "range" and "speed" are compulsory
    #     The parameter : self.stdm_factor can be estimated from these data only for FUEL airplanes.
    #     For battery airplanes, it must be given by the user
    #     """
    #     if "bpr" not in power_system.keys(): power_system["bpr"] = None
    #
    #     category = mission["category"]
    #     design_range = mission["range"]
    #     cruise_speed = mission["speed"]
    #
    #     if "altitude" in mission.keys():
    #         cruise_altp = mission["altitude"]
    #     else:
    #         cruise_altp = self.flight_altitudes[category][0]
    #
    #     mtow_target = mission["mtow"]
    #     owe_target = mission["owe"]
    #
    #     if "npax" in mission.keys() and "payload" not in mission.keys():
    #         npax = mission["npax"]
    #         mpax = self.get_pax_allowance(category)
    #         payload = npax * mpax + self.delta_payload
    #     elif "npax" in mission.keys() and "payload" in mission.keys():
    #         npax = mission["npax"]
    #         payload = mission["payload"]
    #         mpax = payload / npax
    #     elif "npax" not in mission.keys() and "payload" in mission.keys():
    #         npax = 0
    #         mpax = self.get_pax_allowance(category)
    #         payload = mission["payload"]
    #     else:
    #         raise Exception("Key 'npax' or/and key 'payload' must be present in mission input dictionary")
    #
    #     self.mpax = mpax
    #
    #     if "payload_max" in mission.keys():
    #         payload_max = mission["payload_max"]
    #         self.max_payload_factor = payload_max / payload
    #
    #     altitude_data = self.flight_altitude(category, cruise_altp)
    #     reserve_data = self.reserve_data(category)
    #
    #     def fct1(x):
    #         self.lift_to_drag = x[0]
    #         self.stdm_factor = x[1]
    #         total_power = self.ref_power(mtow_target)
    #         dict_p = self.owe_performance(payload, mtow_target, design_range, cruise_speed, total_power, power_system, altitude_data, reserve_data)
    #         dict_s = self.owe_structure(category, npax, mtow_target, design_range, total_power, dict_p["max_fuel"], dict_p["max_energy"], power_system)
    #         return [owe_target - dict_p["owe"], owe_target - dict_s["owe"]]
    #
    #     def fct2(lift_to_drag):
    #         self.lift_to_drag = lift_to_drag
    #         total_power = self.ref_power(mtow_target)
    #         dict_p = self.owe_performance(payload, mtow_target, design_range, cruise_speed, total_power, power_system, altitude_data, reserve_data)
    #         dict_s = self.owe_structure(category, npax, mtow_target, design_range, total_power, dict_p["max_fuel"], dict_p["max_energy"], power_system)
    #         return owe_target - dict_s["owe"]
    #
    #     ac = self.design_from_mtow(power_system, mission)
    #
    #     if power_system["energy_type"] != self.battery:
    #         x_ini = [ac["aerodynamic_efficiency"], ac["stdm_factor"]]
    #         output_dict = fsolve(fct1, x0=x_ini, args=(), full_output=True)
    #         if (output_dict[2] != 1): raise Exception("Convergence problem")
    #         self.lift_to_drag = output_dict[0][0]
    #         self.stdm_factor = output_dict[0][1]
    #     else:
    #         x_ini = ac["aerodynamic_efficiency"]
    #         output_dict = fsolve(fct2, x0=x_ini, args=(), full_output=True)
    #         if (output_dict[2] != 1): raise Exception("Convergence problem")
    #         self.lift_to_drag = output_dict[0][0]
    #
    #     total_power = self.ref_power(mtow_target)
    #     max_power = total_power / power_system["engine_count"]
    #     dict_p = self.owe_performance(payload, mtow_target, design_range, cruise_speed, total_power, power_system, altitude_data, reserve_data)
    #     dict_s = self.owe_structure(category, npax, mtow_target, design_range, total_power, dict_p["max_fuel"], dict_p["max_energy"], power_system)
    #
    #     if cruise_speed > 1:
    #         speed_type = "tas"
    #     else:
    #         speed_type = "mach"
    #
    #     dict = self.total_fuel(mtow_target, 0., cruise_speed, mtow_target, total_power, power_system, altitude_data, reserve_data)
    #
    #     payload_max = min(mtow_target - dict_s["owe"] - dict["total_fuel"], dict_p["payload"] * self.max_payload_factor)
    #     mzfw = dict_s["owe"] + payload_max
    #     mlw = mzfw * self.mlw_factor
    #
    #     storage_energy_density = dict_p["total_energy"] / (dict_s["energy_storage_mass"] + dict_p["total_fuel"])
    #
    #     propulsion_power_density = total_power / (dict_s["propulsion_mass"] + dict_s["fuel_cell_system_mass"])
    #
    #     return self.design_dict(npax, mpax, design_range, max_power, total_power, mtow_target, mzfw, mlw, payload_max,
    #                             power_system, mission, altitude_data, reserve_data, dict_p, dict_s,
    #                             storage_energy_density, propulsion_power_density)


    # def get_data_base(self, file_name):
    #     df, un = uda.read_db(file_name)
    #     return df

    def get_design_data(self, df, airplane, index="name"):
        power_system = []
        design_mission = []
        for ap in airplane:
            print()
            df1 = df[df[index]==ap].reset_index(drop=True).copy()
            print(df1)
            if df1.shape[0]==0:
                raise Exception("Unknown index : ", index)
            power_system.append({"energy_type": df1["energy_type"].iloc[0],
                                 "engine_count": df1["n_engine"].iloc[0],
                                 "engine_type": df1["engine_type"].iloc[0],
                                 "thruster_type": df1["thruster_type"].iloc[0],
                                 "bpr": df1["bpr"].iloc[0]})
            design_mission.append({"category": df1["airplane_type"].iloc[0],
                                   "npax": df1["n_pax"].iloc[0],
                                   "speed": df1["cruise_speed"].iloc[0],
                                   "range": df1["nominal_range"].iloc[0],
                                   "altitude": df1["cruise_altitude"].iloc[0]})
        return power_system, design_mission

    # # Maximum capacity and range per category
    # self.category = {self.general: {"capacity": 6, "distance": unit.m_km(500)},
    #                  self.commuter: {"capacity": 19, "distance": unit.m_km(1500)},
    #                  self.regional: {"capacity": 120, "distance": unit.m_km(4500)},
    #                  self.short_medium: {"capacity": 250, "distance": unit.m_km(8000)},
    #                  self.long_range: {"capacity": 400, "distance": unit.m_km(15000)}}


    def lod_model(self, fuselage_length, fuselage_width, wing_span, wing_area, htp_area, vtp_area,
                  cruise_altp, cruise_speed, mass, design={}, full_output=False):
        """Lift over Drag model construction based on geometrical parameters."""
        if design != {}:
            fuselage_shape = design["fuselage_shape"]       # 0 : standard, 1 : smooth (plastic gliders)
            fuselage_surface = design["fuselage_surface"]   # 0 : standard, 1 : very thin (beam)
            laminar_ratio = design["wing_laminar_ratio"]    # Ratio of the chords with laminar flow
            surface_quality = design["surface_quality"]     # 0 : standard, 1 : very clean (plastic gliders)
        else:
            fuselage_shape = 0.
            fuselage_surface = 0.
            laminar_ratio = 0.
            surface_quality = 0.

        wing_ar = wing_span ** 2 / wing_area
        wing_tr = 0.35
        wing_cref = wing_area / wing_span
        wing_croot = (2 * wing_area) / ((1 + wing_tr) * wing_span)

        htp_ar = 5.0
        htp_cref = sqrt(htp_area / htp_ar)

        vtp_ar = 1.7
        vtp_cref = sqrt(vtp_area / vtp_ar)

        r, gam, cp, cv = phd.gas_data()
        pamb, tamb, g = phd.atmosphere_g(cruise_altp)
        rho, sig = phd.air_density(pamb, tamb)
        if 1 < cruise_speed:
            mach = cruise_speed / phd.sound_speed(tamb)
        else:
            mach = cruise_speed

        fuselage_wa = 0.86 * (1. - 0.8*fuselage_surface) * pi * fuselage_width * fuselage_length
        wing_wa = 2 * (wing_area - wing_croot * fuselage_width)
        htp_wa = 1.63 * htp_area
        vtp_wa = 2 * vtp_area

        chord = [fuselage_length, wing_cref, htp_cref, vtp_cref]
        laminar_factor = [0, laminar_ratio, 0, 0]
        wet_area = [fuselage_wa, wing_wa, htp_wa, vtp_wa]
        form_factor = [1.05, 1.4, 1.4, 1.4]

        cz = (2 * mass * g) / (gam * pamb * wing_area * mach ** 2)
        re = phd.reynolds_number(pamb, tamb, mach)
        fac = (1. + 0.126 * mach ** 2)

        ac_nwa = 0.
        cxf = 0.
        for j in range(4):
            ael = chord[j]
            nwa = wet_area[j]
            frm = form_factor[j]
            # Drag model is based on flat plane turbulent friction drag
            if laminar_factor[j]==0.:
                cxf += frm * (3.913 / log(re * ael) ** 2.58) * nwa / wing_area
            else:
                lam = laminar_factor[j]
                cxf += frm * ( (3.913/log(re*ael)**2.58)*nwa
                              -(3.913/log(re*ael*lam)**2.58)*(nwa*lam)
                              +(1.328/sqrt(re*ael*lam)*(nwa*lam))
                              ) / wing_area
            ac_nwa += nwa

        # Parasitic drag (seals, antennas, sensors, ...)
        # -----------------------------------------------------------------------------------------------------------
        knwa = ac_nwa / 1000.
        kp = (0.0247 * knwa - 0.11) * knwa + 0.166  # Parasitic drag factor, Reymers ? ou Thorenbeck ?
        cx_par = cxf * kp * (1 - surface_quality)

        # Additional drag
        # -----------------------------------------------------------------------------------------------------------
        X = array([1.0, 1.5, 2.4, 3.3, 4.0, 5.0])
        Y = array([0.036, 0.020, 0.0075, 0.0025, 0., 0.])
        param = 3.45
        cx_tap = interp(param, X, Y) * (1 - fuselage_shape)

        # Total zero lift drag
        # -----------------------------------------------------------------------------------------------------------
        cx0 = (cxf + cx_par + cx_tap)

        # Induced drag
        # -----------------------------------------------------------------------------------------------------------
        ki_wing = (1.05 + (fuselage_width / wing_span) ** 2) / (pi * wing_ar)  # Reymers ? ou Thorenbeck ?
        cxi = ki_wing * cz ** 2  # Induced drag

        # Compressibility drag
        # -----------------------------------------------------------------------------------------------------------
        # Freely inspired from Korn equation
        cz_design = 0.5
        mach_div = mach + (0.03 + 0.1 * (cz_design - cz))  # Voir Picolas

        if 0.55 < mach:
            cxc = 0.0025 * exp(40. * (mach - mach_div))   # To have 10% slope and cxc = 25 cts at Mach div
        else:
            cxc = 0.

        # Sum up
        # -----------------------------------------------------------------------------------------------------------
        cx = cx0 + cxi + cxc
        lod = cz / cx
        if full_output:
            return {"lift_to_drag":lod, "cz":cz, "cx":cx, "cx0":cx0, "cxi":cxi, "cxc":cxc, "cxf":cxf, "cx_par":cx_par, "cx_tap":cx_tap}
        else:
            return lod

    # def get_app_speed(self, wing_area, mass):
    #     disa = 0.
    #     altp = unit.m_ft(0.)
    #     pamb, tamb, g = phd.atmosphere_g(altp, disa)
    #     rho = phd.gas_density(pamb, tamb)
    #     cl = self.get_cl_max_ld() / self.kvs1g_ld ** 2
    #     vapp = sqrt((2 * mass * g) / (rho * wing_area * cl))
    #     alpha, betha = self.tuner_app
    #     app_speed = alpha * vapp + betha  # Tuning versus data base
    #     return app_speed
    #
    # def get_tofl(self, total_power, wing_area, mass):
    #     disa = 0.
    #     altp = unit.m_ft(0.)
    #     pamb, tamb, g = phd.atmosphere_g(altp, disa)
    #     rho = phd.gas_density(pamb, tamb)
    #     sigma = rho / 1.225
    #     cl = self.get_cl_max_to() / self.kvs1g_to ** 2
    #     vtas35ft = sqrt((2 * mass * g) / (rho * wing_area * cl))
    #     fn_max = (total_power / (1.0 * vtas35ft)) * self.prop_eff
    #     ml_factor = mass ** 2 / (cl * fn_max * wing_area * sigma ** 0.8)  # Magic Line factor
    #     alpha, betha = self.tuner_to
    #     tofl = alpha * ml_factor + betha  # Magic line
    #     return tofl

    def design_dict(self, npax, mpax, nominal_range, max_power, total_power, mtow, mzfw, mlw, payload_max,
                          power_system, mission, altitude_data, reserve_data, dict_p, dict_s,
                          storage_energy_density, propulsion_power_density):
        return {"airplane_type": mission["category"],
                "npax": npax,
                "mpax": mpax,
                "delta_payload": self.delta_payload,
                "payload": dict_p["payload"],
                "nominal_range": nominal_range,
                "cruise_speed": mission["speed"],
                "nominal_time": nominal_range / mission["speed"],
                "altitude_data": altitude_data,
                "reserve_data": reserve_data,
                "mission_fuel": dict_p["mission_fuel"],
                "reserve_fuel": dict_p["reserve_fuel"],
                "total_fuel": dict_p["total_fuel"],
                "fuel_consumption": (dict_p["mission_fuel"]/dict_s["fuel_density"]) / npax / nominal_range,
                "mission_enrg": dict_p["mission_enrg"],
                "reserve_enrg": dict_p["reserve_enrg"],
                "total_energy": dict_p["total_energy"],
                "enrg_consumption": dict_p["mission_enrg"] / npax / nominal_range,

                "n_engine": power_system["engine_count"],
                "by_pass_ratio": power_system["bpr"],
                "max_power": max_power,
                "total_power": total_power,
                "power_system": power_system,

                "mtow": mtow,
                "mlw": mlw,
                "mzfw": mzfw,
                "payload_max": payload_max,
                "owe": dict_s["owe"],
                "op_item": dict_s["op_item"],
                "mwe": dict_s["mwe"],
                "furnishing": dict_s["furnishing"],
                "std_mwe": dict_s["std_mwe"],
                "propulsion_mass": dict_s["propulsion_mass"],
                "energy_storage_mass": dict_s["energy_storage_mass"],
                "fuel_cell_system_mass": dict_s["fuel_cell_system_mass"],
                "basic_mwe": dict_s["basic_mwe"],
                "stdm_factor": dict_s["stdm_factor"],
                "stdm_shift": dict_s["stdm_shift"],

                "storage_energy_density": storage_energy_density,
                "propulsion_power_density": propulsion_power_density,
                "aero_eff_factor": self.lod_factor,
                "aerodynamic_efficiency": dict_p["aerodynamic_efficiency"],
                "propulsion_system_efficiency": dict_p["propulsion_system_efficiency"],
                "structural_factor": dict_s["owe"] / mtow,

                # "pk_o_mass_min": self.get_pk_o_mass_min(dict_s["owe"]),
                # "pk_o_mass_max": self.get_pk_o_mass_max(dict_s["owe"]),
                "pk_o_mass": npax * nominal_range / dict_s["owe"],
                "pk_o_enrg": npax * nominal_range / dict_p["total_energy"]}

    def print_design(self, dict, name=None):

        print("")
        if name!=None:
            print(name)
        print("=============================================================================")
        print(" Airplane type = ", dict["airplane_type"])
        print(" Number of engine = ", "%.0f" % dict["n_engine"])
        print(" Number of passenger = ", "%.0f" % dict["npax"])
        print("")
        print(" Design range = ", "%.0f" % unit.convert_to("km", dict["nominal_range"]), " km")
        if dict["cruise_speed"] > 1:
            print(" Cruise speed = ", "%.1f" % unit.convert_to("km/h", dict["cruise_speed"]), " km/h")
        else:
            print(" Cruise Mach = ", "%.2f" % dict["cruise_speed"])
        print(" Cruise altitude = ", "%.1f" % unit.convert_to("ft", dict["altitude_data"]["mission"]), " ft")
        print("")
        print(" Nominal pax mass allowance = ", "%.1f" % dict["mpax"], " kg")
        print(" Nominal payload delta = ", "%.0f" % dict["delta_payload"], " kg")
        print(" Nominal payload = ", "%.0f" % dict["payload"], " kg")
        print(" Nominal mission time = ", "%.1f" % unit.h_s(dict["nominal_time"]), " h")
        print("")
        print(" Mission fuel = ", "%.0f" % dict["mission_fuel"], " kg")
        print(" Reserve fuel = ", "%.0f" % dict["reserve_fuel"], " kg")
        print(" Total fuel = ", "%.0f" % dict["total_fuel"], " kg")
        print(" Fuel consumption = ", "%.2f" % unit.convert_to("L", dict["fuel_consumption"] * unit.m_km(100)), " L/pax/100km")
        print("")
        print(" Mission energy = ", "%.0f" % unit.kWh_J(dict["mission_enrg"]), " kWh")
        print(" Reserve energy = ", "%.0f" % unit.kWh_J(dict["reserve_enrg"]), " kWh")
        print(" Total energy = ", "%.0f" % unit.kWh_J(dict["total_energy"]), " kWh")
        print(" Energy consumption = ", "%.2f" % unit.kWh_J(dict["enrg_consumption"] * unit.m_km(100)), " kWh/pax/100km")
        print("--------------------------------------------------------")
        print(" Engine type = ", dict["power_system"]["engine_type"])
        print(" Thruster type = ", dict["power_system"]["thruster_type"])
        print(" Energy source = ", dict["power_system"]["energy_type"])
        print("")
        print(" By Pass Ratio = ", dict["by_pass_ratio"])
        print(" Max power = ", "%.0f" % unit.kW_W(dict["max_power"]), " kW")
        print("")
        print(" Aerodynamic_efficiency_factor", "%.1f" % dict["aero_eff_factor"])
        print(" Aerodynamic_efficiency (L/D)", "%.2f" % dict["aerodynamic_efficiency"])
        print(" Propulsion_system_efficiency", "%.3f" % dict["propulsion_system_efficiency"])
        print(" Storage energy density = ", "%.0f" % unit.Wh_J(dict["storage_energy_density"]), " Wh/kg")
        print(" Propulsion power density = ", "%.2f" % unit.kW_W(dict["propulsion_power_density"]), " kW/kg")
        print(" Structural factor (OWE/MTOW) = ", "%.2f" % dict["structural_factor"])
        print("--------------------------------------------------------")
        print(" MTOW = ", "%.0f" % dict["mtow"], " kg")
        print("")
        print(" MZFW = ", "%.0f" % dict["mzfw"], " kg")
        print(" Payload max = ", "%.0f" % dict["payload_max"], " kg")
        print("")
        print(" OWE = ", "%.0f" % dict["owe"], " kg")
        print(" Operator items = ", "%.0f" % dict["op_item"], " kg")
        print("")
        print(" MWE = ", "%.0f" % dict["mwe"], " kg")
        print(" Furnishing = ", "%.0f" % dict["furnishing"], " kg")
        print(" Standard MWE = ", "%.0f" % dict["std_mwe"], " kg")
        print(" Propulsion mass = ", "%.0f" % dict["propulsion_mass"])
        print(" Energy storage mass = ", "%.0f" % dict["energy_storage_mass"])
        print(" Fuel cell system mass = ", "%.0f" % dict["fuel_cell_system_mass"])
        print(" Basic MWE = ", "%.0f" % dict["basic_mwe"], " kg")
        print(" MWE factor = ", "%.3f" % dict["stdm_factor"])
        print(" MWE sfift = ", "%.0f" % dict["stdm_shift"], " kg")
        print("--------------------------------------------------------")
        print(" Energy efficiency factor, P.K/E = ", "%.2f" % unit.convert_to("km/kWh", dict["pk_o_enrg"]), " pax.km/kWh")
        print(" Mass efficiency factor, P.K/M = ", "%.2f" % unit.convert_to("km/kg", dict["pk_o_mass"]), " pax.km/kg")
        print("")
        # print(" Maximum mass efficiency factor, P.K/M max = ", "%.2f" % unit.convert_to("km/kg", dict["pk_o_mass_max"]), " pax.km/kg")
        # print(" Minimum mass efficiency factor, P.K/M min = ", "%.2f" % unit.convert_to("km/kg", dict["pk_o_mass_min"]), " pax.km/kg")




if __name__ == '__main__':

    gam = GAM()


    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "../database/CADO_airplane_database_v1.0.xlsx"

    df,un = uda.read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

	# remove business jets from the database: their characterics do not match other civil transport aircraft.
    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    # df1 = df.copy()
    un1 = un.copy()

    #-------------------------------------------------------------------------------------------------------------------
    abs = "mtow"
    ord = "total_power"                           # Name of the new column

    df1[ord] = df1['max_power']*df1['n_engine']      # Add the new column to the dataframe
    un1[ord] = un1['max_power']                     # Add its unit

    title = "Generalized Power Index (kW)"
    order = [2, 1]
    dict = uda.do_regression(df1, un1, abs, ord, gam.colors, order, title)

    def guess_wing_area(total_power, mtow, mlw, tofl, vref):
        wa1 = mtow**2 / (2 * ((total_power/0.82) / (1.13*(vref/1.23))) * tofl)
        wa2 = 0.55 * (mlw / vref**2)
        wa3 = 0.5 * (wa1 + wa2)
        wa4 = (0.18007382 * wa3 + 17.26401586) * wa3 + 3.55650928
        return wa4


    # Build basic mass
    #-------------------------------------------------------------------------------------------------------------------
    nap = df1.shape[0]

    thruster = {gam.piston:gam.propeller, gam.turboprop:gam.propeller, gam.turbofan:gam.fan}

    ord = "basic_mwe"

    df1[ord] = df1['owe']
    un1[ord] = un1['owe']

    for n in range(nap):
        # print(df1["name"][n])
        npax = df1["n_pax"][n]
        distance = df1["nominal_range"][n]
        cruise_speed = df1["cruise_speed"][n]
        airplane_type = df1["airplane_type"][n]
        n_engine = df1["n_engine"][n]
        max_power = df1["max_power"][n]
        total_power = max_power*n_engine
        engine_type = df1["engine_type"][n]
        bpr = df1["bpr"][n]
        thruster_type = thruster[engine_type]
        mtow = df1["mtow"][n]

        owe = df1["owe"][n]

        tofl = df1["tofl"][n]
        vapp = df1["approach_speed"][n]

        furnishing = gam.furnishing(npax, airplane_type)
        operator_items = gam.op_item(npax, distance)

        power_system = {"energy_type":gam.petrol, "engine_count": n_engine, "engine_type":engine_type, "thruster_type":thruster_type, "bpr":bpr}
        propulsion_mass,_ = gam.propulsion_mass(power_system, total_power)

        df1.at[n, ord] = owe - propulsion_mass - furnishing - operator_items

    abs = "mtow"

    # dict = uda.draw_reg(df1, un1, abs, ord, [[0,amp],[0,amp]], gam.colors)

    title = "Regression Basic_MWE - MTOW"
    order = [2, 1]
    dict_smwe = uda.do_regression(df1, un1, abs, ord, gam.colors, order, title=title)

    # print("---------------------------------------------------")
    # print(list(df1["mtow"]))
    # print(list(df1[ord]))



    # Design Analysis
    #-------------------------------------------------------------------------------------------------------------------

    nap = df1.shape[0]

    var = "mtow"
    # var = "nominal_range"

    df1["estimated_"+var] = df1[var]
    un1["estimated_"+var] = un1[var]
    un1["estimated_owe"] = un1["owe"]

    thruster = {gam.piston:gam.propeller, gam.turboprop:gam.propeller, gam.turbofan:gam.fan}

    abs = var
    ord = "estimated_"+var

    for n in range(nap):

        power_system = {"energy_type":gam.petrol,
                        "engine_count":df1["n_engine"][n],
                        "engine_type":df1["engine_type"][n],
                        "thruster_type":thruster[df1["engine_type"][n]],
                        "bpr":df1["bpr"][n]}

        category = df1["airplane_type"][n]

        design_mission = {"category": category,
                          "npax": df1["n_pax"][n],
                          "range": df1["nominal_range"][n],
                          "mtow": df1["mtow"][n],
                          "speed": df1["cruise_speed"][n],
                          "altitude": gam.flight_altitude(category)["mission"]}

        if var=="mtow":
            ac_dict = gam.design_airplane(power_system, design_mission)
        elif var=="nominal_range":
            ac_dict = gam.design_from_mtow(power_system, design_mission)

        df1.at[n, "estimated_nominal_range"] = ac_dict["nominal_range"]
        df1.at[n, "estimated_mtow"] = ac_dict["mtow"]
        df1.at[n, "estimated_owe"] = ac_dict["owe"]

        df1.at[n, "error_mtow"] = (ac_dict["mtow"] - df1["mtow"][n]) / df1["mtow"][n]

        if 0.25 < abs(df1["error_mtow"][n]):
            print(df1["name"][n], "       ", "%.2f" % df1["error_mtow"][n])

        # print("%.0f" % (ac_dict[abs] / 1000))

    amp = {"mtow":5e5, "nominal_range":20e6}.get(var)

    title = "Estimated MTOW vs Published MTOW"
    dict = uda.draw_reg(df1, un1, abs, ord, [[0,amp],[0,amp]], gam.colors, title=title)

    title = "Estimated OWE vs Published OWE"
    dict = uda.draw_reg(df1, un1, "owe", "estimated_owe", [[0,amp],[0,amp]], gam.colors, title=title)


    plt.hist(df1["error_mtow"], bins=9)
    plt.suptitle("PDF of relative error MTOW", fontsize=16)
    plt.ylabel("Number of airplane (total: 227)", fontsize=14)
    plt.xlabel("Relative error", fontsize=14)
    plt.grid(True)
    plt.show()

    plt.hist(df1["error_mtow"], cumulative=True)
    plt.suptitle("CDF of relative error MTOW", fontsize=16)
    plt.ylabel("Number of airplane (total: 227)", fontsize=14)
    plt.xlabel("Relative error", fontsize=14)
    plt.grid(True)
    plt.show()



    # order = [1]
    # dict_smwe = uda.do_regression(df1, un1, abs, ord, gam.colors, order)




