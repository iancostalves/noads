"""This is an example of use of the Generic Airplane Model toolbox"""

from generic_airplane_model import GAM # import the toolbox
from utils import unit

# ----- INITIALIZATION
gam=GAM()
# Print all data used in the model
gam.print_model_data()

# ----- AIRPLANE DESIGN
# Design of an "A320-neo like" airplane.
# The model needs two inputs : the propulsion system and the design mission. See
# Parameters must be provided in SI units
power_system = {"energy_type": "kerosene",
               "engine_count": 2,
                "engine_type": "turbofan",
              "thruster_type": "fan",
                        "bpr": 12.5}
design_mission = {"category": "short_medium" , # Airplane category (determines, among others, the amount of furnishing)
                      "npax": 165 , # maximum number of passenger
                     "range": 5800000, # range of the design mission (meters)
                     "speed": 0.78, # cruise design Mach number (non dimensional)
                  "altitude": unit.convert_from("ft", 35000)} # cruise altitude for the design mission (typically 35000 ft for jet aircraft and less than 20000 ft for turboprop)
# Run design procedure
my_airplane = gam.design_airplane(power_system, design_mission)
# Print the resulting mass breakdown, mission fuel etc ...
gam.print_design(my_airplane, name="A320-neo-like")

# ----- EXAMPLE OF AIRPLANE PERFORMANCE
tow = my_airplane['mtow']*0.9              # (kg) let us assume the airplane takes-off at 90% of its maximum mass
distance = 2000e3                          # (m) flight distance
cruise_speed = my_airplane['cruise_speed'] # (-) cruise Mach number (or True Airspeed (m/s) )
mtow = my_airplane['mtow']                 # (kg) Maximum Take-Off Weight
total_power = my_airplane['total_power']   # (W) Total propulsive power
altitude_data = my_airplane['altitude_data'] # Cruise altitude, diversion altitude and holding altitude
reserve_data = my_airplane['reserve_data']   # Typical fuel reserves data

total_fuel_dict = gam.total_fuel(tow,distance,cruise_speed, mtow, total_power, power_system, altitude_data,reserve_data)
print("----- Airplane fuel consumption on custom mission")
for key,val in total_fuel_dict.items():
    print(key,"=",val)

