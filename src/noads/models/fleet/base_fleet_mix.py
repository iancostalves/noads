
# Create Propulsion systems
from gemseo import generate_coupling_graph

from AeroMAX.aviation_mix import battery
from AeroMAX.aviation_mix import drop_in
from AeroMAX.aviation_mix import lh2
from noads.models import AircraftOperation

from noads.models import PropulsionSystem

# Propulsion systems
from noads.models.fleet.fleet import Fleet
from noads.models.fleet.fleet import FleetAssembly

turbofan = PropulsionSystem("turbofan", {drop_in: 1.0})
turboprop = PropulsionSystem("turboprop", {drop_in: 1.0})
hybrid = PropulsionSystem("hybrid_electric", {drop_in: 0.6, battery: 0.4})
e_prop = PropulsionSystem("electric_propulsion", {battery: 1.0})
lh2_fuelcell = PropulsionSystem("lh2_fuel_cell", {lh2: 1.0})
lh2_burn = PropulsionSystem("lh2_burn", {lh2: 1.0})


# AircraftOperation from examples_fleet
sr_old_conventional = AircraftOperation("SR_OLD_CONVENTIONAL", turbofan, energy_per_ask=110.8 / 73.2 * 0.824, entry_into_service=1970)
sr_recent_conventional = AircraftOperation("SR_RECENT_CONVENTIONAL", turbofan, energy_per_ask=84.2 / 73.2 * 0.824, recent=True, entry_into_service=2007.13)
sr_new1_conventional = AircraftOperation("SR_NEW_NARROW_BODY_1", turbofan, energy_per_ask=0.85 * sr_recent_conventional.energy_per_ask, entry_into_service=2035)
sr_new2_conventional = AircraftOperation("SR_NEW_NARROW_BODY_2", turbofan, energy_per_ask=0.7 * sr_recent_conventional.energy_per_ask, entry_into_service=2045)
new1_turboprop = AircraftOperation("NEW_TURBOPROP_1", turbofan, energy_per_ask=0.8 * sr_recent_conventional.energy_per_ask, entry_into_service=2030)
new2_turboprop = AircraftOperation("NEW_TURBOPROP_2", turbofan, energy_per_ask=0.65 * sr_recent_conventional.energy_per_ask, entry_into_service=2035)
hydrogen_examples = AircraftOperation("HYDROGEN_AIRCRAFT", lh2_fuelcell, energy_per_ask=1.1 * sr_recent_conventional.energy_per_ask, entry_into_service=2035)

mr_old_conventional = AircraftOperation("MR_OLD_CONVENTIONAL", turbofan, energy_per_ask=81.4 / 73.2 * 0.824, entry_into_service=1970)
mr_recent_conventional = AircraftOperation("MR_RECENT_CONVENTIONAL", turbofan, energy_per_ask=62.0 / 73.2 * 0.824, recent=True, entry_into_service=2010.35)
mr_new1_conventional = AircraftOperation("MR_NEW_NARROW_BODY_1", turbofan, energy_per_ask=0.85 * mr_recent_conventional.energy_per_ask, entry_into_service=2035)
mr_new2_conventional = AircraftOperation("MR_NEW_NARROW_BODY_2", turbofan, energy_per_ask=0.7 * mr_recent_conventional.energy_per_ask, entry_into_service=2045)

lr_old_conventional = AircraftOperation("LR_OLD_CONVENTIONAL", turbofan, energy_per_ask=96.65 / 73.2 * 0.824, entry_into_service=1970)
lr_recent_conventional = AircraftOperation("LR_RECENT_CONVENTIONAL", turbofan, energy_per_ask=73.45 / 73.2 * 0.824, recent=True, entry_into_service=2009.36)
lr_new1_conventional = AircraftOperation("LR_NEW_NARROW_BODY_1", turbofan, energy_per_ask=0.85 * lr_recent_conventional.energy_per_ask, entry_into_service=2035)
lr_new2_conventional = AircraftOperation("LR_NEW_NARROW_BODY_2", turbofan, energy_per_ask=0.7 * lr_recent_conventional.energy_per_ask, entry_into_service=2045)


# Fleet categories from examples_fleet
examples_short_range = Fleet("ShortRange", [
    sr_old_conventional,
    sr_recent_conventional,
    sr_new1_conventional,
    sr_new2_conventional,
    new1_turboprop,
    new2_turboprop,
    hydrogen_examples,
])
examples_medium_range = Fleet("MediumRange", [
    mr_old_conventional,
    mr_recent_conventional,
    mr_new1_conventional,
    mr_new2_conventional,
])
examples_long_range = Fleet("LongRange", [
    lr_old_conventional,
    lr_recent_conventional,
    lr_new1_conventional,
    lr_new2_conventional,
])
examples_assembly = FleetAssembly(
    [examples_short_range, examples_medium_range, examples_long_range],
    endogenous_fleet_renewal=False,
)
generate_coupling_graph(examples_assembly.to_disciplines(),
    "../../../AeroMAX/figures/examples_fleet.pdf"
)
# print("EXAMPLES:")
# print("Required inputs", [
#     name for name in examples_assembly.jax_chain.input_grammar.names
#     if name not in examples_assembly.jax_chain.output_grammar.names
# ])
# print("Required couplings", [
#     name for name in examples_assembly.jax_chain.input_grammar.names
#     if name in examples_assembly.jax_chain.output_grammar.names
# ])
# print("Outputs:", [name for name in examples_assembly.jax_chain.output_grammar.names])

# AircraftOperation from MEA application
sr_old_conventional_mea = AircraftOperation("SR_OLD_CONVENTIONAL", turbofan, energy_per_ask=110.8 / 73.2 * 0.824, entry_into_service=1970, lifetime=20)
sr_recent_conventional_mea = AircraftOperation("SR_RECENT_CONVENTIONAL", turbofan, energy_per_ask=84.2 / 73.2 * 0.824, recent=True, entry_into_service=2007.13, lifetime=20)
sr_new_conventional_mea = AircraftOperation("SR_NEW_NARROW_BODY", turbofan, energy_per_ask=0.80 * sr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2035)
sr_electric_mea = AircraftOperation("SR_ELECTRIC_AIRCRAFT", e_prop, energy_per_ask=1.5 * sr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2040)
sr_hybrid_elec_mea = AircraftOperation("SR_HYBRID_ELECTRIC", hybrid, energy_per_ask=0.9 * sr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2030)
sr_hydrogen_mea = AircraftOperation("SR_HYDROGEN_AIRCRAFT", lh2_fuelcell, energy_per_ask=1.05 * sr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2035)

mr_old_conventional_mea = AircraftOperation("MR_OLD_CONVENTIONAL", turbofan, energy_per_ask=81.4 / 73.2 * 0.824, entry_into_service=1970, lifetime=20)
mr_recent_conventional_mea = AircraftOperation("MR_RECENT_CONVENTIONAL", turbofan, energy_per_ask=62.0 / 73.2 * 0.824, recent=True, entry_into_service=2010.35, lifetime=20)
mr_new_conventional_mea = AircraftOperation("MR_NEW_NARROW_BODY", turbofan, energy_per_ask=0.8 * mr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2035)
mr_hybrid_elec_mea = AircraftOperation("MR_HYBRID_ELECTRIC", hybrid, energy_per_ask=0.9 * mr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2030)
mr_hydrogen_mea = AircraftOperation("MR_HYDROGEN_AIRCRAFT", lh2_fuelcell, energy_per_ask=1.05 * mr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2035)

lr_old_conventional_mea = AircraftOperation("LR_OLD_CONVENTIONAL", turbofan, energy_per_ask=96.65 / 73.2 * 0.824, entry_into_service=1970, lifetime=20)
lr_recent_conventional_mea = AircraftOperation("LR_RECENT_CONVENTIONAL", turbofan, energy_per_ask=73.45 / 73.2 * 0.824, recent=True, entry_into_service=2009.36, lifetime=20)
lr_new_conventional_mea = AircraftOperation("LR_NEW_NARROW_BODY", turbofan, energy_per_ask=0.80 * lr_recent_conventional.energy_per_ask, lifetime=20, entry_into_service=2035)

mea_short_range = Fleet("ShortRange", [
    sr_old_conventional_mea,
    sr_recent_conventional_mea,
    sr_new_conventional_mea,
    sr_electric_mea,
    sr_hybrid_elec_mea,
    sr_hydrogen_mea,
])
mea_medium_range = Fleet("MediumRange", [
    mr_old_conventional_mea,
    mr_recent_conventional_mea,
    mr_new_conventional_mea,
    mr_hybrid_elec_mea,
    mr_hydrogen_mea,
])
mea_long_range = Fleet("LongRange", [
    lr_old_conventional_mea,
    lr_recent_conventional_mea,
    lr_new_conventional_mea,
])
mea_assembly = FleetAssembly(
    [mea_short_range, mea_medium_range, mea_long_range],
    endogenous_fleet_renewal=False,
)

generate_coupling_graph(mea_assembly.to_disciplines(),
    "../../../AeroMAX/figures/mea_application_fleet.pdf"
)
# print("MEA:")
# print("Required inputs", [
#     name for name in mea_assembly.jax_chain.input_grammar.names
#     if name not in mea_assembly.jax_chain.output_grammar.names
# ])
# print("Required couplings", [
#     name for name in mea_assembly.jax_chain.input_grammar.names
#     if name in mea_assembly.jax_chain.output_grammar.names
# ])
# print("Outputs:", [name for name in mea_assembly.jax_chain.output_grammar.names])
