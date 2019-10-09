"""Example run script for using the building model."""

import numpy as np
import pandas as pd

import cobmo.building
import cobmo.controller
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'scenario_default'

# Obtain a connection to the database.
conn = cobmo.database_interface.connect_database()

building = cobmo.building.Building(
    conn=conn,
    scenario_name=scenario_name
)

# Define initial state and control timeseries.
state_initial = building.set_state_initial
control_timeseries_simulation = pd.DataFrame(
    np.random.rand(len(building.set_timesteps), len(building.set_controls)),
    building.set_timesteps,
    building.set_controls
)

# Define augemented state space model matrices.
building.define_augmented_model()

# Run simulation.
(
    state_timeseries_simulation,
    output_timeseries_simulation
) = building.simulate(
    state_initial=state_initial,
    control_timeseries=control_timeseries_simulation
)

# Outputs for debugging.
print("-----------------------------------------------------------------------------------------------------------")
print("building.state_matrix=")
print(building.state_matrix)
print("-----------------------------------------------------------------------------------------------------------")
print("building.control_matrix=")
print(building.control_matrix)
print("-----------------------------------------------------------------------------------------------------------")
print("building.disturbance_matrix=")
print(building.disturbance_matrix)
print("-----------------------------------------------------------------------------------------------------------")
print("building.state_output_matrix=")
print(building.state_output_matrix)
print("-----------------------------------------------------------------------------------------------------------")
print("building.control_output_matrix=")
print(building.control_output_matrix)
print("-----------------------------------------------------------------------------------------------------------")
print("building.disturbance_output_matrix=")
print(building.disturbance_output_matrix)
print("-----------------------------------------------------------------------------------------------------------")
print("control_timeseries_simulation=")
print(control_timeseries_simulation)
print("-----------------------------------------------------------------------------------------------------------")
print("building.disturbance_timeseries=")
print(building.disturbance_timeseries)
print("-----------------------------------------------------------------------------------------------------------")
print("state_timeseries_simulation=")
print(state_timeseries_simulation)
print("-----------------------------------------------------------------------------------------------------------")
print("output_timeseries_simulation=")
print(output_timeseries_simulation)
print("-----------------------------------------------------------------------------------------------------------")

# Run controller.
controller = cobmo.controller.Controller(
    conn=conn,
    building=building
)
(
    control_timeseries_controller,
    state_timeseries_controller,
    output_timeseries_controller,
    obj_optimum
) = controller.solve()

# Outputs for debugging.
print("-----------------------------------------------------------------------------------------------------------")
print("control_timeseries_controller=")
print(control_timeseries_controller)
print("-----------------------------------------------------------------------------------------------------------")
print("state_timeseries_controller=")
print(state_timeseries_controller)
print("-----------------------------------------------------------------------------------------------------------")
print("output_timeseries_controller=")
print(output_timeseries_controller)
print("-----------------------------------------------------------------------------------------------------------")
print("obj_optimum=")
print(obj_optimum)
print("-----------------------------------------------------------------------------------------------------------")

# Run error calculation function.
(
    error_summary,
    error_timeseries
) = cobmo.utils.calculate_error(
    output_timeseries_simulation.loc[:, output_timeseries_controller.columns.str.contains('temperature')],
    output_timeseries_controller.loc[:, output_timeseries_controller.columns.str.contains('temperature')]
)  # Note: These are exemplary inputs.

# Outputs for debugging.
print("-----------------------------------------------------------------------------------------------------------")
print("error_timeseries=")
print(error_timeseries)
print("-----------------------------------------------------------------------------------------------------------")
print("error_summary=")
print(error_summary)
print("-----------------------------------------------------------------------------------------------------------")
