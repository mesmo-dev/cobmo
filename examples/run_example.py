"""Example run script for using the building model."""

import numpy as np
import os
import pandas as pd

import cobmo.building
import cobmo.controller
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'scenario_default'

# Set results path and create the directory.
results_path = os.path.join(cobmo.config.results_path, 'run_example_' + cobmo.config.timestamp)
os.mkdir(results_path)

# Obtain a connection to the database.
conn = cobmo.database_interface.connect_database()

# Define the building model (main function of the CoBMo toolbox).
# - Generates the building model for given `scenario_name` based on the building definitions in the `data` directory.
building = cobmo.building.Building(
    conn=conn,
    scenario_name=scenario_name
)

# Define augemented state space model matrices.
# TODO: Check if there is any usage for the augmented state space model.
building.define_augmented_model()

# Save building model matrices to CSV for debugging.
building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
building.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

# Define initial state and control timeseries.
state_initial = building.set_state_initial
control_timeseries_simulation = pd.DataFrame(
    np.random.rand(len(building.set_timesteps), len(building.set_controls)),
    building.set_timesteps,
    building.set_controls
)

# Run simulation.
(
    state_timeseries_simulation,
    output_timeseries_simulation
) = building.simulate(
    state_initial=state_initial,
    control_timeseries=control_timeseries_simulation
)

# Save simulation timeseries to CSV for debugging.
control_timeseries_simulation.to_csv(os.path.join(results_path, 'control_timeseries_simulation.csv'))
state_timeseries_simulation.to_csv(os.path.join(results_path, 'state_timeseries_simulation.csv'))
output_timeseries_simulation.to_csv(os.path.join(results_path, 'output_timeseries_simulation.csv'))

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

# Save controller timeseries to CSV for debugging.
control_timeseries_controller.to_csv(os.path.join(results_path, 'control_timeseries_controller.csv'))
state_timeseries_controller.to_csv(os.path.join(results_path, 'state_timeseries_controller.csv'))
output_timeseries_controller.to_csv(os.path.join(results_path, 'output_timeseries_controller.csv'))

# Print controller objective value for debugging.
print("obj_optimum=")
print(obj_optimum)

# Run error calculation function.
(
    error_summary,
    error_timeseries
) = cobmo.utils.calculate_error(
    output_timeseries_simulation.loc[:, output_timeseries_controller.columns.str.contains('temperature')],
    output_timeseries_controller.loc[:, output_timeseries_controller.columns.str.contains('temperature')]
)  # Note: These are exemplary inputs.

# Save error outputs to CSV for debugging.
error_timeseries.to_csv(os.path.join(results_path, 'error_timeseries.csv'))
error_summary.to_csv(os.path.join(results_path, 'error_summary.csv'))

# Print error summary for debugging.
print("error_summary=")
print(error_summary)

# Print results path for debugging.
print("Results are stored in: " + results_path)
