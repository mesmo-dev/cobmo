"""Example run script for using the building model."""

import numpy as np
import os
import pandas as pd

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'scenario_default'

# Set results path and create the directory.
results_path = os.path.join(cobmo.config.results_path, 'run_example_' + cobmo.config.timestamp)
os.mkdir(results_path)

# Obtain a connection to the database.
database_connection = cobmo.database_interface.connect_database()

# Define the building model (main function of the CoBMo toolbox).
# - Generates the building model for given `scenario_name` based on the building definitions in the `data` directory.
building = cobmo.building_model.BuildingModel(scenario_name)

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
    np.ones((len(building.set_timesteps), len(building.set_controls))),
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
controller = cobmo.optimization_problem.OptimizationProblem(
    database_connection=database_connection,
    building=building
)
(
    control_timeseries_controller,
    state_timeseries_controller,
    output_timeseries_controller,
    operation_cost,
    investment_cost,  # Zero when running (default) operation problem.
    storage_size  # Zero when running (default) operation problem.
) = controller.solve()

# Save controller timeseries to CSV for debugging.
control_timeseries_controller.to_csv(os.path.join(results_path, 'control_timeseries_controller.csv'))
state_timeseries_controller.to_csv(os.path.join(results_path, 'state_timeseries_controller.csv'))
output_timeseries_controller.to_csv(os.path.join(results_path, 'output_timeseries_controller.csv'))

# Print operation cost for debugging.
print("operation_cost = {}".format(operation_cost))

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
print("error_summary = \n{}".format(error_summary))

# Calculate total demand for benchmarking.
total_demand = (
    output_timeseries_controller.loc[:, output_timeseries_controller.columns.str.contains('electric_power')].sum().sum()
    * pd.to_timedelta(building.set_timesteps[1] - building.set_timesteps[0]).seconds / 3600.0 / 1000.0  # W in kWh.
)
total_demand_year = (
    total_demand
    * (pd.to_timedelta('1y') / pd.to_timedelta(building.set_timesteps[1] - building.set_timesteps[0]))
    # Theoretical number of time steps in a year.
    / len(building.set_timesteps)
    # Actual number of time steps.
)
total_demand_year_per_area = (
    total_demand_year
    / building.building_zones['zone_area'].apply(building.parse_parameter).sum()  # kWh to kWh/m2.
)

# Print total demand for benchmarking.
print("total_demand = {}".format(total_demand))
print("total_demand_year = {}".format(total_demand_year))
print("total_demand_year_per_area = {}".format(total_demand_year_per_area))

# Print results path for debugging.
print("Results are stored in: " + results_path)
