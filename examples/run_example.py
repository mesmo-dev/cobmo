"""Example run script for using the building model."""

import hvplot
import hvplot.pandas
import numpy as np
import os
import pandas as pd

import cobmo.building
import cobmo.controller
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'validation_1zone_radiator'

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
controller = cobmo.controller.Controller(
    conn=conn,
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

# Hvplot has no default options.
# Workaround: Pass this dict to every new plot.
hvplot_default_options = dict(width=1500, height=300)

# Generate plot handles.
irradiation_plot = (
    building.disturbance_timeseries.loc[
        :, building.disturbance_timeseries.columns.str.contains('irradiation')
    ].stack().rename('irradiation').reset_index()
).hvplot.line(
    x='time',
    y='irradiation',
    by='disturbance_name',
    **hvplot_default_options
)
ambient_air_temperature_plot = (
    building.disturbance_timeseries['ambient_air_temperature'].rename('ambient_air_temperature').reset_index()
).hvplot.line(
    x='time',
    y='ambient_air_temperature',
    **hvplot_default_options
)
thermal_power_plot = (
    output_timeseries_controller.loc[
        :, output_timeseries_controller.columns.str.contains('thermal_power')
    ].stack().rename('thermal_power').reset_index()
).hvplot.step(
    x='time',
    y='thermal_power',
    by='output_name',
    **hvplot_default_options
)
zone_temperature_plot = (
    output_timeseries_controller.loc[
        :, output_timeseries_controller.columns.isin(
            building.building_zones['zone_name'] + '_temperature'
        )
    ].stack().rename('zone_temperature').reset_index()
).hvplot.line(
    x='time',
    y='zone_temperature',
    by='output_name',
    **hvplot_default_options
)
radiator_water_temperature_plot = (
    state_timeseries_controller.loc[
        :, state_timeseries_controller.columns.isin(
            building.building_zones['zone_name'] + '_radiator_water_mean_temperature'
        )
    ].stack().rename('radiator_water_temperature').reset_index()
).hvplot.line(
    x='time',
    y='radiator_water_temperature',
    by='state_name',
    **hvplot_default_options
)
radiator_hull_temperature_plot = (
    state_timeseries_controller.loc[
        :, state_timeseries_controller.columns.isin(
            pd.concat([
                building.building_zones['zone_name'] + '_radiator_hull_front_temperature',
                building.building_zones['zone_name'] + '_radiator_hull_rear_temperature'
            ])
        )
    ].stack().rename('radiator_hull_temperature').reset_index()
).hvplot.line(
    x='time',
    y='radiator_hull_temperature',
    by='state_name',
    **hvplot_default_options
)

# Define layout and labels / render plots.
hvplot.show(
    (
        irradiation_plot
        + ambient_air_temperature_plot
        + thermal_power_plot
        + zone_temperature_plot
        + radiator_water_temperature_plot
        + radiator_hull_temperature_plot
    ).redim.label(
        time="Date / time",
        irradiation="Irradiation [W/m²]",
        ambient_air_temperature="Ambient air temp. [°C]",
        thermal_power="Thermal power [W]",
        zone_temperature="Zone temperature [°C]",
        radiator_water_temperature="Radiator water temperature [°C]",
        radiator_hull_temperature="Radiator hull temperature [°C]",
    ).cols(1),
    # Plots open in browser and are also stored in results directory.
    filename=os.path.join(results_path, 'example_plots.html')
)

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
