"""Example run script for the radiator test case."""

import hvplot
import hvplot.pandas
import numpy as np
import os
import pandas as pd

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'validation_1zone_radiator'

# Set results path and create the directory.
results_path = os.path.join(cobmo.config.results_path, 'run_example_' + cobmo.config.timestamp)
os.mkdir(results_path)

# Obtain a connection to the database.
database_connection = cobmo.database_interface.connect_database()

# Define the building model (main function of the CoBMo toolbox).
# - Generates the building model for given `scenario_name` based on the building definitions in the `data` directory.
building = cobmo.building_model.BuildingModel(scenario_name)

# Save building model matrices to CSV for debugging.
building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
building.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

# Run controller.
controller = cobmo.optimization_problem.OptimizationProblem(
    building
)
(
    control_vector_controller,
    state_vector_controller,
    output_vector_controller,
    operation_cost,
    investment_cost,  # Zero when running (default) operation problem.
    storage_size  # Zero when running (default) operation problem.
) = controller.solve()

# Save controller timeseries to CSV for debugging.
control_vector_controller.to_csv(os.path.join(results_path, 'control_vector_controller.csv'))
state_vector_controller.to_csv(os.path.join(results_path, 'state_vector_controller.csv'))
output_vector_controller.to_csv(os.path.join(results_path, 'output_vector_controller.csv'))

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
    output_vector_controller.loc[
        :, output_vector_controller.columns.str.contains('thermal_power')
    ].stack().rename('thermal_power').reset_index()
).hvplot.step(
    x='time',
    y='thermal_power',
    by='output_name',
    **hvplot_default_options
)
zone_temperature_plot = (
    output_vector_controller.loc[
        :, output_vector_controller.columns.isin(
            building.zones['zone_name'] + '_temperature'
        )
    ].stack().rename('zone_temperature').reset_index()
).hvplot.line(
    x='time',
    y='zone_temperature',
    by='output_name',
    **hvplot_default_options
)
radiator_water_temperature_plot = (
    state_vector_controller.loc[
        :, state_vector_controller.columns.isin(
            building.zones['zone_name'] + '_radiator_water_mean_temperature'
        )
    ].stack().rename('radiator_water_temperature').reset_index()
).hvplot.line(
    x='time',
    y='radiator_water_temperature',
    by='state_name',
    **hvplot_default_options
)
radiator_hull_temperature_plot = (
    state_vector_controller.loc[
        :, state_vector_controller.columns.isin(
            pd.concat([
                building.zones['zone_name'] + '_radiator_hull_front_temperature',
                building.zones['zone_name'] + '_radiator_hull_rear_temperature'
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

# Print results path for debugging.
print("Results are stored in: " + results_path)
