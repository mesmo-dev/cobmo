"""Run script for building model validation."""

import hvplot
import hvplot.pandas
import numpy as np
import os
import pandas as pd

import cobmo.building
import cobmo.config
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'validation_1zone_no_window_no_mass'

# Set results path and create the directory.
results_path = os.path.join(cobmo.config.results_path, 'run_validation_' + cobmo.config.timestamp)
os.mkdir(results_path)

# Obtain a connection to the database.
conn = cobmo.database_interface.connect_database()

# Define the building model (main function of the CoBMo toolbox).
# - Generates the building model for given `scenario_name` based on the building definitions in the `data` directory.
building = cobmo.building.Building(
    conn=conn,
    scenario_name=scenario_name
)

# Define initial state and control timeseries.
state_initial = pd.Series(
    np.concatenate([
        24.0  # in °C
        * np.ones(sum(building.set_states.str.contains('temperature')))
    ]),
    building.set_states
)
control_timeseries_simulation = pd.DataFrame(
    0.0,
    building.set_timesteps,
    building.set_controls
)
control_timeseries_simulation.loc[:]['zone_1_generic_cool_thermal_power'] = 200.0

# Save building model matrices to CSV for debugging.
building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
building.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

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

# Load validation data.
output_timeseries_validation = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.normpath(__file__)), '..', 'data', 'validation', building.scenario_name + '.csv'
    ),
    index_col='time',
    parse_dates=True,
).reindex(
    building.set_timesteps
)  # Do not interpolate here, because it defeats the purpose of validation.
output_timeseries_validation.columns.name = 'output_name'  # For compatibility with output_timeseries.

# Run error calculation function.
(
    error_summary,
    error_timeseries
) = cobmo.utils.calculate_error(
    output_timeseries_validation,
    output_timeseries_simulation,
)

# Combine data for plotting.
zone_temperature_comparison = pd.concat(
    [
        output_timeseries_validation.loc[:, output_timeseries_validation.columns.str.contains('temperature')],
        output_timeseries_simulation.loc[:, output_timeseries_simulation.columns.str.contains('temperature')],
    ],
    keys=[
        'expected',
        'simulated',
    ],
    names=[
        'type',
        'output_name'
    ],
    axis=1
)
surface_irradiation_gain_exterior_comparison = pd.concat(
    [
        output_timeseries_validation.loc[:, output_timeseries_validation.columns.str.contains('irradiation_gain')],
        output_timeseries_simulation.loc[:, output_timeseries_simulation.columns.str.contains('irradiation_gain')],
    ],
    keys=[
        'expected',
        'simulated',
    ],
    names=[
        'type',
        'output_name'
    ],
    axis=1
)
surface_convection_interior_comparison = pd.concat(
    [
        output_timeseries_validation.loc[:, output_timeseries_validation.columns.str.contains(
            'convection_interior'
        )],
        output_timeseries_simulation.loc[:, output_timeseries_simulation.columns.str.contains(
            'convection_interior'
        )],
    ],
    keys=[
        'expected',
        'simulated',
    ],
    names=[
        'type',
        'output_name'
    ],
    axis=1
)

# Hvplot has no default options.
# Workaround: Pass this dict to every new plot.
hvplot_default_options = dict(width=1500, height=300)

# Generate plot handles.
thermal_power_plot = (
    control_timeseries_simulation.stack().rename('thermal_power').reset_index()
).hvplot.step(
    x='time',
    y='thermal_power',
    by='control_name',
    **hvplot_default_options
)
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
surface_irradition_gain_plot = (
    surface_irradiation_gain_exterior_comparison.stack().stack().rename('irradiation_gain').reset_index()
).hvplot.line(
    x='time',
    y='irradiation_gain',
    by=['type', 'output_name'],
    **hvplot_default_options
)
sky_temperature_plot = (
    building.disturbance_timeseries['sky_temperature'].rename('sky_temperature').reset_index()
).hvplot.line(
    x='time',
    y='sky_temperature',
    **hvplot_default_options
)
surface_convection_interior_plot = (
    surface_convection_interior_comparison.stack().stack().rename(
        'convection_interior'
    ).reset_index()
).hvplot.line(
    x='time',
    y='convection_interior',
    by=['type', 'output_name'],
    **hvplot_default_options
)
ambient_air_temperature_plot = (
    building.disturbance_timeseries['ambient_air_temperature'].rename('ambient_air_temperature').reset_index()
).hvplot.line(
    x='time',
    y='ambient_air_temperature',
    **hvplot_default_options
)
zone_temperature_plot = (
    zone_temperature_comparison.stack().stack().rename('zone_temperature').reset_index()
).hvplot.line(
    x='time',
    y='zone_temperature',
    by=['type', 'output_name'],
    **hvplot_default_options
)
zone_temperature_error_plot = (
    error_timeseries.loc[
        :, error_timeseries.columns.str.contains('temperature')
    ].stack().rename('zone_temperature_error').reset_index()
).hvplot.area(
    x='time',
    y='zone_temperature_error',
    by='output_name',
    stacked = False,
    alpha = 0.5,
    **hvplot_default_options
)
error_table = (
    error_summary.stack().rename('error_value').reset_index()
).hvplot.table(
    x='output_name',
    y='error_value',
    by='error_type',
    **hvplot_default_options
)

# Define layout and labels / render plots.
hvplot.show(
    (
        thermal_power_plot
        + irradiation_plot
        + surface_irradition_gain_plot
        + sky_temperature_plot
        + surface_convection_interior_plot
        + ambient_air_temperature_plot
        + zone_temperature_plot
        + zone_temperature_error_plot
        + error_table
    ).redim.label(
        time="Date / time",
        thermal_power="Thermal power [W]",
        ambient_air_temperature="Ambient air temp. [°C]",
        irradiation="Irradiation [W/m²]",
        zone_temperature="Zone temperature [°C]",
        zone_temperature_error="Zone temp. error [K]",
    ).cols(1),
    # Plots open in browser and are also stored in results directory.
    filename=os.path.join(results_path, 'validation_plots.html')
)

# Print error summary for debugging.
print("error_timeseries = ")
print(error_timeseries.head())
print("error_summary = ")
print(error_summary)
