"""Run script for building model validation."""

import hvplot
import hvplot.pandas
import numpy as np
import os
import pandas as pd

import cobmo.building_model
import cobmo.config
import cobmo.database_interface
import cobmo.utils


def main():

    # Settings.
    scenario_name = 'validation_1zone_no_window'
    results_path = os.path.join(cobmo.config.results_path, f'run_validation_{cobmo.config.timestamp}')
    validation_data_path = os.path.join(cobmo.config.data_path, 'validation_data')

    # Recreate database.
    cobmo.database_interface.recreate_database()

    # Instantiate results directory.
    os.mkdir(results_path)

    # Obtain building model.
    building = (
        cobmo.building_model.BuildingModel(
            scenario_name,
            with_validation_outputs=True
        )
    )

    # Print building model matrices and disturbance timeseries.
    print(f"state_matrix = \n{building.state_matrix}")
    print(f"control_matrix = \n{building.control_matrix}")
    print(f"disturbance_matrix = \n{building.disturbance_matrix}")
    print(f"state_output_matrix = \n{building.state_output_matrix}")
    print(f"control_output_matrix = \n{building.control_output_matrix}")
    print(f"disturbance_output_matrix = \n{building.disturbance_output_matrix}")
    print(f"disturbance_timeseries = \n{building.disturbance_timeseries}")

    # Store building model matrices and disturbance timeseries as CSV.
    building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
    building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
    building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
    building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
    building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
    building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
    building.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

    # Load validation data.
    output_vector_validation = (
        pd.read_csv(
            os.path.join(validation_data_path, scenario_name + '.csv'),
            index_col='time',
            parse_dates=True,
        ).reindex(
            building.timesteps
        )  # Do not interpolate here, because it defeats the purpose of validation.
    )
    output_vector_validation.columns.name = 'output_name'  # For compatibility with output_vector.

    # Define initial state.
    state_vector_initial = building.state_vector_initial
    for state in building.states:
        if state in output_vector_validation.columns:
            # Set initial state to match validation data.
            state_vector_initial.at[state] = output_vector_validation.loc[:, state].iat[0]

    # Define control vector.
    control_vector_simulation = pd.DataFrame(
        0.0,
        building.timesteps,
        building.controls
    )
    control_vector_simulation.loc[:, 'zone_1_generic_cool_thermal_power'] = 0.0

    # Use ambient temperature as sky temperature.
    building.disturbance_timeseries.loc[:, 'sky_temperature'] = (
        building.disturbance_timeseries.loc[:, 'ambient_air_temperature']
    )

    # Run simulation.
    (
        state_vector_simulation,
        output_vector_simulation
    ) = building.simulate(
        state_initial=state_vector_initial,
        control_vector=control_vector_simulation
    )

    # Print simulation results.
    print(f"control_vector_simulation = \n{control_vector_simulation}")
    print(f"state_vector_simulation = \n{state_vector_simulation}")
    print(f"output_vector_simulation = \n{output_vector_simulation}")

    # Store simulation results as CSV.
    control_vector_simulation.to_csv(os.path.join(results_path, 'control_vector_simulation.csv'))
    state_vector_simulation.to_csv(os.path.join(results_path, 'state_vector_simulation.csv'))
    output_vector_simulation.to_csv(os.path.join(results_path, 'output_vector_simulation.csv'))

    # Run error calculation function.
    (
        error_summary,
        error_timeseries
    ) = cobmo.utils.calculate_error(
        output_vector_validation,
        output_vector_simulation,
    )

    # Print error summary.
    print("error_timeseries = ")
    print(error_timeseries.head())
    print("error_summary = ")
    print(error_summary)

    # Store error summary as CSV.
    error_summary.to_csv(os.path.join(results_path, 'error_summary.csv'))
    error_timeseries.to_csv(os.path.join(results_path, 'error_timeseries.csv'))

    # Combine data for plotting.
    zone_temperature_comparison = (
        pd.concat(
            [
                output_vector_validation.loc[:, (
                    output_vector_validation.columns.str.contains('zone')
                    & output_vector_validation.columns.str.contains('temperature')
                )],
                output_vector_simulation.loc[:, (
                    output_vector_simulation.columns.str.contains('zone')
                    & output_vector_simulation.columns.str.contains('temperature')
                )],
            ],
            keys=['expected', 'simulated'],
            names=['type', 'output_name'],
            axis=1
        )
    )
    surface_temperature_comparison = (
        pd.concat(
            [
                output_vector_validation.loc[:, (
                    output_vector_validation.columns.str.contains('surface')
                    & output_vector_validation.columns.str.contains('temperature')
                )],
                output_vector_simulation.loc[:, (
                    output_vector_simulation.columns.str.contains('surface')
                    & output_vector_simulation.columns.str.contains('temperature')
                )],
            ],
            keys=['expected', 'simulated'],
            names=['type', 'output_name'],
            axis=1
        )
    )
    surface_irradiation_gain_exterior_comparison = (
        pd.concat(
            [
                output_vector_validation.loc[:, output_vector_validation.columns.str.contains('irradiation_gain')],
                output_vector_simulation.loc[:, output_vector_simulation.columns.str.contains('irradiation_gain')],
            ],
            keys=['expected', 'simulated',],
            names=['type', 'output_name'],
            axis=1
        )
    )
    surface_convection_interior_comparison = (
        pd.concat(
            [
                output_vector_validation.loc[:, output_vector_validation.columns.str.contains('convection_interior')],
                output_vector_simulation.loc[:, output_vector_simulation.columns.str.contains('convection_interior')],
            ],
            keys=['expected', 'simulated',],
            names=['type', 'output_name'],
            axis=1
        )
    )

    # Hvplot has no default options.
    # Workaround: Pass this dict to every new plot.
    hvplot_default_options = dict(width=1500, height=400)

    # Generate plot handles.
    thermal_power_plot = (
        control_vector_simulation.stack().rename('thermal_power').reset_index()
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
    surface_temperature_plot = (
        surface_temperature_comparison.stack().stack().rename('surface_temperature').reset_index()
    ).hvplot.line(
        x='time',
        y='surface_temperature',
        by=['type', 'output_name'],
        **hvplot_default_options
    )
    zone_temperature_error_plot = (
        error_timeseries.loc[:, (
            error_timeseries.columns.str.contains('zone')
            & error_timeseries.columns.str.contains('temperature')
        )].stack().rename('zone_temperature_error').reset_index()
    ).hvplot.area(
        x='time',
        y='zone_temperature_error',
        by='output_name',
        stacked=False,
        alpha=0.5,
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
            + surface_temperature_plot
            + zone_temperature_plot
            + zone_temperature_error_plot
            + error_table
        ).redim.label(
            time="Date / time",
            thermal_power="Thermal power [W]",
            ambient_air_temperature="Ambient air temp. [°C]",
            irradiation="Irradiation [W/m²]",
            surface_temperature="Surface temperature [°C]",
            zone_temperature="Zone temperature [°C]",
            zone_temperature_error="Zone temp. error [K]",
        ).cols(1),
        # Plots open in browser and are also stored in results directory.
        filename=os.path.join(results_path, 'validation_plots.html')
    )

    # Print results path.
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
