"""
Building model main function definitions
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import hvplot
import hvplot.pandas
import time
import cobmo.building
import cobmo.controller
import cobmo.utils


def connect_database(
        data_path=os.path.join(os.path.dirname(os.path.normpath(__file__)), '..', 'data'),
        overwrite_database=True
):
    # Create database, if none
    if overwrite_database or not os.path.isfile(os.path.join(data_path, 'data.sqlite')):
        time_start = time.clock()
        cobmo.utils.create_database(
            sqlite_path=os.path.join(data_path, 'data.sqlite'),
            sql_path=os.path.join(data_path, 'data.sqlite.schema.sql'),
            csv_path=data_path
        )
        print("Database setup time: {:.2f} seconds".format(time.clock() - time_start))

    conn = sqlite3.connect(os.path.join(data_path, 'data.sqlite'))

    # time_start = time.clock()
    # cobmo.utils.calculate_irradiation_surfaces(
    #     conn,
    #     weather_type='singapore_iwec',
    #     irradiation_model='disc'
    # )
    # print("Irradiation processing time: {:.2f} seconds".format(time.clock() - time_start))
    return conn


def get_building_model(
        scenario_name='example_1_zone',
        conn=connect_database()
):
    building = cobmo.building.Building(conn, scenario_name)
    return building


def example():
    """
    Example script
    """

    time_start = time.clock()
    building = get_building_model()
    print("Building model setup time: {:.2f} seconds".format(time.clock() - time_start))

    # Define initial state and control timeseries
    state_initial = pd.Series(
        np.concatenate([
            24.0  # in °C
            * np.ones(sum(building.set_states.str.contains('temperature'))),
            100.0  # in ppm
            * np.ones(sum(building.set_states.str.contains('co2_concentration'))),
            0.013  # in kg(water)/kg(air)
            * np.ones(sum(building.set_states.str.contains('absolute_humidity')))
        ]),
        building.set_states
    )  # TODO: Move intial state defintion to building model
    # control_timeseries_simulation = pd.DataFrame(
    #     np.random.rand(len(building.set_timesteps), len(building.set_controls)),
    #     building.set_timesteps,
    #     building.set_controls
    # )
    control_timeseries_simulation = pd.DataFrame(
        0,
        building.set_timesteps,
        building.set_controls
    )
    control_timeseries_simulation.loc[:]['zone_1_generic_cool_thermal_power'] = 200

    building.disturbance_timeseries.loc[
    :, building.disturbance_timeseries.columns.str.contains('irradiation')
    ] = (
        building.disturbance_timeseries.loc[
            :, building.disturbance_timeseries.columns.str.contains('irradiation')
        ] * 1.0
    )

    # # Define augemented state space model matrices
    # time_start = time.clock()
    # building.define_augmented_model()
    # print("Augmented model setup time: {:.2f} seconds".format(time.clock() - time_start))

    # Run simulation
    time_start = time.clock()
    (
        state_timeseries_simulation,
        output_timeseries_simulation
    ) = building.simulate(
        state_initial=state_initial,
        control_timeseries=control_timeseries_simulation,
        disturbance_timeseries=building.disturbance_timeseries
    )
    print("Simulation solve time: {:.2f} seconds".format(time.clock() - time_start))

    # Outputs for debugging
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.state_matrix=")
    print(building.state_matrix.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.control_matrix=")
    print(building.control_matrix.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.disturbance_matrix=")
    print(building.disturbance_matrix.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.state_output_matrix=")
    print(building.state_output_matrix.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.control_output_matrix=")
    print(building.control_output_matrix.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.disturbance_output_matrix=")
    print(building.disturbance_output_matrix.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("control_timeseries_simulation=")
    print(control_timeseries_simulation.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("building.disturbance_timeseries=")
    print(building.disturbance_timeseries.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("state_timeseries_simulation=")
    print(state_timeseries_simulation.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("output_timeseries_simulation=")
    print(output_timeseries_simulation.head())
    print("-----------------------------------------------------------------------------------------------------------")

    # # Run controller
    # controller = cobmo.controller.Controller(
    #     conn=connect_database(),
    #     building=building
    # )
    # (
    #     control_timeseries_controller,
    #     state_timeseries_controller,
    #     output_timeseries_controller
    # ) = controller.solve()
    #
    # # Outputs for debugging
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("control_timeseries_controller=")
    # print(control_timeseries_controller.head())
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("state_timeseries_controller=")
    # print(state_timeseries_controller.head())
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("output_timeseries_controller=")
    # print(output_timeseries_controller.head())
    # print("-----------------------------------------------------------------------------------------------------------")

    # Load validation data.
    output_timeseries_validation = pd.read_csv(
        os.path.join(os.path.dirname(os.path.normpath(__file__)), '..', 'data', 'temp', 'validation_timeseries.csv'),
        index_col='time',
        parse_dates=True,
    ).reindex(
        building.set_timesteps
    )  # Do not interpolate here, because it defeats the purpose of validation.
    output_timeseries_validation.columns.name = 'output_name'  # For compatibility with output_timeseries.

    # Run error calculation function
    (
        error_summary,
        error_timeseries
    ) = cobmo.utils.calculate_error(
        output_timeseries_validation,
        output_timeseries_simulation,
    )

    # Combine data for plotting
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
    surface_thermal_radiation_gain_exterior_comparison = pd.concat(
        [
            output_timeseries_validation.loc[:, output_timeseries_validation.columns.str.contains(
                'thermal_radiation_gain_exterior'
            )],
            output_timeseries_simulation.loc[:, output_timeseries_simulation.columns.str.contains(
                'thermal_radiation_gain_exterior'
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
    surface_thermal_radiation_gain_exterior_plot = (
        surface_thermal_radiation_gain_exterior_comparison.stack().stack().rename(
            'thermal_radiation_gain_exterior'
        ).reset_index()
    ).hvplot.line(
        x='time',
        y='thermal_radiation_gain_exterior',
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
            + surface_thermal_radiation_gain_exterior_plot
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
        ).cols(1)
    )

    # Outputs for debugging
    print("-----------------------------------------------------------------------------------------------------------")
    print("error_timeseries=")
    print(error_timeseries.head())
    print("-----------------------------------------------------------------------------------------------------------")
    print("error_summary=")
    print(error_summary.head())
    print("-----------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    example()
