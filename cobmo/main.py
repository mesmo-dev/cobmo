"""
Building model main function definitions
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import cobmo.building
import cobmo.controller
import cobmo.utils
import cobmo.config


def connect_database(
        data_path=cobmo.config.data_path,
        overwrite_database=True
):
    # Create database, if none
    if overwrite_database or not os.path.isfile(os.path.join(data_path, 'data.sqlite')):
        cobmo.utils.create_database(
            sqlite_path=os.path.join(data_path, 'data.sqlite'),
            sql_path=os.path.join(cobmo.config.cobmo_path, 'cobmo', 'database_schema.sql'),
            csv_path=data_path
        )

    conn = sqlite3.connect(os.path.join(data_path, 'data.sqlite'))
    return conn


def get_building_model(
        scenario_name='scenario_default',
        conn=connect_database()
):
    building = cobmo.building.Building(conn, scenario_name)
    return building


def example():
    """
    Example script
    """

    conn = connect_database()

    building_storage_types = pd.read_sql(
        """
        select * from building_storage_types
        """,
        conn
    )

    # Here make the changes to the data in the sql
    # building_storage_types.at['storage_round_trip_efficiency'][0] = 0.95

    building_storage_types.to_sql(
        'building_storage_types',
        con=conn,
        if_exists='replace',
        index=False
    )

    building = get_building_model(conn=conn)

    # Define initial state and control timeseries
    state_initial = pd.Series(
        np.concatenate([
            26.0  # in °C
            * np.ones(sum(building.set_states.str.contains('temperature'))),
            100.0  # in ppm
            * np.ones(sum(building.set_states.str.contains('co2_concentration'))),
            0.013  # in kg(water)/kg(air)
            * np.ones(sum(building.set_states.str.contains('absolute_humidity'))),
        ]),
        building.set_states
    )  # TODO: Move intial state defintion to building model
    control_timeseries_simulation = pd.DataFrame(
        np.random.rand(len(building.set_timesteps), len(building.set_controls)),
        building.set_timesteps,
        building.set_controls
    )

    # Define augemented state space model matrices
    building.define_augmented_model()

    # Run simulation
    (
        state_timeseries_simulation,
        output_timeseries_simulation
    ) = building.simulate(
        state_initial=state_initial,
        control_timeseries=control_timeseries_simulation
    )

    # Outputs for debugging
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

    # Run controller
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

    # Outputs for debugging
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

    # Run error calculation function
    (
        error_summary,
        error_timeseries
    ) = cobmo.utils.calculate_error(
        output_timeseries_simulation.loc[:, output_timeseries_controller.columns.str.contains('temperature')],
        output_timeseries_controller.loc[:, output_timeseries_controller.columns.str.contains('temperature')]
    )  # Note: These are exemplary inputs.

    # Outputs for debugging
    print("-----------------------------------------------------------------------------------------------------------")
    print("error_timeseries=")
    print(error_timeseries)
    print("-----------------------------------------------------------------------------------------------------------")
    print("error_summary=")
    print(error_summary)
    print("-----------------------------------------------------------------------------------------------------------")

    print('\n>>Cost per day (only level 8) = %.2f' % obj_optimum)


if __name__ == "__main__":
    example()


