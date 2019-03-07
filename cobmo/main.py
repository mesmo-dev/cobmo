"""
Building model main function definitions
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import cobmo.building
import cobmo.utils


def connect_database(
        data_path=os.path.join(os.path.dirname(os.path.normpath(__file__)), '..', 'data'),
        overwrite_database=True
):
    # Create database, if none
    if overwrite_database or not os.path.isfile(os.path.join(data_path, 'data.sqlite')):
        cobmo.utils.create_database(
            sqlite_path=os.path.join(data_path, 'data.sqlite'),
            sql_path=os.path.join(data_path, 'data.sqlite.schema.sql'),
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

    building = get_building_model()

    # Define initial state and control timeseries
    state_initial = pd.Series(
        np.concatenate([
            26.0  # in °C
            * np.ones(sum(building.set_states.str.contains('temperature'))),
            100.0  # in ppm
            * np.ones(sum(building.set_states.str.contains('co2_concentration'))),
            0.013  # in kg(water)/kg(air)
            * np.ones(sum(building.set_states.str.contains('absolute_humidity')))
        ]),
        building.set_states
    )
    control_timeseries = pd.DataFrame(
        np.random.rand(len(building.set_timesteps), len(building.set_controls)),
        building.set_timesteps,
        building.set_controls
    )

    # Run simulation
    (
        state_timeseries,
        output_timeseries
    ) = building.simulate(
        state_initial=state_initial,
        control_timeseries=control_timeseries
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
    print("control_timeseries=")
    print(control_timeseries)
    print("-----------------------------------------------------------------------------------------------------------")
    print("disturbance_timeseries=")
    print(building.disturbance_timeseries)
    print("-----------------------------------------------------------------------------------------------------------")
    print("state_timeseries=")
    print(state_timeseries)
    print("-----------------------------------------------------------------------------------------------------------")
    print("output_timeseries=")
    print(output_timeseries)
    print("-----------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    example()
