"""
Building model main function definitions
"""

import os
import sqlite3
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

    print("-----------------------------------------------------------------------------------------------------------")
    print('building.state_matrix=')
    print(building.state_matrix)
    print("-----------------------------------------------------------------------------------------------------------")
    print('building.control_matrix=')
    print(building.control_matrix)
    print("-----------------------------------------------------------------------------------------------------------")
    print('building.disturbance_matrix=')
    print(building.disturbance_matrix)
    print("-----------------------------------------------------------------------------------------------------------")
    print('building.state_output_matrix=')
    print(building.state_output_matrix)
    print("-----------------------------------------------------------------------------------------------------------")
    print('building.control_output_matrix=')
    print(building.control_output_matrix)
    print("-----------------------------------------------------------------------------------------------------------")
    print('building.disturbance_output_matrix=')
    print(building.disturbance_output_matrix)
    print("-----------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    example()
