"""
Use this script for single simulations of SENSIBLE storage
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import cobmo.building
import cobmo.controller_sensible
import cobmo.utils
import cobmo.config
import datetime as dt


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
        conn,
        index_col='building_storage_type'
        # Indexing to allow precise modification of the dataframe.
        # If this is used you need to reindex as pandas when using to_sql (meaning NOT using "index=False")
    )

    building_storage_types.to_sql(
        'building_storage_types',
        con=conn,
        if_exists='replace'
        # index=False
    )

    # NB: All the changes to the sql need to be done BEFORE getting the building_model
    building = get_building_model(conn=conn)

    # Define initial state and control timeseries
    state_initial = building.set_state_initial
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

    # Run controller
    controller = cobmo.controller_sensible.Controller_sensible(
        conn=conn,
        building=building
    )
    (
        control_timeseries_controller,
        state_timeseries_controller,
        output_timeseries_controller,
        storage_size,
        optimum_obj
    ) = controller.solve()

    # -------------------------------------------------------------------------------------------------------------------

    # Printing and Plotting

    print_on_csv = 0
    plotting = 0
    save_plot = 0

    if storage_size is not None or storage_size != 0:
        print('\n----------------------------------------------')
        print('\n>> Total opex (storage)= {}'.format(float(optimum_obj)))

        # Calculating the savings and the payback time
        costs_without_storage = 3.834195403e+02  # [SGD/day], 14 levels
        savings_day = (costs_without_storage - optimum_obj)
        storage_investment_per_unit = building.building_scenarios['storage_investment_sgd_per_unit'][0]

        if building.building_scenarios['investment_sgd_per_X'][0] == 'kwh':
            storage_size = storage_size * 1000.0 * 4186.0 * 8.0 * 2.77778e-7
            print('\n>> Storage size = %.2f kWh' % storage_size)
        elif building.building_scenarios['investment_sgd_per_X'][0] == 'm3':
            print('\n>> Storage size = %.2f m3' % storage_size)
        else:
            print('\n Please define a specific unit of the storage investment')
            breakpoint()

        (simple_payback, discounted_payback) = cobmo.utils.discounted_payback_time(
            building,
            storage_size,
            storage_investment_per_unit,
            savings_day,
            save_plot,
            plotting
        )

        print('\n>> Storage type = %s'
              '  |  Optimal storage size = %.2f'
              '  | savings year 〜= %.2f'
              '  | Discounted payback = %i\n'
              % (
                building.building_scenarios['building_storage_type'][0],
                storage_size,
                savings_day * 260.0,
                discounted_payback
              )
              )
    else:
        if 'storage' in building.building_scenarios['building_storage_type'][0]:
            print('\n Storage size is zero. No storage installed by the optimization.')
        else:
            print('\n----------------------------------------------')
            print('\n>> Total opex (baseline)= {}\n'.format(format(optimum_obj, '.2f')))

    # Printing the outputs to dedicated csv files. These are IGNORED by the git
    if print_on_csv == 1:
        if ((building.building_scenarios['building_storage_type'][0] == 'sensible_thermal_storage_default')
                or (building.building_scenarios['building_storage_type'][0] == 'latent_thermal_storage_default')
                or (building.building_scenarios['building_storage_type'][0] == 'battery_storage_default')):

            building.state_matrix.to_csv('delete_me_storage/sensible/state_matrix_SENSIBLE.csv')
            building.control_matrix.to_csv('delete_me_storage/sensible/control_matrix_SENSIBLE.csv')
            building.disturbance_matrix.to_csv('delete_me_storage/sensible/disturbance_matrix_SENSIBLE.csv')

            building.state_output_matrix.to_csv('delete_me_storage/sensible/state_output_matrix_SENSIBLE.csv')
            building.control_output_matrix.to_csv('delete_me_storage/sensible/control_output_matrix_SENSIBLE.csv')
            building.disturbance_output_matrix.to_csv('delete_me_storage/sensible/disturbance_output_matrix_SENSIBLE.csv')

            # np.savetxt(r'my_file_output_state_matrix.txt', building.state_matrix.values) # , fmt='%d'
            state_timeseries_simulation.to_csv('delete_me_storage/sensible/state_timeseries_simulation_SENSIBLE.csv')

            state_timeseries_controller.to_csv('delete_me_storage/sensible/state_timeseries_controller_SENSIBLE.csv')
            date_main = dt.datetime.now()
            filename_out_controller = (
                    'output_timeseries_controller_SENSIBLE' + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                        date_main.year, date_main.month, date_main.day, date_main.hour, date_main.minute,
                        date_main.second)
                    + '.csv'
            )
            output_timeseries_controller.to_csv('delete_me_storage/sensible/' + filename_out_controller)

            control_timeseries_controller.to_csv('delete_me_storage/sensible/control_timeseries_controller_SENSIBLE.csv')

        else:
            building.state_matrix.to_csv('delete_me/state_matrix.csv')
            building.control_matrix.to_csv('delete_me/control_matrix.csv')
            building.disturbance_matrix.to_csv('delete_me/disturbance_matrix.csv')

            building.state_output_matrix.to_csv('delete_me/state_output_matrix.csv')
            building.control_output_matrix.to_csv('delete_me/control_output_matrix.csv')
            building.disturbance_output_matrix.to_csv('delete_me/disturbance_output_matrix.csv')

            # np.savetxt(r'my_file_output_state_matrix.txt', building.state_matrix.values) # , fmt='%d'
            state_timeseries_simulation.to_csv('delete_me/state_timeseries_simulation.csv')

            state_timeseries_controller.to_csv('delete_me/state_timeseries_controller.csv')

            date_main = dt.datetime.now()
            filename_out_controller = (
                    'output_timeseries_controller' + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                        date_main.year, date_main.month, date_main.day, date_main.hour, date_main.minute,
                        date_main.second)
                    + '.csv'
            )
            output_timeseries_controller.to_csv('delete_me/' + filename_out_controller)
            control_timeseries_controller.to_csv('delete_me/control_timeseries_controller.csv')


if __name__ == "__main__":
    example()


