"""
Use this script for single simulations of BATTERY storage.

ATTENTION !! Keep in mind the following points.
- The solution depends on teh lifetime assigned to the storage in the sql (file: building_storage_types.csv)
- Changing the lifetime changes the results
- The lifetime has to be usually very high (100 or 150) to have storage installed.

Hence, teh results from this script are not totally coherent from an economic analysis point of view.

"""

import os
import sqlite3
import numpy as np
import pandas as pd
import cobmo.building
import cobmo.controller_bes
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

    # ==============================================================
    # Creating the battery storage cases

    # Back to sql
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
    controller = cobmo.controller_bes.Controller_bes(
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
    plotting = 1
    save_plot = 0

    # if storage_size is not None:
    if 'storage' in building.building_scenarios['building_storage_type'][0]:
        storage_size_kwh = storage_size * 3.6e-3 * 1.0e-3
        print('\n----------------------------------------------')
        print('\n>> Storage size = %.2f kWh' % storage_size_kwh)
        print('\n>> Total opex + capex (storage)= {}'.format(format(optimum_obj, '.2f')))

        # Calculating the savings and the payback time
        costs_without_storage = 3.834195403e+02  # [SGD/day], 14 levels
        savings_day = (costs_without_storage - optimum_obj)
        (payback, payback_df) = cobmo.utils.discounted_payback_time(
            building,
            storage_size_kwh,
            savings_day,
            save_plot_on_off=save_plot,
            plotting_on_off=plotting
        )

        print('\n>> Storage type = %s'
              '  |  Optimal storage size = %.2f'
              '  | savings year 〜= %.2f'
              '  | Discounted payback = %i\n'
              % (
                building.building_scenarios['building_storage_type'][0],
                storage_size,
                savings_day * 260.0,
                payback
              )
              )
    else:
        print('\n----------------------------------------------')
        print('\n>> Total opex (baseline)= {}\n'.format(format(optimum_obj, '.2f')))

    if print_on_csv == 1:
        if ((building.building_scenarios['building_storage_type'][0] == 'sensible_thermal_storage_default')
                or (building.building_scenarios['building_storage_type'][0] == 'latent_thermal_storage_default')
                or (building.building_scenarios['building_storage_type'][0] == 'battery_storage_default')):

            building.state_matrix.to_csv('delete_me_storage/bes/state_matrix_BES.csv')
            building.control_matrix.to_csv('delete_me_storage/bes/control_matrix_BES.csv')
            building.disturbance_matrix.to_csv('delete_me_storage/bes/disturbance_matrix_BES.csv')

            building.state_output_matrix.to_csv('delete_me_storage/bes/state_output_matrix_BES.csv')
            building.control_output_matrix.to_csv('delete_me_storage/bes/control_output_matrix_BES.csv')
            building.disturbance_output_matrix.to_csv('delete_me_storage/bes/disturbance_output_matrix_BES.csv')

            # np.savetxt(r'my_file_output_state_matrix.txt', building.state_matrix.values) # , fmt='%d'
            state_timeseries_simulation.to_csv('delete_me_storage/bes/state_timeseries_simulation_BES.csv')

            state_timeseries_controller.to_csv('delete_me_storage/bes/state_timeseries_controller_BES.csv')
            date_main = dt.datetime.now()
            filename_out_controller = (
                    'output_timeseries_controller_BES' + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                        date_main.year, date_main.month, date_main.day, date_main.hour, date_main.minute,
                        date_main.second)
                    + '.csv'
            )
            output_timeseries_controller.to_csv('delete_me_storage/bes/' + filename_out_controller)

            control_timeseries_controller.to_csv('delete_me_storage/bes/control_timeseries_controller_BES.csv')

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


