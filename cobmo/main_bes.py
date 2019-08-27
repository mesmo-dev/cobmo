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

    # Here make the changes to the data in the sql
    rt_efficiency = 0.8
    # building_storage_types.at['sensible_thermal_storage_default', 'storage_round_trip_efficiency'] = rt_efficiency

    # print('\nbuilding_storage_types in main = ')
    # print(building_storage_types)

    building_storage_types.to_sql(
        'building_storage_types',
        con=conn,
        if_exists='replace'
        # index=False
    )

    # NB: All the changes to the sql need to be done BEFORE getting the building_model
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
            0.0  # in all the storage units (sensible: m3 | PCM: kg | battery: kWh)
            * np.ones(sum(building.set_states.str.contains('state_of_charge'))),
            0.0  # Mass factor must be coherent with initial volume of bottom layer
            * np.ones(sum(building.set_states.str.contains('storage_mass_factor')))
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
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.state_matrix=")
    # print(building.state_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.control_matrix=")
    # print(building.control_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.disturbance_matrix=")
    # print(building.disturbance_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.state_output_matrix=")
    # print(building.state_output_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.control_output_matrix=")
    # print(building.control_output_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.disturbance_output_matrix=")
    # print(building.disturbance_output_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("control_timeseries_simulation=")
    # print(control_timeseries_simulation)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.disturbance_timeseries=")
    # print(building.disturbance_timeseries)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("state_timeseries_simulation=")
    # print(state_timeseries_simulation)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("output_timeseries_simulation=")
    # print(output_timeseries_simulation)
    # print("-----------------------------------------------------------------------------------------------------------")

    # file_output_text = open("my_file_out.txt", "w")
    # file_output_text.write(building.state_matrix)

    # Run controller
    controller = cobmo.controller.Controller(
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

    print_on_csv = 0  # set to 1 to print results in csv files (not tracked by the git)
    plotting = 1  # set 1 for plotting (to save the plot set "save_plot_on_off" to on

    if storage_size is not None:
        print('\n----------------------------------------------')
        print('\n>> Storage size = %.2f m3' % storage_size)
        print('\n>> Total opex + capex (storage)= {}'.format(format(optimum_obj, '.2f')))

        if plotting == 1:
            # Calculating the savings and the payback time
            costs_without_storage = 3.834195403e+02  # [SGD/day], 14 levels
            savings_day = (costs_without_storage - optimum_obj)
            storage_investment_per_unit = building.building_scenarios['storage_investment_sgd_per_unit'][0]
            (payback, payback_df) = cobmo.utils.discounted_payback_time(
                building,
                storage_size,
                storage_investment_per_unit,
                savings_day,
                save_plot_on_off='off'  # "on" to save the plot as .svg (not tracked by the git)
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

    # # Outputs for debugging
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("control_timeseries_controller=")
    # print(control_timeseries_controller)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("state_timeseries_controller=")
    # print(state_timeseries_controller)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("output_timeseries_controller=")
    # print(output_timeseries_controller)
    # print("-----------------------------------------------------------------------------------------------------------")
    #
    # # Run error calculation function
    # (
    #     error_summary,
    #     error_timeseries
    # ) = cobmo.utils.calculate_error(
    #     output_timeseries_simulation.loc[:, output_timeseries_controller.columns.str.contains('temperature')],
    #     output_timeseries_controller.loc[:, output_timeseries_controller.columns.str.contains('temperature')]
    # )  # Note: These are exemplary inputs.
    #
    # # Outputs for debugging
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("error_timeseries=")
    # print(error_timeseries)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("error_summary=")
    # print(error_summary)
    # print("-----------------------------------------------------------------------------------------------------------")

    if print_on_csv == 1:
        if ((building.building_scenarios['building_storage_type'][0] == 'sensible_thermal_storage_default')
                or (building.building_scenarios['building_storage_type'][0] == 'latent_thermal_storage_default')
                or (building.building_scenarios['building_storage_type'][0] == 'battery_storage_default')):

            building.state_matrix.to_csv('delete_me_storage/state_matrix_STORAGE.csv')
            building.control_matrix.to_csv('delete_me_storage/control_matrix_STORAGE.csv')
            building.disturbance_matrix.to_csv('delete_me_storage/disturbance_matrix_STORAGE.csv')

            building.state_output_matrix.to_csv('delete_me_storage/state_output_matrix_STORAGE.csv')
            building.control_output_matrix.to_csv('delete_me_storage/control_output_matrix_STORAGE.csv')
            building.disturbance_output_matrix.to_csv('delete_me_storage/disturbance_output_matrix_STORAGE.csv')

            # np.savetxt(r'my_file_output_state_matrix.txt', building.state_matrix.values) # , fmt='%d'
            state_timeseries_simulation.to_csv('delete_me_storage/state_timeseries_simulation_STORAGE.csv')

            state_timeseries_controller.to_csv('delete_me_storage/state_timeseries_controller_STORAGE.csv')
            date_main = dt.datetime.now()
            filename_out_controller = (
                    'output_timeseries_controller_STORAGE' + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                        date_main.year, date_main.month, date_main.day, date_main.hour, date_main.minute,
                        date_main.second)
                    + '.csv'
            )
            output_timeseries_controller.to_csv('delete_me_storage/' + filename_out_controller)

            control_timeseries_controller.to_csv('delete_me_storage/control_timeseries_controller_STORAGE.csv')

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


