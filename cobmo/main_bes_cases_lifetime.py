"""
Building model main function definitions
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import cobmo.building
import cobmo.controller_bes_lifetime
import cobmo.utils_bes_cases
import cobmo.config
import datetime
import time as time
import errno



def connect_database(
        data_path=cobmo.config.data_path,
        overwrite_database=True
):
    # Create database, if none
    if overwrite_database or not os.path.isfile(os.path.join(data_path, 'data.sqlite')):
        cobmo.utils_bes_cases.create_database(
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

    cobmo_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
    data_path = os.path.join(cobmo_path, 'data')
    cobmo_cobmo_path = os.path.join(cobmo_path, 'cobmo')

    # Creating the battery storage cases
    do_plotting = 1
    simulate = 1

    # Definition of the IRENA case
    # case = 'reference'
    case = 'best'
    (
        battery_params_2016,
        battery_params_2020,
        battery_params_2025,
        battery_params_2030,
        energy_cost,
        power_cost,
        lifetime,
        efficiency,
        depth_of_discharge
    ) = cobmo.utils_bes_cases.retrieve_battery_parameters(
        case=case
    )

    # lifetime.to_csv('results/results_bes_cases/' + case + '/lifetime_' + case + '.csv')

    # Retrieving the tech names from either of the DataFrames (they all have the same Indexes)
    techs = battery_params_2016.index

    # # Slicing for simulating on less technologies (shorter run time)
    # techs = techs[0:2]
    # energy_cost = energy_cost.iloc[0:2, :]
    # power_cost = energy_cost.iloc[0:2, :]
    # lifetime = energy_cost.iloc[0:2, :]
    # efficiency = energy_cost.iloc[0:2, :]
    # depth_of_discharge = energy_cost.iloc[0:2, :]

    years = pd.Index(['2016', '2020', '2025', '2030'])

    # Initialize dataframes to store results
    simple_payback_df = pd.DataFrame(
        0.0,
        index=techs,
        columns=years
    )
    discounted_payback_df = pd.DataFrame(
        0.0,
        index=techs,
        columns=years
    )
    storage_size_df = pd.DataFrame(
        0.0,
        index=techs,
        columns=years
    )

    if simulate == 1:
        time_start_cycle = time.clock()
        for t in techs:
            building_storage_types = pd.read_sql(
                """
                select * from building_storage_types
                """,
                conn,
                index_col='building_storage_type'
                # Indexing to allow precise modification of the dataframe.
                # If this is used you need to reindex as pandas when using to_sql (meaning NOT using "index=False")
            )

            for i in range(years.shape[0]):
                building_storage_types.at['battery_storage_default', 'storage_round_trip_efficiency'] = (
                    float(efficiency.iloc[techs.str.contains(t), i])
                )
                building_storage_types.at['battery_storage_default', 'storage_depth_of_discharge'] = (
                    float(depth_of_discharge.iloc[techs.str.contains(t), 1])
                )

                building_storage_types.at['battery_storage_default', 'storage_investment_sgd_per_unit'] = (
                    float(energy_cost.iloc[techs.str.contains(t), i])
                )
                building_storage_types.at['battery_storage_default', 'storage_power_installation_cost'] = (
                    float(power_cost.iloc[techs.str.contains(t), i])
                )
                building_storage_types.at['battery_storage_default', 'storage_lifetime'] = (
                    float(lifetime.iloc[techs.str.contains(t), i])
                )

                # Putting back into sql
                building_storage_types.to_sql(
                    'building_storage_types',
                    con=conn,
                    if_exists='replace'
                    # index=False
                )

                # Get the building model
                building = get_building_model(conn=conn)

                # Run controller
                controller = cobmo.controller_bes_lifetime.Controller_bes_lifetime(conn=conn, building=building)
                (_, _, _, storage_size, optimum_obj) = controller.solve()

                # Calculating the savings and the payback time
                storage_size_kwh = storage_size * 3.6e-3 * 1.0e-3
                costs_without_storage = 3.834195403e+02
                savings_day = (
                    costs_without_storage
                    - optimum_obj
                )

                # Running discounted payback function
                (discounted_payback, simple_payback, _) = cobmo.utils_bes_cases.discounted_payback_time(
                    building,
                    storage_size_kwh,
                    lifetime.iloc[techs.str.contains(t), i],  # storage lifetime as input
                    savings_day,
                    save_plot_on_off='off',  # "on" to save the plot as .svg (not tracked by the git)
                    plotting_on_off=0  # set 1 for plotting
                )

                # Storing results
                simple_payback_df.iloc[techs.str.contains(t), i] = simple_payback
                discounted_payback_df.iloc[techs.str.contains(t), i] = discounted_payback
                storage_size_df.iloc[techs.str.contains(t), i] = format(storage_size_kwh, '.2f')

        # Printing loop time
        print("\nCase run: %s"
              "\nTime to solve all techs and all years: %.2f minutes" % (case, (time.clock() - time_start_cycle)/60.0))

        # Saving files
        date_main = datetime.datetime.now()
        date_hr_min_sec = '{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
            date_main.year, date_main.month, date_main.day,
            date_main.hour, date_main.minute, date_main.second)
        save_path = os.path.join(
            cobmo_cobmo_path,
            'results',
            'results_bes_cases',
            case,
            case + '_' + date_hr_min_sec,
            # case + '_simple_payback_' + date_hr_min_sec
        )
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        filename_simple = case + '_simple_payback_' + date_hr_min_sec + '.csv'
        filename_discounted = case + '_discounted_payback_' + date_hr_min_sec + '.csv'
        filename_storage = case + '_storage_size_' + date_hr_min_sec + '.csv'

        simple_payback_df.to_csv(os.path.join(save_path, filename_simple))
        discounted_payback_df.to_csv(os.path.join(save_path, filename_discounted))
        storage_size_df.to_csv(os.path.join(save_path, filename_storage))
        
        if do_plotting == 1:  # Simulating adn plotting at the same time
            filepath_read = os.path.join(save_path, filename_storage)
            filename_plot = case + '_plot_' + date_hr_min_sec
            
            cobmo.utils_bes_cases.plot_battery_cases_storage_sizes(
                case,
                filepath_read,
                save_path,
                filename_plot
            )

    if do_plotting == 1 and simulate == 0:  # Only plotting

        filepath_read = (                                                   # << INPUT BY HAND
            'results/results_bes_cases/best/best_2019-08-30_10-47-00/best_storage_size_2019-08-30_10-47-00.csv'
        )

        save_path = (
            'results/results_bes_cases/best/best_2019-08-30_10-47-00/'      # << INPUT BY HAND
        )

        filename_plot = case + '_plot_' + (
            '2019-08-30_10-47-00'                                           # << INPUT BY HAND
        )

        cobmo.utils_bes_cases.plot_battery_cases_storage_sizes(
            case,
            filepath_read,
            save_path,
            filename_plot
        )


if __name__ == "__main__":
    example()


