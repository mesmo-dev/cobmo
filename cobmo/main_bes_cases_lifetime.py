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
import tkinter as tk
from tkinter import filedialog


cobmo_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
# data_path = os.path.join(cobmo_path, 'data')
cobmo_cobmo_path = os.path.join(cobmo_path, 'cobmo')


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


scenario = 'scenario_default'


def get_building_model(
        scenario_name=scenario,
        conn=connect_database()
):
    building = cobmo.building.Building(conn, scenario_name)
    return building


def example():
    """
    Example script
    """

    conn = connect_database()

    # Extracting tables from the sql
    # CAREFUL!
    # Indexing to allow precise modification of the dataframe.
    # If this is used you need to reindex as pandas when using to_sql (meaning NOT using "index=False")
    building_scenarios_csv = pd.read_sql(
        """
        select * from building_scenarios
        """,
        conn,
        index_col='scenario_name'
    )
    buildings_csv = pd.read_sql(
        """
        select * from buildings
        """,
        conn,
        index_col='building_name'
    )

    # Setting the storage type to into the building
    # This is done to avoid changing by hand the storage type in the buildings.csv
    building_name = building_scenarios_csv.at[scenario, 'building_name']
    buildings_csv.at[building_name, 'building_storage_type'] = 'battery_storage_default'

    # Back to sql
    buildings_csv.to_sql(
        'buildings',
        con=conn,
        if_exists='replace'
        # index=False
    )

    # ___________________________________________________________________________________________________________________
    # Creating the battery storage cases
    do_plotting = 1
    simulate = 0
    plotting_options = [
        'savings_year', 'savings_year_percentage', 'storage_size', 'simple_payback', 'discounted_payback', 'efficiency', 'investment'
    ]
    what_plotting = 'savings_year_percentage'

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

    # Retrieving the tech names from either of the DataFrames (they all have the same Indexes)
    techs = battery_params_2016.index

    # # Slicing for simulating on less technologies (shorter run time)
    # techs = techs[0:2]
    # energy_cost = energy_cost.loc[techs, :]
    # power_cost = power_cost.loc[techs, :]
    # lifetime = lifetime.loc[techs, :]
    # efficiency = efficiency.loc[techs, :]
    # depth_of_discharge = depth_of_discharge.loc[techs, :]

    years = pd.Index(['2016', '2020', '2025', '2030'])
    # years = ['2016', '2020', '2025', '2030']

    # Redefining columns for plotting
    efficiency.columns = years
    energy_cost.columns = years

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
    savings_year_df = pd.DataFrame(
        0.0,
        index=techs,
        columns=years
    )
    savings_year_percentage_df = pd.DataFrame(
        0.0,
        index=techs,
        columns=years
    )

    # ___________________________________________________________________________________________________________________
    if simulate == 1:
        time_start_cycle = time.clock()
        counter = 0
        for t in techs:
            counter = counter + 1  # Needed to print the simulation number
            building_storage_types = pd.read_sql(
                """
                select * from building_storage_types
                """,
                conn,
                index_col='building_storage_type'
                # Indexing to allow precise modification of the dataframe.
                # If this is used you need to reindex as pandas when using to_sql (meaning NOT using "index=False")
            )

            for y in range(years.shape[0]):
                counter = counter + y - 1
                building_storage_types.at['battery_storage_default', 'storage_round_trip_efficiency'] = (
                    float(efficiency.iloc[techs.str.contains(t), y])
                    * 0.95  # Accounting for inverter efficiency
                )
                building_storage_types.at['battery_storage_default', 'storage_depth_of_discharge'] = (
                    float(depth_of_discharge.iloc[techs.str.contains(t), y])
                )

                building_storage_types.at['battery_storage_default', 'storage_investment_sgd_per_unit'] = (
                    float(energy_cost.iloc[techs.str.contains(t), y])
                )
                building_storage_types.at['battery_storage_default', 'storage_power_installation_cost'] = (
                    float(power_cost.iloc[techs.str.contains(t), y])
                )
                building_storage_types.at['battery_storage_default', 'storage_lifetime'] = (
                    float(lifetime.iloc[techs.str.contains(t), y])
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
                # print('\n Simulation: %i/%i' % (int(counter), int(float(techs.shape[0]) * float(years.shape[0]))))
                controller = cobmo.controller_bes_lifetime.Controller_bes_lifetime(conn=conn, building=building)
                (_, _, _, storage_size, optimum_obj) = controller.solve()

                # Calculating the savings and the payback time
                storage_size_kwh = storage_size * 3.6e-3 * 1.0e-3
                costs_without_storage = 3.834195403e+02
                costs_year_baseline = costs_without_storage * 260.0
                savings_day = (
                    costs_without_storage
                    - optimum_obj
                )
                savings_year = savings_day * 260.0
                savings_year_percentage = float(savings_year/costs_year_baseline*100.0)

                # Running discounted payback function
                (discounted_payback, simple_payback, _) = cobmo.utils_bes_cases.discounted_payback_time(
                    building,
                    storage_size_kwh,
                    lifetime.iloc[techs.str.contains(t), y],  # storage lifetime as input
                    savings_day,
                    save_plot_on_off='off',  # "on" to save the plot as .svg (not tracked by the git)
                    plotting_on_off=0  # set 1 for plotting
                )

                # Storing results
                simple_payback_df.iloc[techs.str.contains(t), y] = simple_payback
                discounted_payback_df.iloc[techs.str.contains(t), y] = discounted_payback
                storage_size_df.iloc[techs.str.contains(t), y] = format(storage_size_kwh, '.2f')
                savings_year_df.iloc[techs.str.contains(t), y] = format(savings_year, '.1f')
                savings_year_percentage_df.iloc[techs.str.contains(t), y] = format(savings_year_percentage, '.1f')

        # Creating the mask reflecting where the payback exists
        mask = simple_payback_df == -0.0

        # Changing efficiency and investment frames to reflect the mas conditions
        efficiency[mask] = 0.0
        energy_cost[mask] = 0.0

        # Printing loop time
        print("\n__________________________________________"
              "\nCase run: %s"
              "\n%i simulations run"
              "\nTime to solve all techs and all years: %.2f minutes"
              % (case,
                 int(float(techs.shape[0]) * float(years.shape[0])),
                 (time.clock() - time_start_cycle)/60.0)
              + "\n__________________________________________")

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

        # TODO: make the filepaths smarter and dependent on "what_plotting"
        filename_simple = case + '_simple_payback_' + date_hr_min_sec + '.csv'
        filename_discounted = case + '_discounted_payback_' + date_hr_min_sec + '.csv'
        filename_storage = case + '_storage_size_' + date_hr_min_sec + '.csv'
        filename_savings_year = case + '_savings_year_' + date_hr_min_sec + '.csv'
        filename_savings_year_percentage = case + '_savings_year_percentage_' + date_hr_min_sec + '.csv'
        filename_efficiency = case + '_efficiency_' + date_hr_min_sec + '.csv'
        filename_investment = case + '_investment_' + date_hr_min_sec + '.csv'

        simple_payback_df.to_csv(os.path.join(save_path, filename_simple))
        discounted_payback_df.to_csv(os.path.join(save_path, filename_discounted))
        storage_size_df.to_csv(os.path.join(save_path, filename_storage))
        savings_year_df.to_csv(os.path.join(save_path, filename_savings_year))
        savings_year_percentage_df.to_csv(os.path.join(save_path, filename_savings_year_percentage))
        efficiency.to_csv(os.path.join(save_path, filename_efficiency))
        energy_cost.to_csv(os.path.join(save_path, filename_investment))
        
        if do_plotting == 1:  # Simulating and plotting at the same time
            if what_plotting == 'savings_year':
                filepath_read = os.path.join(save_path, filename_savings_year)
            if what_plotting == 'savings_year_percentage':
                filepath_read = os.path.join(save_path, filename_savings_year_percentage)
            elif what_plotting == 'storage_size':
                filepath_read = os.path.join(save_path, filename_storage)
            elif what_plotting == 'simple_payback':
                filepath_read = os.path.join(save_path, filename_simple)
            elif what_plotting == 'discounted_payback':
                filepath_read = os.path.join(save_path, filename_discounted)
            elif what_plotting == 'efficiency':
                filepath_read = os.path.join(save_path, filename_efficiency)
            elif what_plotting == 'investment':
                filepath_read = os.path.join(save_path, filename_investment)

            filename_plot = what_plotting + '_' + date_hr_min_sec

            save_path_plots = (
                    save_path
                    + '/plots_'
                    + date_hr_min_sec
            )
            if not os.path.exists(save_path_plots):
                try:
                    os.makedirs(save_path_plots)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            cobmo.utils_bes_cases.plot_battery_cases_storage_sizes(
                case,
                filepath_read,
                save_path_plots,
                filename_plot,
                labels=what_plotting,
                savepdf=0
            )

    # ___________________________________________________________________________________________________________________
    if do_plotting == 1 and simulate == 0:  # Only plotting

        for what_plotting in plotting_options:
            datetime_path = (
                '2019-08-31_15-58-05'      # << INPUT BY HAND
            )
            filepath_read = (
                'results/results_bes_cases/best/best_'
                + datetime_path
                + '/best_'
                + what_plotting
                + '_'
                + datetime_path
                + '.csv'
            )
            # root = tk.Tk()
            # root.withdraw()
            #
            # filepath_read = filedialog.askopenfilename()

            save_path = (
                    'results/results_bes_cases/best/best_'
                    + datetime_path
            )

            save_path_plots = (
                    save_path
                    + '/plots_'
                    + datetime_path
            )
            if not os.path.exists(save_path_plots):
                try:
                    os.makedirs(save_path_plots)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            filename_plot = (
                    what_plotting
                    + '_'
                    + datetime_path
            )

            cobmo.utils_bes_cases.plot_battery_cases_storage_sizes(
                case,
                filepath_read,
                save_path_plots,
                filename_plot,
                labels=what_plotting,
                savepdf=1
            )


if __name__ == "__main__":
    example()


