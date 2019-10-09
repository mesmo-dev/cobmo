"""Run script for BES cases lifetime."""

import datetime
import errno
import numpy as np
import pandas as pd
import time as time
import tkinter
import os

import cobmo.building
import cobmo.config
import cobmo.controller_bes_lifetime
import cobmo.database_interface
import cobmo.utils_bes_cases


cobmo_cobmo_path = os.path.join(cobmo.config.cobmo_path, 'cobmo')


# Settings.
scenario_name = 'scenario_default'
pricing_method = 'retailer_peak_offpeak'
# Choices: 'wholesale_market', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40',
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'
do_plotting = 1
simulate = 1

plotting_options = [
    'savings_year', 'savings_year_percentage', 'storage_size', 'simple_payback',
    'discounted_payback', 'efficiency', 'investment'
]

# Definition of the IRENA case.
case = 'reference'
# case = 'best'
# case = 'worst'

interest_rate = 0.02

if simulate == 1:
    print('\n________________________'
          '\nSimulation options:'
          '\n- Case: *%s*'
          '\n- Price signal: *%s*'
          '\n- Interest rate: *%.2f*'
          '\n________________________'
          % (case, pricing_method, interest_rate))

    conn = cobmo.database_interface.connect_database()

    # Load selected database tables for modification.
    building_scenarios = pd.read_sql(
        """
        SELECT * FROM building_scenarios
        """,
        conn,
        index_col='scenario_name'
    )
    buildings = pd.read_sql(
        """
        SELECT * FROM buildings
        """,
        conn,
        index_col='building_name'
    )
    # Importing the parameter_sets to change the storage lifetime
    building_parameter_sets = pd.read_sql(  # TODO: put the lifetime somewhere else and do not modify the parameters
        """
        SELECT * FROM building_parameter_sets
        """,
        conn,
        # index_col='parameter_set'
    )
    position_lifetime = building_parameter_sets.index[
        building_parameter_sets['parameter_name'] == 'storage_lifetime'
    ].tolist()

    # Setting the storage type to into the building
    # This is done to avoid changing by hand the storage type in the buildings.csv
    building_name = building_scenarios.at[scenario_name, 'building_name']

    # Modify `price_type` for current scenario in the database.
    building_scenarios.at[scenario_name, 'price_type'] = pricing_method
    building_scenarios.to_sql(
        'building_scenarios',
        con=conn,
        if_exists='replace'
    )

    # ___________________________________________________________________________________________________________________
    # Creating the battery storage cases
    what_plotting = 'savings_year_percentage'
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
    years = pd.Index(['2016', '2020', '2025', '2030'])

    # ________________________________________________________________________
    # # @Slicing for simulating on less technologies (shorter run time) - de-comment to use

    # techs = techs[0:-1]
    # # years = years[-2:]
    # energy_cost = energy_cost.iloc[0:-1, :]
    # power_cost = power_cost.iloc[0:-1, :]
    # lifetime = lifetime.iloc[0:-1, :]
    # efficiency = efficiency.iloc[0:-1, :]
    # depth_of_discharge = depth_of_discharge.iloc[0:-1, :]
    # ________________________________________________________________________

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

    # ________________________________________________________
    if simulate == 1:
        time_start_cycle = time.clock()
        counter = 0
        for t in techs:
            building_storage_types = pd.read_sql(
                """
                SELECT * FROM building_storage_types
                """,
                conn,
                index_col='building_storage_type'
            )

            for y in range(years.shape[0]):
                counter = counter + 1
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

                # Lifetime
                building_storage_types.at['battery_storage_default', 'storage_lifetime'] = (
                    float(lifetime.iloc[techs.str.contains(t), y])
                )
                building_parameter_sets.loc[position_lifetime, 'parameter_value'] = (
                    float(lifetime.iloc[techs.str.contains(t), y])
                )

                # Putting back into sql
                building_storage_types.to_sql(
                    'building_storage_types',
                    con=conn,
                    if_exists='replace'
                )

                building_parameter_sets.to_sql(
                    'building_parameter_sets',
                    con=conn,
                    if_exists='replace',
                    index=False
                )

                # Run baseline scenario
                buildings.at[building_name, 'building_storage_type'] = ''
                buildings.to_sql(
                    'buildings',
                    con=conn,
                    if_exists='replace'
                )
                print('\n-----------------------------------------------------------------------------')
                print('\n>> Simulation # %i' % counter)
                print('\n\n________________________Setup @BASELINE scenario________________________')
                building_baseline = cobmo.building.Building(conn, scenario_name)
                controller = cobmo.controller_bes_lifetime.Controller_bes_lifetime(conn=conn, building=building_baseline)
                (_, _, _, _, obj_baseline) = controller.solve()  # TODO: solve this with normal controller

                # Run storage scenario
                print('\n\n________________________Setup @STORAGE scenario________________________')

                # print('\n Simulation: %i/%i' % (int(counter), int(float(techs.shape[0]) * float(years.shape[0]))))
                buildings.at[building_name, 'building_storage_type'] = 'battery_storage_default'
                buildings.to_sql(
                    'buildings',
                    con=conn,
                    if_exists='replace'
                )
                building = cobmo.building.Building(conn, scenario_name)
                controller = cobmo.controller_bes_lifetime.Controller_bes_lifetime(conn=conn, building=building)
                (_, _, _, storage_size, obj_storage) = controller.solve()

                # Calculating the savings and the payback time
                storage_size_kwh = storage_size * 2.77778e-7  # * 3.6e-3 * 1.0e-3
                costs_year_baseline = obj_baseline * 260.0
                savings_day = (obj_baseline - obj_storage)
                savings_year = savings_day * 260.0

                # Running discounted payback function
                print("Savings / day = %.2f $" % savings_day)
                if float(savings_day) > 0.00001:
                    (discounted_payback, simple_payback, _) = cobmo.utils_bes_cases.discounted_payback_time(
                        building,
                        storage_size_kwh,
                        lifetime.iloc[techs.str.contains(t), y],  # storage lifetime as input
                        savings_day,
                        save_plot_on_off='off',  # "on" to save the plot as .svg (not tracked by the git)
                        plotting_on_off=0,  # set 1 for plotting
                        interest_rate=interest_rate
                    )
                else:  # Passing over the calculation of payback
                    discounted_payback = 0.0
                    simple_payback = 0.0

                if discounted_payback == 0.0:
                    savings_year = 0.0

                savings_year_percentage = float(savings_year / costs_year_baseline * 100.0)
                # Storing results
                simple_payback_df.iloc[techs.str.contains(t), y] = simple_payback
                discounted_payback_df.iloc[techs.str.contains(t), y] = discounted_payback
                storage_size_df.iloc[techs.str.contains(t), y] = format(storage_size_kwh, '.2f')
                savings_year_df.iloc[techs.str.contains(t), y] = format(savings_year, '.1f')
                savings_year_percentage_df.iloc[techs.str.contains(t), y] = format(savings_year_percentage, '.1f')

        # Creating the mask reflecting where the payback exists
        # and changing efficiency and investment frames to reflect the mask conditions
        mask = discounted_payback_df == -0.0
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
            pricing_method,
            # pricing_method + '_' + date_hr_min_sec,
            case + '_' + pricing_method + '_' + date_hr_min_sec + ' - ' + str(int(interest_rate * 100.0)) + '%'
            # case + '_simple_payback_' + date_hr_min_sec
        )
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # TODO: make the filepaths smarter and dependent on "what_plotting"
        filename_simple = 'simple_payback_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'
        filename_discounted = 'discounted_payback_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'
        filename_storage = 'storage_size_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'
        filename_savings_year = 'savings_year_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'
        filename_savings_year_percentage = 'savings_year_percentage_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'
        filename_efficiency = 'efficiency_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'
        filename_investment = 'investment_' + date_hr_min_sec + (' - ' + str(int(interest_rate * 100.0)) + '%') + '.csv'

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
            cobmo.utils_bes_cases.plot_battery_cases_bubbles(
                case,
                filepath_read,
                save_path_plots,
                filename_plot,
                labels=what_plotting,
                savepdf=0,
                pricing_method=pricing_method
            )
        tkinter.messagebox.showinfo("Simulation status", "Simulation concluded.\nSimulations = {} x 2".format(
            int(float(techs.shape[0]) * float(years.shape[0])))
                                    )

if do_plotting == 1 and simulate == 0:  # Only plotting

    inter = '2'     # << INPUT BY HAND
    #  -->
    datetime_path = (
        '2019-09-06_17-17-36'   # << INPUT BY HAND
        ' - '
        + inter
        + '%'
    )
    save_path = os.path.join(
        cobmo_cobmo_path,
        'results',
        'results_bes_cases',
        case,
        pricing_method,
        case + '_' + pricing_method + '_' + datetime_path
    )
    for what_plotting in plotting_options:
        read_path = os.path.join(
            save_path,
            what_plotting + '_' + datetime_path
            + '.csv'
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

        cobmo.utils_bes_cases.plot_battery_cases_bubbles(
            case,
            read_path,
            save_path_plots,
            filename_plot,
            labels=what_plotting,
            savepdf=1,
            pricing_method=pricing_method
        )
