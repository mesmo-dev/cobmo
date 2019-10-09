"""Run script for BES cases."""

import datetime
import numpy as np
import pandas as pd
import time

import cobmo.building
import cobmo.controller_bes
import cobmo.database_interface
import cobmo.utils_bes_cases


# Settings.
scenario_name = 'scenario_default'
do_plotting = 1
payback_type = 'simple'
iterate = 0
case = 'best'  # Choices: 'best', 'reference'

# Obtain a connection to the database.
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

# Modify `building_storage_type` for the current scenario in the database.
building_name = building_scenarios.at[scenario_name, 'building_name']
buildings.at[building_name, 'building_storage_type'] = 'battery_storage_default'
buildings.to_sql(
    'buildings',
    con=conn,
    if_exists='replace'
)

# Obtain battery parameters.
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

lifetime.to_csv('results/results_bes_cases/' + case + '/lifetime_' + case + '.csv')

# Retrieving the tech names from either of the dataframes (all have the same indexes).
techs = battery_params_2016.index
years = pd.Index(['2016', '2020', '2025', '2030'])

# Initialize dataframes to store results.
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

if iterate == 1:
    time_start_cycle = time.clock()
    for t in techs:
        building_storage_types = pd.read_sql(
            """
            SELECT * FROM building_storage_types
            """,
            conn,
            index_col='building_storage_type'
        )

        for i in range(energy_cost.shape[1]):
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

            # Write back into the database.
            building_storage_types.to_sql(
                'building_storage_types',
                con=conn,
                if_exists='replace'
            )

            # Get the building model.
            building = cobmo.building.Building(conn=conn, scenario_name=scenario_name)

            # Run controller.
            controller = cobmo.controller_bes.Controller_bes(conn=conn, building=building)
            (_, _, _, storage_size, optimum_obj) = controller.solve()

            # Calculating the savings and the payback time.
            storage_size_kwh = storage_size * 3.6e-3 * 1.0e-3
            costs_without_storage = 3.834195403e+02
            savings_day = (
                costs_without_storage
                - optimum_obj
            )

            # Running discounted payback function.
            (discounted_payback, simple_payback, _) = cobmo.utils_bes_cases.discounted_payback_time(
                building,
                storage_size_kwh,
                lifetime.iloc[techs.str.contains(t), i],  # storage lifetime as input
                savings_day,
                save_plot_on_off='off',  # "on" to save the plot as .svg (not tracked by the git)
                plotting_on_off=0  # set 1 for plotting
            )

            # Storing results.
            simple_payback_df.iloc[techs.str.contains(t), i] = simple_payback
            discounted_payback_df.iloc[techs.str.contains(t), i] = discounted_payback

    print("\nTime to solve all techs and all years: {:.2f} minutes".format((time.clock() - time_start_cycle)/60.0))

    date_main = datetime.datetime.now()
    filename_simple = 'simple_payback_' + case + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                date_main.year, date_main.month, date_main.day,
                date_main.hour, date_main.minute, date_main.second) + '.csv'
    filename_discounted = 'discounted_payback_' + case + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
        date_main.year, date_main.month, date_main.day,
        date_main.hour, date_main.minute, date_main.second) + '.csv'

    simple_payback_df.to_csv('results/results_bes_cases/' + case + '/' + filename_simple)
    discounted_payback_df.to_csv('results/results_bes_cases/' + case + '/' + filename_discounted)

if do_plotting == 1:
    filepath_read = (
            'results/results_bes_cases/'
            + case
            + '/'
            + payback_type
            + '_payback_best - 2019_08_29 - 14_33_26.csv'
    )

    savepath = 'results/results_bes_cases/best/plots-2019-08-29_14-33-26/'

    cobmo.utils_bes_cases.plot_battery_cases(
        case,
        'simple',
        filepath_read,
        savepath,
        save_plots='summary'
    )
