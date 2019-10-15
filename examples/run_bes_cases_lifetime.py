"""Run script for BES cases lifetime."""

import os
import pandas as pd
import time as time

import cobmo.building
import cobmo.config
import cobmo.controller_bes_lifetime
import cobmo.database_interface
import cobmo.utils_bes_cases

# Settings.
scenario_name = 'scenario_default'
price_type = 'wholesale_market'
# Choices: 'wholesale_market', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40',
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'
case = 'best'  # IRENA battery parameters case.
# Choices: 'best', 'reference', 'worst'
interest_rate = 0.02
what_plotting = 'savings_year_percentage'
# Choices: 'savings_year', 'savings_year_percentage', 'storage_size', 'simple_payback', 'discounted_payback',
# 'efficiency', 'investment'
plotting = 1
simulate = 1

# Set results path and create the directory.
results_path = (
    os.path.join(
        cobmo.config.results_path,
        'run_bes_cases__' + case + '__' + '{:.2f}'.format(interest_rate) + '__' + price_type + '__' + what_plotting + '__' + cobmo.config.timestamp
    )
)
os.mkdir(results_path)

if simulate == 1:
    # Print status message.
    print("Simulation options:")
    print("- Case: {}".format(case))
    print("- Price type: {}".format(price_type))
    print("- Interest rate: {:.2f}".format(interest_rate))

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
    building_storage_types = pd.read_sql(
        """
        SELECT * FROM building_storage_types
        """,
        conn,
        index_col='building_storage_type'
    )
    building_parameter_sets = pd.read_sql(
        """
        SELECT * FROM building_parameter_sets
        """,
        conn,
        index_col='parameter_name'
    )

    # Modify `building_storage_type` for the current scenario in the database.
    building_name = building_scenarios.at[scenario_name, 'building_name']
    buildings.at[building_name, 'building_storage_type'] = 'battery_storage_default'
    buildings.to_sql(
        'buildings',
        con=conn,
        if_exists='replace'
    )

    # Modify `price_type` for current scenario in the database.
    building_scenarios.at[scenario_name, 'price_type'] = price_type
    building_scenarios.to_sql(
        'building_scenarios',
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

    # Retrieving the tech names from either of the DataFrames (they all have the same Indexes)
    # Obtain the indexes.
    set_technologies = battery_params_2016.index  # Retrieve technology names (all dataframes have the same indexes).
    set_years = pd.Index(['2016', '2020', '2025', '2030'])

    # Redefining columns for plotting
    efficiency.columns = set_years
    energy_cost.columns = set_years

    # Initialize dataframes to store results
    simple_payback_df = pd.DataFrame(
        0.0,
        index=set_technologies,
        columns=set_years
    )
    discounted_payback_df = pd.DataFrame(
        0.0,
        index=set_technologies,
        columns=set_years
    )
    storage_size_df = pd.DataFrame(
        0.0,
        index=set_technologies,
        columns=set_years
    )
    savings_year_df = pd.DataFrame(
        0.0,
        index=set_technologies,
        columns=set_years
    )
    savings_year_percentage_df = pd.DataFrame(
        0.0,
        index=set_technologies,
        columns=set_years
    )

    time_start = time.clock()
    counter = 0
    for technology in set_technologies:
        for year, i_year in zip(set_years, range(len(set_years))):
            counter += 1

            # Print status info.
            print("Starting simulation #{}.".format(counter))

            # Modify `building_storage_types` for the current scenario in the database.
            building_storage_types.at['battery_storage_default', 'storage_round_trip_efficiency'] = (
                float(efficiency.iloc[set_technologies.str.contains(technology), i_year])
                * 0.95  # Accounting for inverter efficiency. # TODO: Move inverter efficiency to CSV.
            )
            building_storage_types.at['battery_storage_default', 'storage_depth_of_discharge'] = (
                float(depth_of_discharge.iloc[set_technologies.str.contains(technology), i_year])
            )
            building_storage_types.at['battery_storage_default', 'storage_investment_sgd_per_unit'] = (
                float(energy_cost.iloc[set_technologies.str.contains(technology), i_year])
            )
            building_storage_types.at['battery_storage_default', 'storage_power_installation_cost'] = (
                float(power_cost.iloc[set_technologies.str.contains(technology), i_year])
            )
            building_storage_types.at['battery_storage_default', 'storage_lifetime'] = (
                float(lifetime.iloc[set_technologies.str.contains(technology), i_year])
            )
            building_storage_types.to_sql(
                'building_storage_types',
                con=conn,
                if_exists='replace'
            )

            # Modify `building_parameter_sets` to change the storage lifetime.
            # TODO: Change storage lifetime without using the parameters table.
            building_parameter_sets.loc['storage_lifetime', 'parameter_value'] = (
                float(building_storage_types.at['battery_storage_default', 'storage_lifetime'])
            )
            building_parameter_sets.to_sql(
                'building_parameter_sets',
                con=conn,
                if_exists='replace'
            )

            # Baseline case.
            # Print status info.
            print("Starting baseline case.")

            # Modify `building_storage_type` for the baseline case.
            building_name = building_scenarios.at[scenario_name, 'building_name']
            buildings.at[building_name, 'building_storage_type'] = ''
            buildings.to_sql(
                'buildings',
                con=conn,
                if_exists='replace'
            )

            # Obtain building model object for the baseline case.
            building_baseline = cobmo.building.Building(conn, scenario_name)

            # Run controller for the baseline case.
            # TODO: Solve baseline with normal controller.
            controller_baseline = cobmo.controller_bes_lifetime.Controller_bes_lifetime(conn=conn, building=building_baseline)
            (
                control_timeseries_baseline,
                state_timeseries_baseline,
                output_timeseries_baseline,
                storage_size_baseline,
                optimum_obj_baseline
            ) = controller_baseline.solve()

            # Storage case.
            # Print status info.
            print("Starting storage case.")

            # Modify `building_storage_type` for the storage case.
            building_name = building_scenarios.at[scenario_name, 'building_name']
            buildings.at[building_name, 'building_storage_type'] = 'battery_storage_default'
            buildings.to_sql(
                'buildings',
                con=conn,
                if_exists='replace'
            )

            # Obtain building model object for the storage case.
            building_storage = cobmo.building.Building(conn, scenario_name)

            # Run controller for the storage case.
            controller_storage = cobmo.controller_bes_lifetime.Controller_bes_lifetime(conn=conn, building=building_storage)
            (
                control_timeseries_storage,
                state_timeseries_storage,
                output_timeseries_storage,
                storage_size_storage,
                optimimum_obj_storage
            ) = controller_storage.solve()

            # Calculate savings and payback time
            storage_size_kwh = storage_size_storage / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
            costs_year_baseline = optimum_obj_baseline * 260.0  # 260 working days per year.
            savings_day = optimum_obj_baseline - optimimum_obj_storage
            savings_year = savings_day * 260.0  # 260 working days per year.
            storage_lifetime = lifetime.iloc[set_technologies.str.contains(technology), i_year]
            if float(savings_day) > 0.0:  # TODO: Payback function should internally manage those zero payback cases.
                (
                    discounted_payback,
                    simple_payback,
                    payback_df
                ) = cobmo.utils_bes_cases.discounted_payback_time(
                    building_storage,
                    storage_size_kwh,
                    storage_lifetime,
                    savings_day,
                    interest_rate=interest_rate,
                    plotting_on_off=0,  # No intermediate plots.
                    save_plot_on_off='off'  # "on" to save plot as SVG.
                )
            else:
                discounted_payback = 0.0
                simple_payback = 0.0
            if discounted_payback == 0.0:
                savings_year = 0.0
            savings_year_percentage = float(savings_year / costs_year_baseline * 100.0)

            # Store results.
            simple_payback_df.loc[technology, year] = simple_payback
            discounted_payback_df.loc[technology, year] = discounted_payback
            storage_size_df.loc[technology, year] = format(storage_size_kwh, '.2f')
            savings_year_df.loc[technology, year] = format(savings_year, '.1f')
            savings_year_percentage_df.loc[technology, year] = format(savings_year_percentage, '.1f')

    # Print status info.
    print("Simulation total solve time: {:.2f} minutes".format((time.clock() - time_start) / 60.0))

    # Amend efficiency and investment dataframes where there is no payback.
    efficiency[discounted_payback_df == 0.0] = 0.0
    energy_cost[discounted_payback_df == 0.0] = 0.0

    # Save results to CSV.
    simple_payback_df.to_csv(os.path.join(results_path, 'simple_payback.csv'))
    discounted_payback_df.to_csv(os.path.join(results_path, 'discounted_payback.csv'))
    storage_size_df.to_csv(os.path.join(results_path, 'storage_size.csv'))
    savings_year_df.to_csv(os.path.join(results_path, 'savings_year.csv'))
    savings_year_percentage_df.to_csv(os.path.join(results_path, 'savings_year_percentage.csv'))
    efficiency.to_csv(os.path.join(results_path, 'efficiency.csv'))
    energy_cost.to_csv(os.path.join(results_path, 'energy_cost'))

# Plots.
if plotting == 1:
    # Please note: Change `filepath_read` manually for plotting without simulation.
    filepath_read = os.path.join(results_path, what_plotting + '.csv')
    filename_plot = what_plotting
    cobmo.utils_bes_cases.plot_battery_cases_bubbles(
        case,
        filepath_read,
        results_path,
        filename_plot,
        labels=what_plotting,
        savepdf=1,
        pricing_method=price_type
    )
