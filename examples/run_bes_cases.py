"""Run script for BES cases."""

import os
import pandas as pd
import time

import cobmo.building
import cobmo.config
import cobmo.controller_bes
import cobmo.database_interface
import cobmo.plots
import cobmo.utils


# Settings.
scenario_name = 'scenario_default'
price_type = 'wholesale_market'
# Choices: 'wholesale_market', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40'
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'
case = 'best'  # Choices: 'best', 'reference'
iterate = 1
plotting = 1

# Set results path and create the directory.
results_path = (
    os.path.join(
        cobmo.config.results_path,
        'run_bes_cases__' + case + '__' + price_type + '__' + cobmo.config.timestamp
    )
)
os.mkdir(results_path)

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
) = cobmo.utils.retrieve_battery_parameters(
    case=case
)

# Save lifetime to CSV for debugging.
lifetime.to_csv(os.path.join(results_path, 'lifetime.csv'))

# Obtain the indexes.
set_technologies = battery_params_2016.index  # Retrieve technology names (all dataframes have the same indexes).
set_years = pd.Index(['2016', '2020', '2025', '2030'])

if iterate == 1:
    time_start = time.clock()

    # Initialize dataframes to store results.
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

    for technology in set_technologies:
        for year, i_year in zip(set_years, range(len(set_years))):
            # Modify `building_storage_types` for the current scenario in the database.
            building_storage_types.at['battery_storage_default', 'storage_round_trip_efficiency'] = (
                float(efficiency.iloc[set_technologies.str.contains(technology), i_year])
            )
            building_storage_types.at['battery_storage_default', 'storage_depth_of_discharge'] = (
                float(depth_of_discharge.iloc[set_technologies.str.contains(technology), 1])
            )
            building_storage_types.at['battery_storage_default', 'storage_investment_sgd_per_unit'] = (
                float(energy_cost.iloc[set_technologies.str.contains(technology), i_year])
            )
            building_storage_types.at['battery_storage_default', 'storage_power_installation_cost'] = (
                float(power_cost.iloc[set_technologies.str.contains(technology), i_year])
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

            # Obtain building model object.
            building = cobmo.building.Building(conn=conn, scenario_name=scenario_name)

            # Run controller.
            controller = cobmo.controller_bes.Controller_bes(conn=conn, building=building)
            (
                control_timeseries_controller,
                state_timeseries_controller,
                output_timeseries_controller,
                storage_size,
                objective_value
            ) = controller.solve()

            # Calculate the savings and the payback time.
            storage_size_kwh = storage_size / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
            costs_without_storage = 3.834195403e+02  # TODO: Calculate costs without storage dynamically.
            savings_day = costs_without_storage - objective_value
            storage_lifetime = lifetime.iloc[set_technologies.str.contains(technology), i_year]
            (
                discounted_payback,
                simple_payback,
                payback_df
            ) = cobmo.utils.discounted_payback_time(
                building,
                storage_size_kwh,
                storage_lifetime,
                savings_day,
                plotting_on_off=0,  # No intermediate plots.
                save_plot_on_off='off'  # "on" to save plot as SVG.
            )
            # TODO: Why not using same function as in `run_storage_single`?

            # Store results.
            simple_payback_df.loc[technology, year] = simple_payback
            discounted_payback_df.loc[technology, year] = discounted_payback

    print("Run BES cases solve time: {:.2f} minutes".format((time.clock() - time_start) / 60.0))

    # Save results to CSV.
    simple_payback_df.to_csv(os.path.join(results_path, 'simple_payback.csv'))
    discounted_payback_df.to_csv(os.path.join(results_path, 'discounted_payback.csv'))

if plotting == 1:
    for payback_type in ['simple', 'discounted']:
        filepath_read = os.path.join(results_path, payback_type + '_payback.csv')
        cobmo.plots.plot_battery_cases(
            case,
            payback_type,
            filepath_read,
            results_path,
            save_plots='summary'
        )
