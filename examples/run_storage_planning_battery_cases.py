"""Run script for BES cases lifetime."""

import os
import pandas as pd
import time as time

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.database_interface
import cobmo.plots
import cobmo.utils

# Settings.
scenario_name = 'scenario_default'
price_type = 'wholesale'
# Choices: 'wholesale', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40',
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'
case = 'best'  # IRENA battery parameters case.
# Choices: 'best', 'reference', 'worst'
interest_rate = 0.02
save_plots = True
run_simulation = True  # Please note: Change `read_path` manually for plotting without simulation.

# Set results path and create the directory.
results_path = (
    os.path.join(
        cobmo.config.results_path,
        'run_bes_cases__' + case + '__' + '{:.2f}'.format(interest_rate) + '__' + price_type + '__' + cobmo.config.timestamp
    )
)
os.mkdir(results_path)

if run_simulation:
    # Print status message.
    print("Starting simulation.")

    # Print settings.
    print("case = {}".format(case))
    print("price_type = {}".format(price_type))
    print("interest_rate = {}".format(interest_rate))

    # Obtain a connection to the database.
    database_connection = cobmo.database_interface.connect_database()

    # Load selected database tables for modification.
    scenarios = pd.read_sql(
        """
        SELECT * FROM scenarios
        """,
        database_connection,
        index_col='scenario_name'
    )
    buildings = pd.read_sql(
        """
        SELECT * FROM buildings
        """,
        database_connection,
        index_col='building_name'
    )
    storage_types = pd.read_sql(
        """
        SELECT * FROM storage_types
        """,
        database_connection,
        index_col='building_storage_type'
    )

    # Modify `buildings` to change the `building_storage_type`.
    building_name = scenarios.at[scenario_name, 'building_name']
    buildings.at[building_name, 'building_storage_type'] = 'default_battery_storage'
    buildings.to_sql(
        'buildings',
        con=database_connection,
        if_exists='replace'
    )

    # Modify `scenarios` to change the `price_type`.
    scenarios.at[scenario_name, 'price_type'] = price_type
    scenarios.to_sql(
        'scenarios',
        con=database_connection,
        if_exists='replace'
    )

    # Obtain battery parameters.
    battery_parameters = cobmo.utils.get_battery_parameters()

    # Initialize dataframes to store results
    set_battery_technologies = battery_parameters.index.levels[0]
    set_years = battery_parameters.index.levels[1]
    simple_payback_time_results = pd.DataFrame(
        0.0,
        index=set_battery_technologies,
        columns=set_years
    )
    discounted_payback_time_results = pd.DataFrame(
        0.0,
        index=set_battery_technologies,
        columns=set_years
    )
    storage_size_results = pd.DataFrame(
        0.0,
        index=set_battery_technologies,
        columns=set_years
    )
    operation_cost_savings_annual_results = pd.DataFrame(
        0.0,
        index=set_battery_technologies,
        columns=set_years
    )
    operation_cost_savings_annual_percentage_results = pd.DataFrame(
        0.0,
        index=set_battery_technologies,
        columns=set_years
    )

    time_start = time.clock()
    counter = 0
    for battery_technology in set_battery_technologies:
        for year in set_years:
            counter += 1

            # Print status info.
            print("Starting simulation #{}.".format(counter))

            # Modify `storage_types`.
            storage_types.at['default_battery_storage', 'storage_round_trip_efficiency'] = (
                battery_parameters.loc[(battery_technology, year, case), 'round_trip_efficiency']
                * 0.95  # Accounting for inverter efficiency. # TODO: Move inverter efficiency to CSV.
            )
            storage_types.at['default_battery_storage', 'storage_battery_depth_of_discharge'] = (
                battery_parameters.loc[(battery_technology, year, case), 'depth_of_discharge']
            )
            storage_types.at['default_battery_storage', 'storage_planning_energy_installation_cost'] = (
                battery_parameters.loc[(battery_technology, year, case), 'energy_installation_cost']
            )
            storage_types.at['default_battery_storage', 'storage_planning_power_installation_cost'] = (
                battery_parameters.loc[(battery_technology, year, case), 'power_installation_cost']
            )
            storage_types.at['default_battery_storage', 'storage_lifetime'] = (
                battery_parameters.loc[(battery_technology, year, case), 'lifetime']
            )
            storage_types.to_sql(
                'storage_types',
                con=database_connection,
                if_exists='replace'
            )

            # Baseline case.
            # Print status info.
            print("Starting baseline case.")

            # Obtain building model object for the baseline case.
            building_baseline = cobmo.building_model.BuildingModel(scenario_name, database_connection)

            # Run controller for the baseline case.
            controller_baseline = cobmo.optimization_problem.OptimizationProblem(
                building_baseline,
                problem_type='storage_planning_baseline'
            )
            (
                control_vector_baseline,
                state_vector_baseline,
                output_vector_baseline,
                operation_cost_baseline,
                investment_cost_baseline,
                storage_size_baseline
            ) = controller_baseline.solve()

            # Print results.
            print("operation_cost_baseline = {}".format(operation_cost_baseline))

            # Storage case.
            # Print status info.
            print("Starting storage case.")

            # Obtain building model object for the storage case.
            building_storage = cobmo.building_model.BuildingModel(scenario_name, database_connection)

            # Run controller for the storage case.
            controller_storage = cobmo.optimization_problem.OptimizationProblem(
                building_storage,
                problem_type='storage_planning'
            )
            (
                control_vector_storage,
                state_vector_storage,
                output_vector_storage,
                operation_cost_storage,
                investment_cost_storage,
                storage_size_storage
            ) = controller_storage.solve()

            # Calculate savings and payback time.
            storage_size_kwh = storage_size_storage / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
            storage_lifetime = battery_parameters.loc[(battery_technology, year, case), 'lifetime']
            operation_cost_savings = operation_cost_baseline - operation_cost_storage
            operation_cost_savings_annual = operation_cost_savings / storage_lifetime
            operation_cost_savings_annual_percentage = operation_cost_savings_annual / operation_cost_baseline * 100.0
            (
                simple_payback_time,
                discounted_payback_time,
            ) = cobmo.utils.calculate_discounted_payback_time(
                storage_lifetime,
                investment_cost_storage,
                operation_cost_storage,
                operation_cost_baseline,
                interest_rate,
                investment_type='battery_storage',
                save_plots=save_plots,
                results_path=results_path,
                file_id='_{}_{}'.format(battery_technology, year)
            )

            # Print results.
            print("storage_size_kwh = {}".format(storage_size_kwh))
            print("investment_cost_storage = {}".format(investment_cost_storage))
            print("operation_cost_storage = {}".format(operation_cost_storage))
            print("operation_cost_savings_annual = {}".format(operation_cost_savings_annual))
            print("storage_lifetime = {}".format(storage_lifetime))
            print("discounted_payback_time = {}".format(discounted_payback_time))

            # Store results.
            simple_payback_time_results.loc[battery_technology, year] = (
                simple_payback_time
            )
            discounted_payback_time_results.loc[battery_technology, year] = (
                discounted_payback_time
            )
            storage_size_results.loc[battery_technology, year] = (
                format(storage_size_kwh, '.2f')
            )
            operation_cost_savings_annual_results.loc[battery_technology, year] = (
                format(operation_cost_savings_annual, '.1f')
            )
            operation_cost_savings_annual_percentage_results.loc[battery_technology, year] = (
                format(operation_cost_savings_annual_percentage, '.1f')
            )

    # Print status info.
    print("Simulation total solve time: {:.2f} minutes".format((time.clock() - time_start) / 60.0))

    # Create `efficiency` and `energy_cost` dataframes for plotting.
    efficiency = battery_parameters['round_trip_efficiency'][:, :, case].unstack()
    energy_cost = battery_parameters['energy_installation_cost'][:, :, case].unstack()

    # Modify entries without payback to improve plot readability.
    efficiency[discounted_payback_time_results.isnull()] = 0.0
    energy_cost[discounted_payback_time_results.isnull()] = 0.0
    operation_cost_savings_annual_results[discounted_payback_time_results.isnull()] = 0.0
    operation_cost_savings_annual_percentage_results[discounted_payback_time_results.isnull()] = 0.0

    # Save results to CSV.
    simple_payback_time_results.to_csv(
        os.path.join(results_path, 'simple_payback_time.csv')
    )
    discounted_payback_time_results.to_csv(
        os.path.join(results_path, 'discounted_payback_time.csv')
    )
    storage_size_results.to_csv(
        os.path.join(results_path, 'storage_size.csv')
    )
    efficiency.to_csv(
        os.path.join(results_path, 'efficiency.csv')
    )
    energy_cost.to_csv(
        os.path.join(results_path, 'energy_cost.csv')
    )
    operation_cost_savings_annual_results.to_csv(
        os.path.join(results_path, 'operation_cost_savings_annual.csv')
    )
    operation_cost_savings_annual_percentage_results.to_csv(
        os.path.join(results_path, 'operation_cost_savings_annual_percentage.csv')
    )

# Plots.
if save_plots:
    # Please note: Change `read_path` manually for plotting without simulation.

    for plot_type in [
        'simple_payback_time',
        'discounted_payback_time',
        'storage_size',
        'efficiency',
        'operation_cost_savings_annual',
        'operation_cost_savings_annual_percentage'
    ]:
        read_path = os.path.join(results_path, plot_type + '.csv')
        cobmo.plots.plot_battery_cases_comparisons_bubbles(
            read_path,
            results_path,
            plot_type
        )

    for plot_type in ['simple_payback_time', 'discounted_payback_time']:
        read_path = os.path.join(results_path, plot_type + '.csv')
        cobmo.plots.plot_battery_cases_payback_comparison_lines(
            read_path,
            results_path,
            plot_type
        )

# Print results path.
print("Results are stored in: " + results_path)
