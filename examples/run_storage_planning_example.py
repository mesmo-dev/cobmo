"""Run script for single simulation / optimization of sensible thermal or battery storage."""

import os
import pandas as pd

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.database_interface
import cobmo.utils


# Settings.
scenario_name = 'scenario_default'
price_type = 'wholesale_market'
# Choices: 'wholesale_market', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40'
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'
building_storage_type = 'default_battery_storage'
# Choices: 'default_sensible_thermal_storage', 'default_battery_storage'
save_plots = True
save_csv = True

# Set results path and create the directory.
results_path = (
    os.path.join(
        cobmo.config.results_path,
        'run_storage_single__' + building_storage_type + '__' + price_type + '__' + cobmo.config.timestamp
    )
)
if save_csv or save_plots:
    os.mkdir(results_path)

# Print settings.
print("building_storage_type = {}".format(building_storage_type))
print("pricing_method = {}".format(price_type))

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
buildings.at[scenarios.at[scenario_name, 'building_name'], 'building_storage_type'] = building_storage_type
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

# Baseline case.
# Print status info.
print('Starting baseline case.')

# Obtain building model object for the baseline case.
building_baseline = cobmo.building_model.BuildingModel(scenario_name, database_connection)

# Run controller for the baseline case.
controller_baseline = cobmo.optimization_problem.OptimizationProblem(
    database_connection=database_connection,
    building=building_baseline,
    problem_type='storage_planning_baseline'
)
(
    control_timeseries_controller_baseline,
    state_timeseries_controller_baseline,
    output_timeseries_controller_baseline,
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
    database_connection=database_connection,
    building=building_storage,
    problem_type='storage_planning'
)
(
    control_timeseries_controller_storage,
    state_timeseries_controller_storage,
    output_timeseries_controller_storage,
    operation_cost_storage,
    investment_cost_storage,
    storage_size_storage
) = controller_storage.solve()

# Calculate savings and payback time.
storage_lifetime = storage_types.at[building_storage_type, 'storage_lifetime']
operation_cost_savings_annual = (operation_cost_baseline - operation_cost_storage) / storage_lifetime
if 'sensible' in building_storage_type:
    storage_size_kwh = (
            storage_size_storage
            * 1000.0  # Density in kg/m3 (water).
            * 4186.0  # Specific heat capacity in J/(kg*K) (water)
            * storage_types.at[building_storage_type, 'storage_sensible_temperature_delta']  # Temp. dif. in K.
            / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
    )
elif 'battery' in building_storage_type:
    storage_size_kwh = storage_size_storage / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
(
    simple_payback_time,
    discounted_payback_time
) = cobmo.utils.calculate_discounted_payback_time(
    storage_lifetime,
    investment_cost_storage,
    operation_cost_storage,
    operation_cost_baseline,
    interest_rate=0.06,
    investment_type=building_storage_type,
    save_plots=save_plots,
    results_path=results_path,
    file_id=''
)

# Print results.
print("storage_size_kwh = {}".format(storage_size_kwh))
print("investment_cost_storage = {}".format(investment_cost_storage))
print("operation_cost_storage = {}".format(operation_cost_storage))
print("operation_cost_savings_annual = {}".format(operation_cost_savings_annual))
print("storage_lifetime = {}".format(storage_lifetime))
print("discounted_payback_time = {}".format(discounted_payback_time))

# Save results to CSV.
if save_csv:
    building_baseline.state_matrix.to_csv(os.path.join(results_path, 'building_baseline_state_matrix.csv'))
    building_baseline.control_matrix.to_csv(os.path.join(results_path, 'building_baseline_control_matrix.csv'))
    building_baseline.disturbance_matrix.to_csv(os.path.join(results_path, 'building_baseline_disturbance_matrix.csv'))
    building_baseline.state_output_matrix.to_csv(os.path.join(results_path, 'building_baseline_state_output_matrix.csv'))
    building_baseline.control_output_matrix.to_csv(os.path.join(results_path, 'building_baseline_control_output_matrix.csv'))
    building_baseline.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_baseline_disturbance_output_matrix.csv'))

    control_timeseries_controller_storage.to_csv(os.path.join(results_path, 'storage_control_timeseries_controller.csv'))
    state_timeseries_controller_storage.to_csv(os.path.join(results_path, 'storage_state_timeseries_controller.csv'))
    output_timeseries_controller_storage.to_csv(os.path.join(results_path, 'storage_output_timeseries_controller.csv'))

    building_storage.state_matrix.to_csv(os.path.join(results_path, 'building_storage_state_matrix.csv'))
    building_storage.control_matrix.to_csv(os.path.join(results_path, 'building_storage_control_matrix.csv'))
    building_storage.disturbance_matrix.to_csv(os.path.join(results_path, 'building_storage_disturbance_matrix.csv'))
    building_storage.state_output_matrix.to_csv(os.path.join(results_path, 'building_storage_state_output_matrix.csv'))
    building_storage.control_output_matrix.to_csv(os.path.join(results_path, 'building_storage_control_output_matrix.csv'))
    building_storage.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_storage_disturbance_output_matrix.csv'))

    control_timeseries_controller_baseline.to_csv(os.path.join(results_path, 'baseline_control_timeseries_controller.csv'))
    state_timeseries_controller_baseline.to_csv(os.path.join(results_path, 'baseline_state_timeseries_controller.csv'))
    output_timeseries_controller_baseline.to_csv(os.path.join(results_path, 'baseline_output_timeseries_controller.csv'))

# Print results path.
if save_csv or save_plots:
    print("Results are stored in: " + results_path)
