"""Run script for single simulation / optimization of sensible thermal or battery storage."""

import os
import pandas as pd

import cobmo.building
import cobmo.config
import cobmo.controller_baseline
import cobmo.controller_sensible
import cobmo.controller_bes
import cobmo.database_interface
import cobmo.utils


# Settings.
scenario_name = 'scenario_default'
pricing_method = 'wholesale_market'
# Choices: 'wholesale_market', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40'
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'
building_storage_type = 'sensible_thermal_storage_default'
# Choices: 'sensible_thermal_storage_default', 'battery_storage_default'
plotting = 1
save_csv = 1
save_plot = 1

# Check for valid settings.
if building_storage_type not in ['sensible_thermal_storage_default', 'battery_storage_default']:
    raise ValueError("No valid building_storage_type = '{}'".format(building_storage_type))

# Set results path and create the directory.
results_path = (
    os.path.join(
        cobmo.config.results_path,
        'run_storage_single__' + building_storage_type + '__' + pricing_method + '__' + cobmo.config.timestamp
    )
)
if (save_csv == 1) or (save_plot == 1):
    os.mkdir(results_path)

# Print the settings.
print("building_storage_type = {}".format(building_storage_type))
print("pricing_method = {}".format(pricing_method))

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

# Modify `building_parameter_sets` to change the storage lifetime.
# TODO: Change storage lifetime without using the parameters table.
building_parameter_sets.loc['storage_lifetime', 'parameter_value'] = (
    float(building_storage_types.at['sensible_thermal_storage_default', 'storage_lifetime'])
)
building_parameter_sets.to_sql(
    'building_parameter_sets',
    con=conn,
    if_exists='replace'
)

# Modify `price_type` for current scenario in the database.
building_scenarios.at[scenario_name, 'price_type'] = pricing_method
building_scenarios.to_sql(
    'building_scenarios',
    con=conn,
    if_exists='replace'
)

#
# Baseline case.
#

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
controller_sensible = cobmo.controller_baseline.ControllerBaseline(
    conn=conn,
    building=building_baseline
)
(
    control_timeseries_controller_baseline,
    state_timeseries_controller_baseline,
    output_timeseries_controller_baseline,
    optimum_obj_baseline
) = controller_sensible.solve()

#
# Storage case.
#

# Modify `building_storage_type` for the storage case.
building_name = building_scenarios.at[scenario_name, 'building_name']
buildings.at[building_name, 'building_storage_type'] = building_storage_type
buildings.to_sql(
    'buildings',
    con=conn,
    if_exists='replace'
)

# Obtain building model object for the storage case.
building_storage = cobmo.building.Building(conn, scenario_name)
building_storage.define_augmented_model()

# Run controller for the storage case.
if 'sensible' in building_storage_type:
    controller_sensible = cobmo.controller_sensible.Controller_sensible(
        conn=conn,
        building=building_storage
    )
    (
        control_timeseries_controller_storage,
        state_timeseries_controller_storage,
        output_timeseries_controller_storage,
        storage_size,
        optimum_obj_storage
    ) = controller_sensible.solve()

elif 'battery' in building_storage_type:
    controller_battery = cobmo.controller_bes.Controller_bes(
        conn=conn,
        building=building_storage
    )
    (
        control_timeseries_controller_storage,
        state_timeseries_controller_storage,
        output_timeseries_controller_storage,
        storage_size,
        optimum_obj_storage
    ) = controller_battery.solve()

#
# Outputs.
#

if storage_size != 0.0:
    # Calculate savings and payback time.
    if 'sensible' in building_storage_type:
        savings_day = (optimum_obj_baseline - optimum_obj_storage)
        if building_storage.building_scenarios['investment_sgd_per_X'][0] == 'kwh':
            storage_size = storage_size * 1000.0 * 4186.0 * 8.0 * 2.77778e-7
            # TODO: Take value from building_sensible.
            print("Storage size = {} kWh".format(round(storage_size, 2)))
        elif building_storage.building_scenarios['investment_sgd_per_X'][0] == 'm3':
            print("Storage size = {} m3".format(round(storage_size, 2)))
        else:
            raise ValueError("No valid specific unit of the storage investment.")

    elif 'battery' in building_storage_type:
        savings_day = (optimum_obj_baseline - optimum_obj_storage)
        storage_size = storage_size * 2.77778e-7  # * 3.6e-3 * 1.0e-3 # TODO: Why?

    (simple_payback, discounted_payback) = cobmo.utils.discounted_payback_time(
        building_storage,
        storage_size,
        savings_day,
        save_plot_on_off=save_plot,
        save_path=results_path,
        plotting_on_off=plotting,
        storage=building_storage_type,
        pricing_method=pricing_method,
        interest_rate=0.06
    )

    # Print results.
    print("Storage type = {}".format(building_storage_type))
    print("Optimal storage size = {}".format(round(storage_size, 2)))
    print("Savings per year ~= {}".format(round(savings_day * 260.0, 2)))
    print("Discounted payback = {}".format(round(discounted_payback)))
    print("Total OPEX + CAPEX with storage = {}".format(round(optimum_obj_storage, 2)))

else:
    # Print results.
    print("No storage installed:")
    print("Storage size = {}".format(round(storage_size, 2)))

# Save results to CSV.
if save_csv == 1:
    building_baseline.state_matrix.to_csv(os.path.join(results_path, 'building_baseline_state_matrix.csv'))
    building_baseline.control_matrix.to_csv(os.path.join(results_path, 'building_baseline_control_matrix.csv'))
    building_baseline.disturbance_matrix.to_csv(os.path.join(results_path, 'building_baseline_disturbance_matrix.csv'))
    building_baseline.state_output_matrix.to_csv(os.path.join(results_path, 'building_baseline_state_output_matrix.csv'))
    building_baseline.control_output_matrix.to_csv(os.path.join(results_path, 'building_baseline_control_output_matrix.csv'))
    building_baseline.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_baseline_disturbance_output_matrix.csv'))

    control_timeseries_controller_storage.to_csv(os.path.join(results_path, 'sttorage_control_timeseries_controller.csv'))
    state_timeseries_controller_storage.to_csv(os.path.join(results_path, 'sttorage_state_timeseries_controller.csv'))
    output_timeseries_controller_storage.to_csv(os.path.join(results_path, 'sttorage_output_timeseries_controller.csv'))

    building_storage.state_matrix.to_csv(os.path.join(results_path, 'building_storage_state_matrix.csv'))
    building_storage.control_matrix.to_csv(os.path.join(results_path, 'building_storage_control_matrix.csv'))
    building_storage.disturbance_matrix.to_csv(os.path.join(results_path, 'building_storage_disturbance_matrix.csv'))
    building_storage.state_output_matrix.to_csv(os.path.join(results_path, 'building_storage_state_output_matrix.csv'))
    building_storage.control_output_matrix.to_csv(os.path.join(results_path, 'building_storage_control_output_matrix.csv'))
    building_storage.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_storage_disturbance_output_matrix.csv'))

    control_timeseries_controller_baseline.to_csv(os.path.join(results_path, 'baseline_control_timeseries_controller.csv'))
    state_timeseries_controller_baseline.to_csv(os.path.join(results_path, 'baseline_state_timeseries_controller.csv'))
    output_timeseries_controller_baseline.to_csv(os.path.join(results_path, 'baseline_output_timeseries_controller.csv'))
