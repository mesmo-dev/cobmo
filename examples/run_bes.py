"""Run script for single simulation / optimization of battery storage."""

import numpy as np
import os
import pandas as pd

import cobmo.building
import cobmo.config
import cobmo.controller_bes
import cobmo.database_interface
import cobmo.utils


# Settings.
scenario_name = 'scenario_default'
pricing_method = 'wholesale_market'  # Choices: 'wholesale_market', 'retailer_peak_offpeak'.
building_storage_type = 'battery_storage_default'  # Choices: 'battery_storage_default', ''.
save_csv = 1
plotting = 0
save_plot = 0

# Set results path and create the directory.
# TODO: Throw error when thermal instead of battery storage.
if save_csv == 1:
    if building_storage_type == 'battery_storage_default':
        results_path = os.path.join(cobmo.config.results_path, 'run_bes__with_storage_' + cobmo.config.timestamp)
    else:
        results_path = os.path.join(cobmo.config.results_path, 'run_bes__without_storage_' + cobmo.config.timestamp)
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

# Modify `building_storage_type` for the current scenario in the database.
building_name = building_scenarios.at[scenario_name, 'building_name']
buildings.at[building_name, 'building_storage_type'] = building_storage_type
buildings.to_sql(
    'buildings',
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

# Obtain building model object.
# - Note: All changes to the database need to be done before loading the building model.
building = cobmo.building.Building(conn, scenario_name)

# Define initial state and control timeseries.
state_initial = building.set_state_initial
control_timeseries_simulation = pd.DataFrame(
    np.random.rand(len(building.set_timesteps), len(building.set_controls)),
    building.set_timesteps,
    building.set_controls
)

# Run simulation.
(
    state_timeseries_simulation,
    output_timeseries_simulation
) = building.simulate(
    state_initial=state_initial,
    control_timeseries=control_timeseries_simulation
)

# Run controller.
controller = cobmo.controller_bes.Controller_bes(
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

if building_storage_type == 'battery_storage_default':
    # Calculate savings and payback time.
    storage_size_kwh = storage_size * 3.6e-3 * 1.0e-3  # Ws to kWh
    costs_without_storage = 3.834195403e+02  # [SGD/day], 14 levels, for CREATE Tower. # TODO: should not be hardcoded.
    savings_day = (costs_without_storage - optimum_obj)
    (payback, payback_df) = cobmo.utils.discounted_payback_time(
        building,
        storage_size_kwh,
        savings_day,
        save_plot_on_off=save_plot,
        plotting_on_off=plotting,
        storage='battery'
    )

    # Print results.
    print("Storage size = {} kWh".format(round(storage_size_kwh, 2)))
    print("Total OPEX + CAPEX with storage = {}".format(round(optimum_obj, 2)))
    print("Storage type = {}".format(building.building_scenarios['building_storage_type'][0]))
    print("Optimal storage size = {}".format(round(storage_size, 2)))
    print("Savings per year ~= {}".format(round(savings_day * 260.0, 2)))
    print("Discounted payback = {}".format(round(payback)))
else:
    # Print results.
    print("Total OPEX without storage = {}".format(round(optimum_obj, 2)))

# Save results to CSV.
if save_csv == 1:
    building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
    building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
    building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
    building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
    building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
    building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))

    control_timeseries_simulation.to_csv(os.path.join(results_path, 'control_timeseries_simulation.csv'))
    state_timeseries_simulation.to_csv(os.path.join(results_path, 'state_timeseries_simulation.csv'))
    output_timeseries_simulation.to_csv(os.path.join(results_path, 'output_timeseries_simulation.csv'))

    control_timeseries_controller.to_csv(os.path.join(results_path, 'control_timeseries_controller.csv'))
    state_timeseries_controller.to_csv(os.path.join(results_path, 'state_timeseries_controller.csv'))
    output_timeseries_controller.to_csv(os.path.join(results_path, 'output_timeseries_controller.csv'))

# Print results path.
if save_csv == 1:
    print("Results are stored in: " + results_path)
