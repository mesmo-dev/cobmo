"""
Use this script for single simulations of SENSIBLE storage
"""

import datetime as dt
import errno
import numpy as np
import os
import pandas as pd

import cobmo.building
import cobmo.config
import cobmo.controller_baseline
import cobmo.controller_sensible
import cobmo.controller_bes
import cobmo.database_interface
import cobmo.utils


cobmo_cobmo_path = os.path.join(cobmo.config.cobmo_path, 'cobmo')

date_main = dt.datetime.now()
date_hr_min_sec = '{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
    date_main.year, date_main.month, date_main.day,
    date_main.hour, date_main.minute, date_main.second)

# Settings.
scenario_name = 'scenario_default'
pricing_method = 'wholesale_market'
# Options: 'wholesale_market', 'retailer_peak_offpeak', 'wholesale_squeezed_20', 'wholesale_squeezed_40'
# 'wholesale_squeezed_60', 'wholesale_squeezed_80'

storage = 'sensible'  # Options: 'sensible' or 'battery'

plotting = 0
print_on_csv = 0
save_plot = 0

print('\n________________________'
      '\nSimulation options:'
      '\n- Storage tech: *%s*'
      '\n- Price signal: *%s*'
      '\n________________________'
      % (storage, pricing_method))

storage_tech = storage
storage_type = ''
if storage_tech == 'sensible':
    storage_type = 'sensible_thermal_storage_default'
elif storage_tech == 'battery':
    storage_type = 'battery_storage_default'
else:
    print('\n{!} No valid storage technology input (storage = "%s"). Running baseline scenario against itself.\n'
          % storage)

conn = cobmo.database_interface.connect_database()

# Extracting tables from the sql
# CAREFUL!
# Indexing to allow precise modification of the dataframe.
# If this is used you need to reindex as pandas when using to_sql (meaning NOT using "index=False")
building_scenarios = pd.read_sql(
    """
    select * from building_scenarios
    """,
    conn,
    index_col='scenario_name'
)
buildings = pd.read_sql(
    """
    select * from buildings
    """,
    conn,
    index_col='building_name'
)
building_storage_types = pd.read_sql(
    """
    select * from building_storage_types
    """,
    conn,
    index_col='building_storage_type'
)

# Importing the parameter_sets to change the storage lifetime
building_parameter_sets = pd.read_sql(  # TODO: do this without using the parameters table.
    """
    select * from building_parameter_sets
    """,
    conn
)
position_lifetime = building_parameter_sets.index[
    building_parameter_sets['parameter_name'] == 'storage_lifetime'
    ].tolist()
building_parameter_sets.loc[position_lifetime, 'parameter_value'] = (
    float(building_storage_types.at['sensible_thermal_storage_default', 'storage_lifetime'])
)

building_parameter_sets.to_sql(
    'building_parameter_sets',
    con=conn,
    if_exists='replace',
    index=False
)

building_name = building_scenarios.at[scenario_name, 'building_name']

# Modify `price_type` for current scenario in the database.
building_scenarios.at[scenario_name, 'price_type'] = pricing_method
building_scenarios.to_sql(
    'building_scenarios',
    con=conn,
    if_exists='replace'
)

# ____________________________________________ Running BASELINE _________________________________________________

# Baseline scenario
buildings.at[building_name, 'building_storage_type'] = ''
buildings.to_sql(
    'buildings',
    con=conn,
    if_exists='replace'
    # index=False
)

building_baseline = cobmo.building.Building(conn, scenario_name)
building_baseline.define_augmented_model()

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

# ____________________________________________ Running STORAGE _________________________________________________

# Setting storage option for the building + getting another building + running controller again
buildings.at[building_name, 'building_storage_type'] = storage_type
buildings.to_sql(
    'buildings',
    con=conn,
    if_exists='replace'
    # index=False
)
building_storage = cobmo.building.Building(conn, scenario_name)
building_storage.define_augmented_model()

optimum_obj_sensible = optimum_obj_baseline
optimum_obj_battery = optimum_obj_baseline
storage_size = 0.0
control_timeseries_controller_storage = control_timeseries_controller_baseline
state_timeseries_controller_storage = state_timeseries_controller_baseline
output_timeseries_controller_storage = output_timeseries_controller_baseline

if storage_tech == 'sensible':
    controller_sensible = cobmo.controller_sensible.Controller_sensible(
        conn=conn,
        building=building_storage
    )
    (
        control_timeseries_controller_storage,
        state_timeseries_controller_storage,
        output_timeseries_controller_storage,
        storage_size,
        optimum_obj_sensible
    ) = controller_sensible.solve()

elif storage_tech == 'battery':
    controller_battery = cobmo.controller_bes.Controller_bes(
        conn=conn,
        building=building_storage
    )
    (
        control_timeseries_controller_storage,
        state_timeseries_controller_storage,
        output_timeseries_controller_storage,
        storage_size,
        optimum_obj_battery
    ) = controller_battery.solve()

# __________________________________________________ Payback _____________________________________________________

print('\n----------------------------------------------')
print('\n>> Total opex (storage)= {}'.format(format(optimum_obj_sensible, '.2f')))

savings_day = 0.0
if storage_size != 0.0:
    # Calculating the savings and the payback time

    if storage == 'sensible':
        savings_day = (optimum_obj_baseline - optimum_obj_sensible)
        if building_storage.building_scenarios['investment_sgd_per_X'][0] == 'kwh':
            storage_size = storage_size * 1000.0 * 4186.0 * 8.0 * 2.77778e-7
            # TODO: take value from building_sensible
            print('\n>> Storage size = %.2f kWh' % storage_size)
        elif building_storage.building_scenarios['investment_sgd_per_X'][0] == 'm3':
            print('\n>> Storage size = %.2f m3' % storage_size)
        else:
            print('\n Please define a specific unit of the storage investment')

    elif storage == 'battery':
        savings_day = (optimum_obj_baseline - optimum_obj_battery)
        storage_size = storage_size * 2.77778e-7  # * 3.6e-3 * 1.0e-3

# __________________________________________________ Plotting _____________________________________________________

    save_path = os.path.join(
        cobmo_cobmo_path,
        'results',
        'results_single_simulation/',
        storage + '/',
        pricing_method + '/'
    )
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    (simple_payback, discounted_payback) = cobmo.utils.discounted_payback_time(
        building_storage,
        storage_size,
        savings_day,
        save_plot_on_off=save_plot,
        save_path=save_path,
        plotting_on_off=plotting,
        storage=storage,
        pricing_method=pricing_method,
        interest_rate=0.06
    )

    print('\n>> Storage type = %s'
          '  |  Optimal storage size = %.2f'
          '  | savings year 〜= %.2f'
          '  | Discounted payback = %i\n'
          % (
            building_storage.building_scenarios['building_storage_type'][0],
            storage_size,
            savings_day * 260.0,
            discounted_payback
          )
          )
else:
    print('\n>> Storage size is 0.0 --> No storage is installed as it is not economically feasible.')

# Printing the outputs to dedicated csv files. These are IGNORED by the git
if print_on_csv == 1:

    # Storage scenario
    building_storage.state_matrix.to_csv('delete_me_storage/' + storage + '/state_matrix-'  + storage + '-' + pricing_method + '.csv')
    building_storage.control_matrix.to_csv('delete_me_storage/' + storage + '/control_matrix-'  + storage + '-' + pricing_method + '.csv')
    building_storage.disturbance_matrix.to_csv('delete_me_storage/' + storage + '/disturbance_matrix-'  + storage + '-' + pricing_method + '.csv')

    building_storage.state_output_matrix.to_csv('delete_me_storage/' + storage + '/state_output_matrix-'  + storage + '-' + pricing_method + '.csv')
    building_storage.control_output_matrix.to_csv('delete_me_storage/' + storage + '/control_output_matrix-'  + storage + '-' + pricing_method + '.csv')
    building_storage.disturbance_output_matrix.to_csv('delete_me_storage/' + storage + '/disturbance_output_matrix-'  + storage + '-' + pricing_method + '.csv')

    # state_timeseries_simulation_storage.to_csv('delete_me_storage/sensible/state_timeseries_simulation_SENSIBLE.csv')

    state_timeseries_controller_storage.to_csv('delete_me_storage/' + storage + '/state_timeseries_controller-' + storage + '-' + pricing_method + '.csv')

    date_main = dt.datetime.now()
    filename_out_controller = (
            'output_timeseries_controller-'  + storage + '-' + pricing_method + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                date_main.year, date_main.month, date_main.day, date_main.hour, date_main.minute,
                date_main.second)
            + '.csv'
    )
    output_timeseries_controller_storage.to_csv('delete_me_storage/' + storage + '/' + filename_out_controller)

    # control_timeseries_controller_storage.to_csv('delete_me_storage/' + storage + '/control_timeseries_controller_SENSIBLE.csv')

    # Baseline scenario
    building_baseline.state_matrix.to_csv('delete_me/state_matrix.csv')
    building_baseline.control_matrix.to_csv('delete_me/control_matrix.csv')
    building_baseline.disturbance_matrix.to_csv('delete_me/disturbance_matrix.csv')

    building_baseline.state_output_matrix.to_csv('delete_me/state_output_matrix.csv')
    building_baseline.control_output_matrix.to_csv('delete_me/control_output_matrix.csv')
    building_baseline.disturbance_output_matrix.to_csv('delete_me/disturbance_output_matrix.csv')

    # np.savetxt(r'my_file_output_state_matrix.txt', building.state_matrix.values) # , fmt='%d'
    # state_timeseries_simulation_baseline.to_csv('delete_me/state_timeseries_simulation.csv')

    state_timeseries_controller_baseline.to_csv('delete_me/state_timeseries_controller.csv')

    date_main = dt.datetime.now()
    filename_out_controller = (
            'output_timeseries_controller' + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                date_main.year, date_main.month, date_main.day, date_main.hour, date_main.minute,
                date_main.second)
            + '.csv'
    )
    output_timeseries_controller_baseline.to_csv('delete_me/' + filename_out_controller)
    control_timeseries_controller_baseline.to_csv('delete_me/control_timeseries_controller.csv')
