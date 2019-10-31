"""Run script for generating demand side flexibility indicators."""

import hvplot
import hvplot.pandas
import os
import pandas as pd

import cobmo.building
import cobmo.controller
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'scenario_default'

# Set results path and create the directory.
results_path = os.path.join(cobmo.config.results_path, 'run_demand_side_flexibility_' + cobmo.config.timestamp)
os.mkdir(results_path)

# Obtain a connection to the database.
conn = cobmo.database_interface.connect_database()

# Define the building model (main function of the CoBMo toolbox).
# - Generates the building model for given `scenario_name` based on the building definitions in the `data` directory.
building = cobmo.building.Building(
    conn=conn,
    scenario_name=scenario_name
)

# Save building model matrices to CSV for debugging.
building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
building.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

# Run controller for baseline case.
controller_baseline = cobmo.controller.Controller(
    conn=conn,
    building=building
)
(
    control_timeseries_baseline,
    state_timeseries_baseline,
    output_timeseries_baseline,
    operation_cost_baseline,
    investment_cost_baseline,  # Zero when running (default) operation problem.
    storage_size_baseline  # Zero when running (default) operation problem.
) = controller_baseline.solve()

# Print operation cost for debugging.
print("operation_cost_baseline = {}".format(operation_cost_baseline))

# Save controller timeseries to CSV for debugging.
control_timeseries_baseline.to_csv(os.path.join(results_path, 'control_timeseries_baseline.csv'))
state_timeseries_baseline.to_csv(os.path.join(results_path, 'state_timeseries_baseline.csv'))
output_timeseries_baseline.to_csv(os.path.join(results_path, 'output_timeseries_baseline.csv'))

# Run controller for forced flexibility case.
controller_forced_flexibility = cobmo.controller.Controller(
    conn=conn,
    building=building,
    problem_type='forced_flexibility',
    output_timeseries_reference=output_timeseries_baseline,
    forced_flexibility_start_time=pd.to_datetime('2017-01-02T012:00:00'),
    forced_flexibility_end_time=pd.to_datetime('2017-01-02T012:00:00') + pd.to_timedelta('3h')
)
(
    control_timeseries_forced_flexibility,
    state_timeseries_forced_flexibility,
    output_timeseries_forced_flexibility,
    operation_cost_forced_flexibility,
    investment_cost_forced_flexibility,  # Zero when running (default) operation problem.
    storage_size_forced_flexibility  # Zero when running (default) operation problem.
) = controller_forced_flexibility.solve()

# Print investment cost for debugging.
print("investment_cost_forced_flexibility = {}".format(investment_cost_forced_flexibility))

# Save controller timeseries to CSV for debugging.
control_timeseries_forced_flexibility.to_csv(os.path.join(results_path, 'control_timeseries_forced_flexibility.csv'))
state_timeseries_forced_flexibility.to_csv(os.path.join(results_path, 'state_timeseries_forced_flexibility.csv'))
output_timeseries_forced_flexibility.to_csv(os.path.join(results_path, 'output_timeseries_forced_flexibility.csv'))

# Plot demand comparison for debugging.
electric_power_comparison = pd.concat(
    [
        output_timeseries_baseline.loc[:, output_timeseries_baseline.columns.str.contains('electric_power')].sum(axis=1),
        output_timeseries_forced_flexibility.loc[:, output_timeseries_forced_flexibility.columns.str.contains('electric_power')].sum(axis=1),
    ],
    keys=[
        'baseline',
        'forced_flexibility',
    ],
    names=[
        'type'
    ],
    axis=1
)

# Hvplot has no default options.
# Workaround: Pass this dict to every new plot.
hvplot_default_options = dict(width=1500, height=300)

electric_power_plot = (
    electric_power_comparison.stack().rename('electric_power').reset_index()
).hvplot.line(
    x='time',
    y='electric_power',
    by=['type'],
    **hvplot_default_options
)

# Define layout and labels / render plots.
hvplot.show(
    (
        electric_power_plot
    ).redim.label(
        time="Date / time",
        electric_power="Electric power [W]",
    ),
    # ).cols(1),
    # Plots open in browser and are also stored in results directory.
    filename=os.path.join(results_path, 'plots.html')
)

# Print results path for debugging.
print("Results are stored in: " + results_path)
