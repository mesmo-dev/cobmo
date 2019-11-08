"""Run script for evaluating demand side flexibility in terms of price sensitivity."""

import hvplot
import hvplot.pandas
import numpy as np
import os
import pandas as pd

import cobmo.building
import cobmo.controller
import cobmo.database_interface
import cobmo.utils


# Set `scenario_name`.
scenario_name = 'scenario_default'

# Set results path and create the directory.
results_path = os.path.join(cobmo.config.results_path, 'run_evaluation_price_sensitivity_' + cobmo.config.timestamp)
os.mkdir(results_path)
os.mkdir(os.path.join(results_path, 'plots'))
os.mkdir(os.path.join(results_path, 'details'))

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

# Instantiate load reduction iteration variables.
set_price_factors = pd.Index(np.concatenate([np.arange(0.0, 1.0, 0.2), np.arange(1.0, 105.0, 5.0)]))
set_timesteps = building.set_timesteps
load_change_percent_results = pd.DataFrame(
    None,
    set_timesteps,
    set_price_factors
)

# Iterate load reduction calculation.
for price_factor in set_price_factors:
    for timestep in set_timesteps:
        # TODO: Check if this condition is necessary.
        if (
            output_timeseries_baseline.loc[timestep, output_timeseries_baseline.columns.str.contains('electric_power')]
            == 0.0
        ).all():
            continue  # Skip loop if there is no baseline demand in the start timestep (no reduction possible).
        else:
            # Print status info.
            print("Calculate price sensitivity for: price_factor = {} / timestep = {}".format(price_factor, timestep))

            # Run controller for load reduction case.
            controller_price_sensitivity = cobmo.controller.Controller(
                conn=conn,
                building=building,
                problem_type='price_sensitivity',
                price_sensitivity_factor=price_factor,
                price_sensitivity_timestep=timestep
            )
            (
                control_timeseries_price_sensitivity,
                state_timeseries_price_sensitivity,
                output_timeseries_price_sensitivity,
                operation_cost_price_sensitivity,
                investment_cost_price_sensitivity,  # Represents load reduction.
                storage_size_price_sensitivity
            ) = controller_price_sensitivity.solve()

            # Save controller timeseries to CSV for debugging.
            control_timeseries_price_sensitivity.to_csv(os.path.join(
                results_path, 'details', '{} - {} control_timeseries.csv'.format(price_factor, timestep).replace(':', '-')
            ))
            state_timeseries_price_sensitivity.to_csv(os.path.join(
                results_path, 'details', '{} - {} state_timeseries.csv'.format(price_factor, timestep).replace(':', '-')
            ))
            output_timeseries_price_sensitivity.to_csv(os.path.join(
                results_path, 'details', '{} - {} output_timeseriesd.csv'.format(price_factor, timestep).replace(':', '-')
            ))

            # Plot demand comparison for debugging.
            electric_power_comparison = pd.concat(
                [
                    output_timeseries_baseline.loc[:, output_timeseries_baseline.columns.str.contains('electric_power')].sum(axis=1),
                    output_timeseries_price_sensitivity.loc[:, output_timeseries_price_sensitivity.columns.str.contains('electric_power')].sum(axis=1),
                ],
                keys=[
                    'baseline',
                    'price_sensitivity',
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
            ).hvplot.step(
                x='time',
                y='electric_power',
                by=['type'],
                **hvplot_default_options
            )

            # Define layout and labels / render plots.
            hvplot.save(
                (
                    electric_power_plot
                ).redim.label(
                    time="Date / time",
                    electric_power="Electric power [W]",
                ),
                # ).cols(1),
                # Plots open in are also stored in results directory.
                filename=os.path.join(
                    results_path, 'plots', '{} - {}.html'.format(price_factor, timestep).replace(':', '-')
                )
            )

            # Calculate results.
            baseline_power = (
                output_timeseries_baseline.loc[
                    timestep,
                    output_timeseries_baseline.columns.str.contains('electric_power')
                ].sum()
            )
            price_sensitivity_power = (
                output_timeseries_price_sensitivity.loc[
                    timestep,
                    output_timeseries_price_sensitivity.columns.str.contains('electric_power')
                ].sum()
            )
            load_change_percent = (
                (price_sensitivity_power - baseline_power)
                / baseline_power
                * 100.0
            )  # In percent.

            # Print results.
            print("load_change_percent = {}".format(load_change_percent))

            # Store results.
            load_change_percent_results.at[timestep, price_factor] = load_change_percent

# Aggregate load reduction results.
load_change_percent_mean = load_change_percent_results.mean()

# Print load reduction results for debugging.
print("load_change_percent_results = \n{}".format(load_change_percent_results))
print("load_change_percent_mean = \n{}".format(load_change_percent_mean))

# Save results to CSV.
load_change_percent_results.to_csv(os.path.join(results_path, 'load_change_percent_results.csv'))
load_change_percent_mean.to_csv(os.path.join(results_path, 'load_change_percent_mean.csv'))

# Print results path for debugging.
print("Results are stored in: " + results_path)
