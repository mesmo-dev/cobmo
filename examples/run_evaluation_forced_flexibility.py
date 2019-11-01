"""Run script for generating demand side flexibility indicators."""

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

# Instantiate forced flexibility iteration variables.
set_time_duration = (
    pd.Index([
        pd.to_timedelta('{}h'.format(time_duration))
        for time_duration in np.arange(0.5, 6.5, 0.5)
    ])
)
set_timesteps = building.set_timesteps
forced_flexibility_energy_results = pd.DataFrame(
    None,
    set_timesteps,
    set_time_duration
)
forced_flexibility_power_results = pd.DataFrame(
    None,
    set_timesteps,
    set_time_duration
)
forced_flexibility_percent_results = pd.DataFrame(
    None,
    set_timesteps,
    set_time_duration
)

# Iterate forced flexibility calculation.
for time_duration in set_time_duration:
    for timestep in set_timesteps:
        if (timestep + time_duration) > building.set_timesteps[-1]:
            break  # Interrupt loop if end time goes beyond building model time horizon.
        elif (
            output_timeseries_baseline.loc[timestep, output_timeseries_baseline.columns.str.contains('electric_power')]
            == 0.0
        ).all():
            continue  # Skip loop if there is no baseline demand in the start timestep (no reduction possible).
        else:
            # Print status info.
            print("Calculate forced flexibility for: time_duration = {} / timestep = {}".format(time_duration, timestep))

            # Run controller for forced flexibility case.
            controller_forced_flexibility = cobmo.controller.Controller(
                conn=conn,
                building=building,
                problem_type='forced_flexibility',
                output_timeseries_reference=output_timeseries_baseline,
                forced_flexibility_start_time=timestep,
                forced_flexibility_end_time=timestep + time_duration
            )
            (
                control_timeseries_forced_flexibility,
                state_timeseries_forced_flexibility,
                output_timeseries_forced_flexibility,
                operation_cost_forced_flexibility,
                investment_cost_forced_flexibility,  # Represents forced flexibility.
                storage_size_forced_flexibility
            ) = controller_forced_flexibility.solve()

            # # Save controller timeseries to CSV for debugging.
            # control_timeseries_forced_flexibility.to_csv(os.path.join(results_path, 'control_timeseries_forced_flexibility.csv'))
            # state_timeseries_forced_flexibility.to_csv(os.path.join(results_path, 'state_timeseries_forced_flexibility.csv'))
            # output_timeseries_forced_flexibility.to_csv(os.path.join(results_path, 'output_timeseries_forced_flexibility.csv'))
            #
            # # Plot demand comparison for debugging.
            # electric_power_comparison = pd.concat(
            #     [
            #         output_timeseries_baseline.loc[:, output_timeseries_baseline.columns.str.contains('electric_power')].sum(axis=1),
            #         output_timeseries_forced_flexibility.loc[:, output_timeseries_forced_flexibility.columns.str.contains('electric_power')].sum(axis=1),
            #     ],
            #     keys=[
            #         'baseline',
            #         'forced_flexibility',
            #     ],
            #     names=[
            #         'type'
            #     ],
            #     axis=1
            # )
            #
            # # Hvplot has no default options.
            # # Workaround: Pass this dict to every new plot.
            # hvplot_default_options = dict(width=1500, height=300)
            #
            # electric_power_plot = (
            #     electric_power_comparison.stack().rename('electric_power').reset_index()
            # ).hvplot.line(
            #     x='time',
            #     y='electric_power',
            #     by=['type'],
            #     **hvplot_default_options
            # )
            #
            # # Define layout and labels / render plots.
            # hvplot.show(
            #     (
            #         electric_power_plot
            #     ).redim.label(
            #         time="Date / time",
            #         electric_power="Electric power [W]",
            #     ),
            #     # ).cols(1),
            #     # Plots open in browser and are also stored in results directory.
            #     filename=os.path.join(results_path, 'plots.html')
            # )

            # Calculate results.
            # TODO: Move timestep_delta into building model.
            timestep_delta = building.set_timesteps[1] - building.set_timesteps[0]
            baseline_energy = (
                output_timeseries_baseline.loc[
                    timestep:(timestep + time_duration),
                    output_timeseries_baseline.columns.str.contains('electric_power')
                ].sum().sum()
                * timestep_delta.seconds / 3600.0 / 1000.0  # W in kWh.
            )
            forced_flexibility_energy = abs(investment_cost_forced_flexibility)  # in kWh.
            forced_flexibility_power = (
                forced_flexibility_energy
                / (time_duration.total_seconds() / 3600.0)  # kWh in kW.
            )
            forced_flexibility_percent = (
                forced_flexibility_energy
                / baseline_energy
                * 100.0
            )

            # Print results.
            print("forced_flexibility_energy = {}".format(forced_flexibility_energy))
            print("forced_flexibility_power = {}".format(forced_flexibility_power))
            print("forced_flexibility_percent = {}".format(forced_flexibility_percent))

            # Store results.
            forced_flexibility_energy_results.at[timestep, time_duration] = forced_flexibility_energy
            forced_flexibility_power_results.at[timestep, time_duration] = forced_flexibility_power
            forced_flexibility_percent_results.at[timestep, time_duration] = forced_flexibility_percent

# Aggregate forced flexibility results.
forced_flexibility_energy_mean = forced_flexibility_energy_results.mean()
forced_flexibility_power_mean = forced_flexibility_power_results.mean()
forced_flexibility_percent_mean = forced_flexibility_percent_results.mean()

# Print forced flexibility results for debugging.
print("forced_flexibility_percent_results = \n{}".format(forced_flexibility_percent_results))
print("forced_flexibility_percent_mean = \n{}".format(forced_flexibility_percent_mean))

# Save results to CSV.
forced_flexibility_energy_results.to_csv(os.path.join(results_path, 'forced_flexibility_energy_results.csv'))
forced_flexibility_power_results.to_csv(os.path.join(results_path, 'forced_flexibility_power_results.csv'))
forced_flexibility_percent_results.to_csv(os.path.join(results_path, 'forced_flexibility_percent_results.csv'))
forced_flexibility_energy_mean.to_csv(os.path.join(results_path, 'forced_flexibility_energy_mean.csv'))
forced_flexibility_power_mean.to_csv(os.path.join(results_path, 'forced_flexibility_power_mean.csv'))
forced_flexibility_percent_mean.to_csv(os.path.join(results_path, 'forced_flexibility_percent_mean.csv'))

# Print results path for debugging.
print("Results are stored in: " + results_path)
