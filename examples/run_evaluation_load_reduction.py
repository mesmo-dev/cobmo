"""Run script for evaluating demand side flexibility in terms of load reduction."""

import hvplot
import hvplot.pandas
import numpy as np
import os
import pandas as pd

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.data_interface
import cobmo.utils


def main():

    # Settings.
    scenario_name = 'paper_pesgm_2020'
    results_path = cobmo.utils.get_results_path(f'run_evaluation_load_reduction_{scenario_name}')

    # Instantiate results subfolders.
    os.mkdir(os.path.join(results_path, 'plots'))
    os.mkdir(os.path.join(results_path, 'details'))

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Obtain building model.
    building = cobmo.building_model.BuildingModel(scenario_name)

    # Save building model matrices to CSV for debugging.
    building.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
    building.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
    building.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
    building.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
    building.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
    building.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
    building.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

    # Run controller for baseline case.
    controller_baseline = cobmo.optimization_problem.OptimizationProblem(
        building
    )
    (
        control_vector_baseline,
        state_vector_baseline,
        output_vector_baseline,
        operation_cost_baseline,
        investment_cost_baseline,  # Zero when running (default) operation problem.
        storage_size_baseline  # Zero when running (default) operation problem.
    ) = controller_baseline.solve()

    # Print operation cost for debugging.
    print("operation_cost_baseline = {}".format(operation_cost_baseline))

    # Save controller timeseries to CSV for debugging.
    control_vector_baseline.to_csv(os.path.join(results_path, 'control_vector_baseline.csv'))
    state_vector_baseline.to_csv(os.path.join(results_path, 'state_vector_baseline.csv'))
    output_vector_baseline.to_csv(os.path.join(results_path, 'output_vector_baseline.csv'))

    # Instantiate load reduction iteration variables.
    set_time_duration = (
        pd.Index([
            pd.to_timedelta('{}h'.format(time_duration))
            for time_duration in np.arange(0.5, 6.5, 0.5)
        ])
    )
    timesteps = building.timesteps
    load_reduction_energy_results = pd.DataFrame(
        None,
        timesteps,
        set_time_duration
    )
    load_reduction_power_results = pd.DataFrame(
        None,
        timesteps,
        set_time_duration
    )
    load_reduction_percent_results = pd.DataFrame(
        None,
        timesteps,
        set_time_duration
    )

    # Iterate load reduction calculation.
    for time_duration in set_time_duration:
        for timestep in timesteps:
            if (timestep + time_duration) > building.timesteps[-1]:
                break  # Interrupt loop if end time goes beyond building model time horizon.
            elif (
                output_vector_baseline.loc[timestep, output_vector_baseline.columns.str.contains('electric_power')]
                == 0.0
            ).all():
                continue  # Skip loop if there is no baseline demand in the start timestep (no reduction possible).
            else:
                # Print status info.
                print("Calculate load reduction for: time_duration = {} / timestep = {}".format(time_duration, timestep))

                # Run controller for load reduction case.
                controller_load_reduction = cobmo.optimization_problem.OptimizationProblem(
                    building,
                    problem_type='load_reduction',
                    output_vector_reference=output_vector_baseline,
                    load_reduction_start_time=timestep,
                    load_reduction_end_time=timestep + time_duration
                )
                (
                    control_vector_load_reduction,
                    state_vector_load_reduction,
                    output_vector_load_reduction,
                    operation_cost_load_reduction,
                    investment_cost_load_reduction,  # Represents load reduction.
                    storage_size_load_reduction
                ) = controller_load_reduction.solve()

                # Save controller timeseries to CSV for debugging.
                control_vector_load_reduction.to_csv(os.path.join(
                    results_path, 'details', '{} - {} control_vector.csv'.format(time_duration, timestep).replace(':', '-')
                ))
                control_vector_load_reduction.to_csv(os.path.join(
                    results_path, 'details', '{} - {} state_vector.csv'.format(time_duration, timestep).replace(':', '-')
                ))
                control_vector_load_reduction.to_csv(os.path.join(
                    results_path, 'details', '{} - {} output_vectord.csv'.format(time_duration, timestep).replace(':', '-')
                ))

                # Plot demand comparison for debugging.
                electric_power_comparison = pd.concat(
                    [
                        output_vector_baseline.loc[:, output_vector_baseline.columns.str.contains('electric_power')].sum(axis=1),
                        output_vector_load_reduction.loc[:, output_vector_load_reduction.columns.str.contains('electric_power')].sum(axis=1),
                    ],
                    keys=[
                        'baseline',
                        'load_reduction',
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
                        results_path, 'plots', '{} - {}.html'.format(time_duration, timestep).replace(':', '-')
                    )
                )

                # Calculate results.
                # TODO: Move timestep_interval into building model.
                timestep_interval = building.timesteps[1] - building.timesteps[0]
                baseline_energy = (
                    output_vector_baseline.loc[
                        timestep:(timestep + time_duration),
                        output_vector_baseline.columns.str.contains('electric_power')
                    ].sum().sum()
                    * timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                )
                load_reduction_percent = - investment_cost_load_reduction  # In percent.
                load_reduction_energy = (
                    (load_reduction_percent / 100.0)
                    * baseline_energy
                )  # in kWh.
                load_reduction_power = (
                    load_reduction_energy
                    / (time_duration.total_seconds() / 3600.0)  # kWh in kW.
                )

                # Print results.
                print("load_reduction_energy = {}".format(load_reduction_energy))
                print("load_reduction_power = {}".format(load_reduction_power))
                print("load_reduction_percent = {}".format(load_reduction_percent))

                # Store results.
                load_reduction_energy_results.at[timestep, time_duration] = load_reduction_energy
                load_reduction_power_results.at[timestep, time_duration] = load_reduction_power
                load_reduction_percent_results.at[timestep, time_duration] = load_reduction_percent

    # Aggregate load reduction results.
    load_reduction_energy_mean = load_reduction_energy_results.mean()
    load_reduction_power_mean = load_reduction_power_results.mean()
    load_reduction_percent_mean = load_reduction_percent_results.mean()

    # Print load reduction results for debugging.
    print("load_reduction_percent_results = \n{}".format(load_reduction_percent_results))
    print("load_reduction_percent_mean = \n{}".format(load_reduction_percent_mean))

    # Save results to CSV.
    load_reduction_energy_results.to_csv(os.path.join(results_path, 'load_reduction_energy_results.csv'))
    load_reduction_power_results.to_csv(os.path.join(results_path, 'load_reduction_power_results.csv'))
    load_reduction_percent_results.to_csv(os.path.join(results_path, 'load_reduction_percent_results.csv'))
    load_reduction_energy_mean.to_csv(os.path.join(results_path, 'load_reduction_energy_mean.csv'))
    load_reduction_power_mean.to_csv(os.path.join(results_path, 'load_reduction_power_mean.csv'))
    load_reduction_percent_mean.to_csv(os.path.join(results_path, 'load_reduction_percent_mean.csv'))

    # Print results path for debugging.
    print("Results are stored in: " + results_path)


if __name__ == '__main__':
    main()
