"""Run script for evaluating demand side flexibility in terms of load reduction."""

import matplotlib.dates
import matplotlib.pyplot as plt
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
    time_intervals = (
        pd.Index([
            pd.to_timedelta('{}h'.format(time_duration))
            for time_duration in np.arange(0.5, 3.5, 0.5)
        ])
    )
    results_path = cobmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Obtain building model.
    building_model = cobmo.building_model.BuildingModel(scenario_name)

    # Save building model matrices to CSV for debugging.
    building_model.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
    building_model.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
    building_model.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
    building_model.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
    building_model.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
    building_model.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
    building_model.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

    # Setup / solve optimization problem for baseline case.
    optimization_problem_baseline = cobmo.optimization_problem.OptimizationProblem(building_model)
    (
        control_vector_baseline,
        state_vector_baseline,
        output_vector_baseline,
        operation_cost_baseline,
        investment_cost_baseline,  # Zero when running (default) operation problem.
        storage_size_baseline  # Zero when running (default) operation problem.
    ) = optimization_problem_baseline.solve()

    # Save controller timeseries to CSV for debugging.
    control_vector_baseline.to_csv(os.path.join(results_path, 'baseline_control_vector.csv'))
    state_vector_baseline.to_csv(os.path.join(results_path, 'baseline_state_vector.csv'))
    output_vector_baseline.to_csv(os.path.join(results_path, 'baseline_output_vector.csv'))

    # Instantiate results collection variables.
    load_reduction_energy_collection = pd.DataFrame(None, index=building_model.timesteps, columns=time_intervals)
    load_reduction_power_collection = pd.DataFrame(None, index=building_model.timesteps, columns=time_intervals)
    load_reduction_percent_collection = pd.DataFrame(None, index=building_model.timesteps, columns=time_intervals)

    # Obtain timesteps during which the HVAC system is expected to be active (operational hours).
    timesteps = (
        building_model.timesteps[(
            building_model.output_maximum_timeseries
            != building_model.output_maximum_timeseries.max()
        ).any(axis='columns')]
    )
    timesteps = building_model.timesteps if len(timesteps) == 0 else timesteps

    # Iterate load reduction calculation.
    optimization_problem_load_reduction = None
    for time_interval in time_intervals:
        for timestep in timesteps:
            if (timestep + time_interval) > timesteps[-1]:
                break  # Interrupt loop if end time goes beyond building model time horizon.
            elif (
                output_vector_baseline.loc[timestep, output_vector_baseline.columns.str.contains('electric_power')]
                == 0.0
            ).all():
                continue  # Skip loop if there is no baseline demand in the start timestep (no reduction possible).
            else:
                # Print status info.
                print(f"Calculate load reduction at time step {timestep} for {time_interval}")

                # Define optimization problem.
                # - If optimization problem already exists, only redefine load reduction constraints.
                if optimization_problem_load_reduction is None:
                    optimization_problem_load_reduction = cobmo.optimization_problem.OptimizationProblem(
                        building_model,
                        problem_type='load_reduction',
                        output_vector_reference=output_vector_baseline,
                        load_reduction_start_time=timestep,
                        load_reduction_end_time=timestep + time_interval
                    )
                else:
                    optimization_problem_load_reduction.define_load_reduction_constraints(
                        output_vector_reference=output_vector_baseline,
                        load_reduction_start_time=timestep,
                        load_reduction_end_time=timestep + time_interval
                    )

                # Solve optimization problem.
                (
                    control_vector_load_reduction,
                    state_vector_load_reduction,
                    output_vector_load_reduction,
                    operation_cost_load_reduction,
                    investment_cost_load_reduction,
                    storage_size_load_reduction
                ) = optimization_problem_load_reduction.solve()

                # Calculate load reduction.
                baseline_energy = (
                    output_vector_baseline.loc[timestep:(timestep + time_interval), 'grid_electric_power'].sum()
                    * building_model.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                )
                load_reduction_percent = (
                    -1.0
                    * optimization_problem_load_reduction.problem.variable_load_reduction[0].value  # In percent.
                )
                load_reduction_energy = (
                    (load_reduction_percent / 100.0)
                    * baseline_energy
                )  # in kWh.
                load_reduction_power = (
                    load_reduction_energy
                    / (time_interval.total_seconds() / 3600.0)  # kWh in kW.
                )

                # Print results.
                print(f"load_reduction_energy = {load_reduction_energy}")
                print(f"load_reduction_power = {load_reduction_power}")
                print(f"load_reduction_percent = {load_reduction_percent}")

                # Store results to collection variables.
                load_reduction_energy_collection.at[timestep, time_interval] = load_reduction_energy
                load_reduction_power_collection.at[timestep, time_interval] = load_reduction_power
                load_reduction_percent_collection.at[timestep, time_interval] = load_reduction_percent

                # Save results to CSV.
                control_vector_load_reduction.to_csv(os.path.join(
                    results_path,
                    cobmo.utils.get_alphanumeric_string(f'load_reduction_{time_interval}_{timestep}_control_vector')
                    + '.csv'
                ))
                state_vector_load_reduction.to_csv(os.path.join(
                    results_path,
                    cobmo.utils.get_alphanumeric_string(f'load_reduction_{time_interval}_{timestep}_state_vector')
                    + '.csv'
                ))
                output_vector_load_reduction.to_csv(os.path.join(
                    results_path,
                    cobmo.utils.get_alphanumeric_string(f'load_reduction_{time_interval}_{timestep}_output_vector')
                    + '.csv'
                ))

                # Plot load reduction vs baseline.
                plt.figure()
                plt.title(f"Load reduction at time step {timestep} for {time_interval}")
                plt.plot(
                    output_vector_baseline.loc[:, 'grid_electric_power'],
                    label='Baseline',
                    drawstyle='steps-post'
                )
                plt.plot(
                    output_vector_load_reduction.loc[:, 'grid_electric_power'],
                    label='Load reduction',
                    drawstyle='steps-post'
                )
                plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
                plt.legend()
                # plt.show()
                plt.savefig(os.path.join(
                    results_path,
                    cobmo.utils.get_alphanumeric_string(f'plot_load_reduction_{time_interval}_{timestep}') + '.png'
                ))
                plt.close()

    # Add mean / min / max values.
    load_reduction_energy_collection.loc['mean', :] = load_reduction_energy_collection.mean()
    load_reduction_energy_collection.loc['max', :] = load_reduction_energy_collection.max()
    load_reduction_energy_collection.loc['min', :] = load_reduction_energy_collection.min()
    load_reduction_power_collection.loc['mean', :] = load_reduction_power_collection.mean()
    load_reduction_power_collection.loc['max', :] = load_reduction_power_collection.max()
    load_reduction_power_collection.loc['min', :] = load_reduction_power_collection.min()
    load_reduction_percent_collection.loc['mean', :] = load_reduction_percent_collection.mean()
    load_reduction_percent_collection.loc['max', :] = load_reduction_percent_collection.max()
    load_reduction_percent_collection.loc['min', :] = load_reduction_percent_collection.min()

    # Print results.
    print(f"load_reduction_energy_collection = \n{load_reduction_energy_collection}")
    print(f"load_reduction_power_collection = \n{load_reduction_power_collection}")
    print(f"load_reduction_percent_results = \n{load_reduction_percent_collection}")

    # Save results to CSV.
    load_reduction_energy_collection.to_csv(os.path.join(results_path, 'load_reduction_energy_results.csv'))
    load_reduction_power_collection.to_csv(os.path.join(results_path, 'load_reduction_power_results.csv'))
    load_reduction_percent_collection.to_csv(os.path.join(results_path, 'load_reduction_percent_results.csv'))

    # Launch & print results path.
    cobmo.utils.launch(results_path)
    print(f"Results are stored in {results_path}")


if __name__ == '__main__':
    main()
