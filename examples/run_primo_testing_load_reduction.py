"""Run script for evaluating demand side flexibility in terms of load reduction."""

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
    sit_scenario_names = [
        'singapore_pdd_sit_w1',
        'singapore_pdd_sit_w3',
        'singapore_pdd_sit_w5',
        'singapore_pdd_sit_w6',
        'singapore_pdd_sit_w7',
        'singapore_pdd_sit_e1',
        'singapore_pdd_sit_e2',
        'singapore_pdd_sit_e3',
        'singapore_pdd_sit_e4',
        'singapore_pdd_sit_e5',
        'singapore_pdd_sit_e6'
    ]
    jtc_scenario_names = [
        'singapore_pdd_jtc_t1',
        'singapore_pdd_jtc_t2',
        'singapore_pdd_jtc_t3',
        'singapore_pdd_jtc_t4',
        'singapore_pdd_jtc_t5',
        'singapore_pdd_jtc_t6',
        'singapore_pdd_jtc_t7',
        'singapore_pdd_jtc_t8',
        'singapore_pdd_jtc_t9',
        'singapore_pdd_jtc_t10',
        'singapore_pdd_jtc_podium'
    ]
    scenario_names = [*sit_scenario_names, *jtc_scenario_names]
    time_intervals = (
        pd.Index([
            pd.to_timedelta('{}h'.format(time_duration))
            for time_duration in np.arange(0.5, 3.5, 0.5)
        ])
    )
    results_path = cobmo.utils.get_results_path(f'run_primo_testing_load_reduction')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Instantiate results collection variables.
    load_reduction_energy_mean_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_power_mean_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_percent_mean_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_energy_min_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_power_min_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_percent_min_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_energy_max_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_power_max_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)
    load_reduction_percent_max_collection = pd.DataFrame(None, index=scenario_names, columns=time_intervals)

    for scenario_name in scenario_names:

        # Obtain building model.
        building_model = cobmo.building_model.BuildingModel(scenario_name)

        # Save building model matrices to CSV for debugging.
        building_model.state_matrix.to_csv(os.path.join(results_path, f'{scenario_name}_building_state_matrix.csv'))
        building_model.control_matrix.to_csv(os.path.join(results_path, f'{scenario_name}_building_control_matrix.csv'))
        building_model.disturbance_matrix.to_csv(os.path.join(results_path, f'{scenario_name}_building_disturbance_matrix.csv'))
        building_model.state_output_matrix.to_csv(os.path.join(results_path, f'{scenario_name}_building_state_output_matrix.csv'))
        building_model.control_output_matrix.to_csv(os.path.join(results_path, f'{scenario_name}_building_control_output_matrix.csv'))
        building_model.disturbance_output_matrix.to_csv(os.path.join(results_path, f'{scenario_name}_building_disturbance_output_matrix.csv'))
        building_model.disturbance_timeseries.to_csv(os.path.join(results_path, f'{scenario_name}_building_disturbance_timeseries.csv'))

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
        control_vector_baseline.to_csv(os.path.join(results_path, f'{scenario_name}_baseline_control_vector.csv'))
        state_vector_baseline.to_csv(os.path.join(results_path, f'{scenario_name}_baseline_state_vector.csv'))
        output_vector_baseline.to_csv(os.path.join(results_path, f'{scenario_name}_baseline_output_vector.csv'))

        # Instantiate results collection variables.
        load_reduction_energy_collection = pd.DataFrame(None, index=building_model.timesteps, columns=time_intervals)
        load_reduction_power_collection = pd.DataFrame(None, index=building_model.timesteps, columns=time_intervals)
        load_reduction_percent_collection = pd.DataFrame(None, index=building_model.timesteps, columns=time_intervals)

        # Obtain timesteps during which the HVAC system is expected to be active (operational hours).
        timesteps = (
            building_model.timesteps[(
                building_model.output_constraint_timeseries_maximum
                != building_model.output_constraint_timeseries_maximum.max()
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
        load_reduction_energy_collection.to_csv(os.path.join(results_path, f'{scenario_name}_load_reduction_energy_results.csv'))
        load_reduction_power_collection.to_csv(os.path.join(results_path, f'{scenario_name}_load_reduction_power_results.csv'))
        load_reduction_percent_collection.to_csv(os.path.join(results_path, f'{scenario_name}_load_reduction_percent_results.csv'))

        # Store results to collection variables.
        load_reduction_energy_mean_collection.loc[scenario_name, :] = load_reduction_energy_collection.loc['mean', :]
        load_reduction_power_mean_collection.loc[scenario_name, :] = load_reduction_power_collection.loc['mean', :]
        load_reduction_percent_mean_collection.loc[scenario_name, :] = load_reduction_percent_collection.loc['mean', :]
        load_reduction_energy_min_collection.loc[scenario_name, :] = load_reduction_energy_collection.loc['min', :]
        load_reduction_power_min_collection.loc[scenario_name, :] = load_reduction_power_collection.loc['min', :]
        load_reduction_percent_min_collection.loc[scenario_name, :] = load_reduction_percent_collection.loc['min', :]
        load_reduction_energy_max_collection.loc[scenario_name, :] = load_reduction_energy_collection.loc['max', :]
        load_reduction_power_max_collection.loc[scenario_name, :] = load_reduction_power_collection.loc['max', :]
        load_reduction_percent_max_collection.loc[scenario_name, :] = load_reduction_percent_collection.loc['max', :]

    # Save results to CSV.
    load_reduction_energy_mean_collection.to_csv(os.path.join(results_path, 'load_reduction_energy_mean.csv'))
    load_reduction_power_mean_collection.to_csv(os.path.join(results_path, 'load_reduction_power_mean.csv'))
    load_reduction_percent_mean_collection.to_csv(os.path.join(results_path, 'load_reduction_percent_mean.csv'))
    load_reduction_energy_min_collection.to_csv(os.path.join(results_path, 'load_reduction_energy_min.csv'))
    load_reduction_power_min_collection.to_csv(os.path.join(results_path, 'load_reduction_power_min.csv'))
    load_reduction_percent_min_collection.to_csv(os.path.join(results_path, 'load_reduction_percent_min.csv'))
    load_reduction_energy_max_collection.to_csv(os.path.join(results_path, 'load_reduction_energy_max.csv'))
    load_reduction_power_max_collection.to_csv(os.path.join(results_path, 'load_reduction_power_max.csv'))
    load_reduction_percent_max_collection.to_csv(os.path.join(results_path, 'load_reduction_percent_max.csv'))

    # Launch & print results path.
    cobmo.utils.launch(results_path)
    print(f"Results are stored in {results_path}")


if __name__ == '__main__':
    main()
