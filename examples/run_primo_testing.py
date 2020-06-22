"""Run script for testing of the building scenarios related to project PRIMO.

- This script relies on the PRIMO building scenarios which are not included in this repository. If you have the
  scenario definition files, add the path to the definition in `config.yml` at `additional_data: []`.
"""

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
    scenario_names = [
        'primo_1_SIT_P1_W1',
        'primo_2_SIT_P1_W3',
        'primo_3_SIT_P1_W5',
        'primo_4_SIT_P1_W6',
        'primo_5_SIT_P1_W7',
        'primo_7_SIT_P2_E1',
        'primo_8_SIT_P2_E2',
        'primo_9_SIT_P2_E3',
        'primo_10_SIT_P2_E4',
        'primo_11_SIT_P2_E5',
        'primo_12_SIT_P2_E6',
        'primo_13_JTC_CC1_Tower_1',
        'primo_14_JTC_CC1_Tower_2',
        'primo_15_JTC_CC1_Tower_3',
        'primo_16_JTC_CC1_Tower_4',
        'primo_17_JTC_CC1_Podium',
        'primo_18_JTC_CC2_Tower_5',
        'primo_19_JTC_CC2_Tower_6',
        'primo_20_JTC_CC2_Tower_7',
        'primo_21_JTC_CC3_Tower_8',
        'primo_22_JTC_CC3_Tower_9',
        'primo_23_JTC_CC3_Tower_10'
    ]
    results_path_main = cobmo.utils.get_results_path(f'run_primo_testing')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Run each scenario.
    results = []
    for scenario_name in scenario_names:

        # Print progress.
        print(f"Running scenario '{scenario_name}'...")

        # Obtain scenario results path.
        results_path_scenario = cobmo.utils.get_results_path(f'run_evaluation_energy_use_{scenario_name}')

        # Obtain building model.
        building = cobmo.building_model.BuildingModel(scenario_name)

        # Obtain and solve optimization problem.
        optimization_problem = cobmo.optimization_problem.OptimizationProblem(
            building
        )
        (
            control_vector_optimization,
            state_vector_optimization,
            output_vector_optimization,
            operation_cost,
            investment_cost,  # Zero when running (default) operation problem.
            storage_size  # Zero when running (default) operation problem.
        ) = optimization_problem.solve()

        # Print optimization results.
        print(f"operation_cost = {operation_cost}")
        print(f"control_vector_optimization = \n{control_vector_optimization}")
        print(f"state_vector_optimization = \n{state_vector_optimization}")
        print(f"output_vector_optimization = \n{output_vector_optimization}")

        # Store optimization results as CSV.
        control_vector_optimization.to_csv(os.path.join(results_path_scenario, 'control_vector_optimization.csv'))
        state_vector_optimization.to_csv(os.path.join(results_path_scenario, 'state_vector_optimization.csv'))
        output_vector_optimization.to_csv(os.path.join(results_path_scenario, 'output_vector_optimization.csv'))

        # Obtain energy use intensity (EUI).
        energy_use_timeseries = (
            output_vector_optimization['grid_electric_power']  # in W
            / building.building_data.zones.loc[:, 'zone_area'].sum()  # in W/m²
            * (building.timestep_interval.seconds / 3600)  # in Wh/m²
            / 1000  # in kWh/m²
        )
        energy_use_timestep = energy_use_timeseries.mean()
        energy_use_day = (
            energy_use_timeseries.sum()
            / len(building.timesteps)
            * (pd.to_timedelta('1d') / building.timestep_interval)
        )
        energy_use_week = (
            energy_use_timeseries.sum()
            / len(building.timesteps)
            * (pd.to_timedelta('1w') / building.timestep_interval)
        )
        energy_use_month = (
            energy_use_timeseries.sum()
            / len(building.timesteps)
            * (pd.to_timedelta('1y') / 12 / building.timestep_interval)
        )
        energy_use_year = (
            energy_use_timeseries.sum()
            / len(building.timesteps)
            * (pd.to_timedelta('1y') / building.timestep_interval)
        )
        energy_use_timeseries = round(energy_use_timeseries, 2)
        energy_use_timestep = round(energy_use_timestep, 2)
        energy_use_day = round(energy_use_day, 2)
        energy_use_week = round(energy_use_week, 2)
        energy_use_month = round(energy_use_month, 2)
        energy_use_year = round(energy_use_year, 2)

        # Print energy use intensity (EUI).
        print(f"energy_use_timeseries = \n{energy_use_timeseries}")
        print(f"energy_use_timestep = {energy_use_timestep} kWh/m²")
        print(f"energy_use_day = {energy_use_day} kWh/m²")
        print(f"energy_use_week = {energy_use_week} kWh/m²")
        print(f"energy_use_month = {energy_use_month} kWh/m²")
        print(f"energy_use_year = {energy_use_year} kWh/m²")

        # Store energy use intensity (EUI) as CSV.
        energy_use_summary = (
            pd.Series(
                [
                    energy_use_timestep,
                    energy_use_day,
                    energy_use_week,
                    energy_use_month,
                    energy_use_year
                ],
                index=[
                    'energy_use_timestep',
                    'energy_use_day',
                    'energy_use_week',
                    'energy_use_month',
                    'energy_use_year'
                ]
            )
        )
        energy_use_summary.to_csv(os.path.join(results_path_scenario, 'energy_use_summary.csv'))
        energy_use_timeseries.to_csv(os.path.join(results_path_scenario, 'energy_use_timeseries.csv'))

        # Append results.
        results.append(energy_use_summary.rename(scenario_name))

    # Process / print / store results.
    results = pd.concat(results, axis='columns')
    print(f"results {results}")
    results.to_csv(os.path.join(results_path_main, 'results.csv'))

    # Print results path.
    print(f"Results are stored in: {results_path_main}")


if __name__ == '__main__':
    main()
