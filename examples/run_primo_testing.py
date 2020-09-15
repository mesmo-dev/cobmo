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
        'singapore_pdd_jtc_tower_1',
        'singapore_pdd_jtc_podium_1',
        'singapore_pdd_jtc_tower_2',
        'singapore_pdd_jtc_podium_2',
        'singapore_pdd_jtc_tower_3',
        'singapore_pdd_jtc_podium_3',
        'singapore_pdd_jtc_tower_4',
        'singapore_pdd_jtc_tower_5',
        'singapore_pdd_jtc_tower_6',
        'singapore_pdd_jtc_tower_7',
        'singapore_pdd_jtc_tower_8',
        'singapore_pdd_jtc_tower_9',
        'singapore_pdd_jtc_tower_10'
    ]
    scenario_names = [*sit_scenario_names, *jtc_scenario_names]
    building_gross_floor_area = {
        'singapore_pdd_sit_w1': 23404.6,
        'singapore_pdd_sit_w3': 44284.6,
        'singapore_pdd_sit_w5': 30164.6,
        'singapore_pdd_sit_w6': 9596.6,
        'singapore_pdd_sit_w7': 13843.8,
        'singapore_pdd_sit_e1': 22965.4,
        'singapore_pdd_sit_e2': 46176.8,
        'singapore_pdd_sit_e3': 8853.6,
        'singapore_pdd_sit_e4': 1861,
        'singapore_pdd_sit_e5': 10493.7,
        'singapore_pdd_sit_e6': 39438.8,
        'singapore_pdd_jtc_tower_1': 29856,
        'singapore_pdd_jtc_podium_1': 5114,
        'singapore_pdd_jtc_tower_2': 20604,
        'singapore_pdd_jtc_podium_2': 5114,
        'singapore_pdd_jtc_tower_3': 25312,
        'singapore_pdd_jtc_podium_3': 5114,
        'singapore_pdd_jtc_tower_4': 5140,
        'singapore_pdd_jtc_tower_5': 21574,
        'singapore_pdd_jtc_tower_6': 20483,
        'singapore_pdd_jtc_tower_7': 49274,
        'singapore_pdd_jtc_tower_8': 36185,
        'singapore_pdd_jtc_tower_9': 12365,
        'singapore_pdd_jtc_tower_10': 67413
    }
    results_path_main = cobmo.utils.get_results_path(f'run_primo_testing')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Instantiate results collection variables.
    state_vector_collection = dict().fromkeys(scenario_names)
    control_vector_collection = dict().fromkeys(scenario_names)
    output_vector_collection = dict().fromkeys(scenario_names)
    energy_use_summary_collection = []

    # Run each scenario.
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
            / (
                building_gross_floor_area[scenario_name]
                if scenario_name in building_gross_floor_area.keys()
                else building.building_data.zones.loc[:, 'zone_area'].sum()
            )  # in W/m²
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
        state_vector_collection[scenario_name] = state_vector_optimization
        control_vector_collection[scenario_name] = control_vector_optimization
        output_vector_collection[scenario_name] = output_vector_optimization
        energy_use_summary_collection.append(energy_use_summary.rename(scenario_name))

    # Merge results collections.
    state_vector_collection = pd.concat(state_vector_collection, axis='columns')
    control_vector_collection = pd.concat(control_vector_collection, axis='columns')
    output_vector_collection = pd.concat(output_vector_collection, axis='columns')
    energy_use_summary_collection = pd.concat(energy_use_summary_collection, axis='columns')
    # Set multi-index level names for more convenient processing.
    state_vector_collection.columns.set_names('scenario_name', level=0, inplace=True)
    control_vector_collection.columns.set_names('scenario_name', level=0, inplace=True)
    output_vector_collection.columns.set_names('scenario_name', level=0, inplace=True)

    # Obtain cooling / electric power timeseries.
    cooling_power_timeseries = output_vector_collection.loc[:, (slice(None), 'plant_thermal_power_cooling')]
    cooling_power_timeseries = cooling_power_timeseries.groupby('scenario_name', axis='columns').sum()
    cooling_power_timeseries.loc[:, 'pdd_total'] = cooling_power_timeseries.sum(axis='columns')
    cooling_power_timeseries.loc[:, 'sit_total'] = (
        cooling_power_timeseries.loc[:, cooling_power_timeseries.columns.isin(sit_scenario_names)].sum(axis='columns')
    )
    cooling_power_timeseries.loc[:, 'jtc_total'] = (
        cooling_power_timeseries.loc[:, cooling_power_timeseries.columns.isin(jtc_scenario_names)].sum(axis='columns')
    )
    electric_power_timeseries = output_vector_collection.loc[:, (slice(None), 'grid_electric_power')]
    electric_power_timeseries = electric_power_timeseries.groupby('scenario_name', axis='columns').sum()
    electric_power_timeseries.loc[:, 'pdd_total'] = electric_power_timeseries.sum(axis='columns')
    electric_power_timeseries.loc[:, 'sit_total'] = (
        electric_power_timeseries.loc[:, electric_power_timeseries.columns.isin(sit_scenario_names)].sum(axis='columns')
    )
    electric_power_timeseries.loc[:, 'jtc_total'] = (
        electric_power_timeseries.loc[:, electric_power_timeseries.columns.isin(jtc_scenario_names)].sum(axis='columns')
    )

    # Print results.
    print(f"cooling_power_timeseries = \n{cooling_power_timeseries}")
    print(f"electric_power_timeseries = \n{electric_power_timeseries}")
    print(f"energy_use_summary = \n{energy_use_summary_collection}")

    # Store results.
    state_vector_collection.to_csv(os.path.join(results_path_main, 'state_vector_collection.csv'))
    control_vector_collection.to_csv(os.path.join(results_path_main, 'control_vector_collection.csv'))
    output_vector_collection.to_csv(os.path.join(results_path_main, 'output_vector_collection.csv'))
    energy_use_summary_collection.to_csv(os.path.join(results_path_main, 'results.csv'))
    cooling_power_timeseries.to_csv(os.path.join(results_path_main, 'cooling_power_timeseries.csv'))
    electric_power_timeseries.to_csv(os.path.join(results_path_main, 'electric_power_timeseries.csv'))

    # Print results path.
    print(f"Results are stored in: {results_path_main}")


if __name__ == '__main__':
    main()
