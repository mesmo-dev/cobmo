"""Run script for evaluating the energy use intensity (EUI)."""

import numpy as np
import os
import pandas as pd

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.database_interface


def main():

    # Settings.
    scenario_name = 'scenario_default'
    results_path = os.path.join(cobmo.config.results_path, f'run_evaluation_energy_use_{cobmo.config.timestamp}')

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.database_interface.recreate_database()

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
    print(f"control_vector_optimization = \n{control_vector_optimization.to_string()}")
    print(f"state_vector_optimization = \n{state_vector_optimization.to_string()}")
    print(f"output_vector_optimization = \n{output_vector_optimization.to_string()}")

    # Store optimization results as CSV.
    control_vector_optimization.to_csv(os.path.join(results_path, 'control_vector_optimization.csv'))
    state_vector_optimization.to_csv(os.path.join(results_path, 'state_vector_optimization.csv'))
    output_vector_optimization.to_csv(os.path.join(results_path, 'output_vector_optimization.csv'))

    # Obtain energy use intensity (EUI).
    print()
    energy_use_timeseries = (
        output_vector_optimization['grid_electric_power']  # in W
        / building.building_data.zones.loc[:, 'zone_area'].sum()  # in W/m²
        * (building.timestep_delta.seconds / 3600)  # in Wh/m²
        / 1000  # in kWh/m²
    )
    energy_use_timestep = energy_use_timeseries.mean()
    energy_use_day = (
        energy_use_timeseries.sum()
        / len(building.timesteps)
        * (pd.to_timedelta('1d') / building.timestep_delta)
    )
    energy_use_week = (
        energy_use_timeseries.sum()
        / len(building.timesteps)
        * (pd.to_timedelta('1w') / building.timestep_delta)
    )
    energy_use_month = (
        energy_use_timeseries.sum()
        / len(building.timesteps)
        * (pd.to_timedelta('1y') / 12 / building.timestep_delta)
    )
    energy_use_year = (
        energy_use_timeseries.sum()
        / len(building.timesteps)
        * (pd.to_timedelta('1y') / building.timestep_delta)
    )
    energy_use_timeseries = round(energy_use_timeseries, 2)
    energy_use_timestep = round(energy_use_timestep, 2)
    energy_use_day = round(energy_use_day, 2)
    energy_use_week = round(energy_use_week, 2)
    energy_use_month = round(energy_use_month, 2)
    energy_use_year = round(energy_use_year, 2)

    # Print energy use intensity (EUI).
    print(f"energy_use_timeseries = \n{energy_use_timeseries.to_string()}")
    print(f"energy_use_timestep = \n{energy_use_timestep} kWh/m²")
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
    energy_use_summary.to_csv(os.path.join(results_path, 'energy_use_summary.csv'))
    energy_use_timeseries.to_csv(os.path.join(results_path, 'energy_use_timeseries.csv'))

    # Print results path.
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
