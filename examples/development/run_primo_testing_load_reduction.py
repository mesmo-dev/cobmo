"""Run script for evaluating demand side flexibility in terms of load reduction."""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    results_path = cobmo.utils.get_results_path(__file__)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Instantiate results collection variables.
    load_reduction_energy_collections = list()
    load_reduction_power_collections = list()
    load_reduction_percent_collections = list()

    for scenario_name in scenario_names:

        # Print progress.
        print(f"Starting scenario: {scenario_name}")

        # Obtain building model.
        building_model = cobmo.building_model.BuildingModel(scenario_name)

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

        # Instantiate results collection variables.
        load_reduction_energy_collection = (
            pd.DataFrame(None, index=building_model.timesteps, columns=pd.Index(time_intervals, name='time_interval'))
        )
        load_reduction_power_collection = (
            pd.DataFrame(None, index=building_model.timesteps, columns=pd.Index(time_intervals, name='time_interval'))
        )
        load_reduction_percent_collection = (
            pd.DataFrame(None, index=building_model.timesteps, columns=pd.Index(time_intervals, name='time_interval'))
        )

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

                    # Print progress.
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
                    optimization_problem_load_reduction.solve()

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

        # Print results.
        print(f"load_reduction_energy_collection = \n{load_reduction_energy_collection}")
        print(f"load_reduction_power_collection = \n{load_reduction_power_collection}")
        print(f"load_reduction_percent_collection = \n{load_reduction_percent_collection}")

        # Save results to CSV.
        load_reduction_energy_collection.to_csv(os.path.join(results_path, f'{scenario_name}_load_reduction_energy.csv'))
        load_reduction_power_collection.to_csv(os.path.join(results_path, f'{scenario_name}_load_reduction_power.csv'))
        load_reduction_percent_collection.to_csv(os.path.join(results_path, f'{scenario_name}_load_reduction_percent.csv'))

        # Store results to collection variables.
        load_reduction_energy_collections.append(load_reduction_energy_collection)
        load_reduction_power_collections.append(load_reduction_power_collection)
        load_reduction_percent_collections.append(load_reduction_percent_collection)

    # Merge collection variables.
    load_reduction_energy_collections = (
        pd.concat(load_reduction_energy_collections, axis='columns', keys=scenario_names, names=['scenario_name'])
    )
    load_reduction_power_collections = (
        pd.concat(load_reduction_power_collections, axis='columns', keys=scenario_names, names=['scenario_name'])
    )
    load_reduction_percent_collections = (
        pd.concat(load_reduction_percent_collections, axis='columns', keys=scenario_names, names=['scenario_name'])
    )

    # Save results to CSV.
    load_reduction_energy_collections.to_csv(os.path.join(results_path, 'load_reduction_energy.csv'))
    load_reduction_power_collections.to_csv(os.path.join(results_path, 'load_reduction_power.csv'))
    load_reduction_percent_collections.to_csv(os.path.join(results_path, 'load_reduction_percent.csv'))

    # Plots.

    # Load reduction aggregate.
    figure = go.Figure()
    values = load_reduction_percent_collections.abs().max().groupby('time_interval').max().copy()
    values.index = (values.index.seconds / 3600).astype(str) + 'h'
    figure.add_trace(go.Bar(
        x=values.index,
        y=values.values,
        name='Maximum reduction'
    ))
    values = load_reduction_percent_collections.abs().mean().groupby('time_interval').mean().copy()
    values.index = (values.index.seconds / 3600).astype(str) + 'h'
    figure.add_trace(go.Bar(
        x=values.index,
        y=values.values,
        name='Average reduction'
    ))
    figure.update_layout(
        xaxis_title='Load reduction duration',
        yaxis_title='Load reduction [%]',
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    )
    figure.write_image(os.path.join(results_path, 'load_reduction_percent_aggregate.pdf'))

    # Load reduction maximum buildings.
    values = load_reduction_percent_collections.abs().max().unstack(level='scenario_name').copy()
    values.index = (values.index.seconds / 3600).astype(str) + 'h'
    figure = go.Figure()
    for column in values.columns:
        figure.add_trace(go.Bar(
            x=values.index,
            y=values.loc[:, column].values,
            name=column[14:].replace("_", " ").upper()
        ))
    figure.update_layout(
        # xaxis_title='Load reduction duration',
        yaxis_title='Maximum load reduction [%]',
        legend=go.layout.Legend(orientation='h', title='Building: ')
    )
    figure.write_image(os.path.join(results_path, 'load_reduction_percent_max_buildings.pdf'))

    # Load reduction mean buildings.
    values = load_reduction_percent_collections.abs().mean().unstack(level='scenario_name').copy()
    values.index = (values.index.seconds / 3600).astype(str) + 'h'
    figure = go.Figure()
    for column in values.columns:
        figure.add_trace(go.Bar(
            x=values.index,
            y=values.loc[:, column].values,
            name=column[14:].replace("_", " ").upper()
        ))
    figure.update_layout(
        # xaxis_title='Load reduction duration',
        yaxis_title='Average load reduction [%]',
        legend=go.layout.Legend(orientation='h', title='Building: ')
    )
    figure.write_image(os.path.join(results_path, 'load_reduction_percent_mean_buildings.pdf'))

    # Load reduction maximum aggregate timeseries.
    values = load_reduction_percent_collections.abs().fillna(0.0).groupby('time_interval', axis='columns').max().copy()
    values.columns = (values.columns.seconds / 3600).astype(str) + 'h'
    figure = go.Figure()
    for column in values.columns:
        figure.add_trace(go.Scatter(
            x=values.index,
            y=values.loc[:, column].values,
            name=column,
            line=go.scatter.Line(shape='hv'),
            fill='tozeroy'
        ))
    figure.update_layout(
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        yaxis_title='Maximum load reduction [%]',
        legend=go.layout.Legend(orientation='h', title='Load reduction duration: ')
    )
    figure.write_image(os.path.join(results_path, 'load_reduction_percent_timeseries_max.pdf'))

    # Load reduction mean aggregate timeseries.
    values = load_reduction_percent_collections.abs().fillna(0.0).groupby('time_interval', axis='columns').mean().copy()
    values.columns = (values.columns.seconds / 3600).astype(str) + 'h'
    figure = go.Figure()
    for column in values.columns:
        figure.add_trace(go.Scatter(
            x=values.index,
            y=values.loc[:, column].values,
            name=column,
            line=go.scatter.Line(shape='hv'),
            fill='tozeroy'
        ))
    figure.update_layout(
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        yaxis_title='Average load reduction [%]',
        legend=go.layout.Legend(orientation='h', title='Load reduction duration: ')
    )
    figure.write_image(os.path.join(results_path, 'load_reduction_percent_timeseries_mean.pdf'))

    # Load reduction building timeseries.
    for scenario_name in scenario_names:
        values = load_reduction_percent_collections.loc[:, (scenario_name, slice(None))].abs().fillna(0.0).copy()
        values.columns = (values.columns.get_level_values('time_interval').seconds / 3600).astype(str) + 'h'
        figure = go.Figure()
        for column in values.columns:
            figure.add_trace(go.Scatter(
                x=values.index,
                y=values.loc[:, column].values,
                name=column,
                line=go.scatter.Line(shape='hv'),
                fill='tozeroy'
            ))
        figure.update_layout(
            title=f'Building: {scenario_name[14:].replace("_", " ").upper()}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            yaxis_title='Load reduction [%]',
            legend=go.layout.Legend(orientation='h', title='Load reduction duration: ')
        )
        figure.write_image(os.path.join(results_path, f'load_reduction_percent_timeseries_{scenario_name}.pdf'))

    # Launch & print results path.
    cobmo.utils.launch(results_path)
    print(f"Results are stored in {results_path}")


if __name__ == '__main__':
    main()
