"""Example run script to demonstrate setting up and solving an optimization problem for building operation with CoBMo."""

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
    scenario_name = 'create_level8_4zones_a'
    results_path = cobmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Obtain building model.
    building_model = cobmo.building_model.BuildingModel(scenario_name)

    # Print building model matrices and disturbance timeseries.
    print(f"state_matrix = \n{building_model.state_matrix}")
    print(f"control_matrix = \n{building_model.control_matrix}")
    print(f"disturbance_matrix = \n{building_model.disturbance_matrix}")
    print(f"state_output_matrix = \n{building_model.state_output_matrix}")
    print(f"control_output_matrix = \n{building_model.control_output_matrix}")
    print(f"disturbance_output_matrix = \n{building_model.disturbance_output_matrix}")
    print(f"disturbance_timeseries = \n{building_model.disturbance_timeseries}")

    # Store building model matrices and disturbance timeseries as CSV.
    building_model.state_matrix.to_csv(os.path.join(results_path, 'building_state_matrix.csv'))
    building_model.control_matrix.to_csv(os.path.join(results_path, 'building_control_matrix.csv'))
    building_model.disturbance_matrix.to_csv(os.path.join(results_path, 'building_disturbance_matrix.csv'))
    building_model.state_output_matrix.to_csv(os.path.join(results_path, 'building_state_output_matrix.csv'))
    building_model.control_output_matrix.to_csv(os.path.join(results_path, 'building_control_output_matrix.csv'))
    building_model.disturbance_output_matrix.to_csv(os.path.join(results_path, 'building_disturbance_output_matrix.csv'))
    building_model.disturbance_timeseries.to_csv(os.path.join(results_path, 'building_disturbance_timeseries.csv'))

    # Obtain optimization problem.
    optimization_problem = cobmo.optimization_problem.OptimizationProblem(
        building_model
    )

    # Solve optimization problem and obtain results.
    (
        control_vector,
        state_vector,
        output_vector,
        operation_cost,
        investment_cost,  # Zero when running (default) operation problem.
        storage_size  # Zero when running (default) operation problem.
    ) = optimization_problem.solve()

    # Print results.
    print(f"operation_cost = {operation_cost}")
    print(f"control_vector = \n{control_vector}")
    print(f"state_vector = \n{state_vector}")
    print(f"output_vector = \n{output_vector}")

    # Store results TO CSV.
    control_vector.to_csv(os.path.join(results_path, 'control_vector.csv'))
    state_vector.to_csv(os.path.join(results_path, 'state_vector.csv'))
    output_vector.to_csv(os.path.join(results_path, 'output_vector.csv'))

    # Plot results.
    for output in building_model.outputs:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=building_model.output_constraint_timeseries_maximum.index,
            y=building_model.output_constraint_timeseries_maximum.loc[:, output].values,
            name='Maximum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=building_model.output_constraint_timeseries_minimum.index,
            y=building_model.output_constraint_timeseries_minimum.loc[:, output].values,
            name='Minimum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=output_vector.index,
            y=output_vector.loc[:, output].values,
            name='Optimal',
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Output: {output}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
        )
        # figure.show()
        cobmo.utils.write_figure_plotly(figure, os.path.join(results_path, output))

    # Print results path.
    cobmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
