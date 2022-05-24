"""Example run script to demonstrate setting up and solving a simulation problem for building operation with CoBMo."""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import cobmo


def main():

    # Settings.
    scenario_name = "create_level8_4zones_a"
    results_path = cobmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Obtain building model.
    building_model = cobmo.building_model.BuildingModel(scenario_name)

    # Print building model matrices.
    print(f"state_matrix = \n{building_model.state_matrix}")
    print(f"control_matrix = \n{building_model.control_matrix}")
    print(f"disturbance_matrix = \n{building_model.disturbance_matrix}")
    print(f"state_output_matrix = \n{building_model.state_output_matrix}")
    print(f"control_output_matrix = \n{building_model.control_output_matrix}")
    print(f"disturbance_output_matrix = \n{building_model.disturbance_output_matrix}")
    print(f"disturbance_timeseries = \n{building_model.disturbance_timeseries}")

    # Store building model matrices to CSV.
    building_model.state_matrix.to_csv(results_path / "building_state_matrix.csv")
    building_model.control_matrix.to_csv(results_path / "building_control_matrix.csv")
    building_model.disturbance_matrix.to_csv(results_path / "building_disturbance_matrix.csv")
    building_model.state_output_matrix.to_csv(results_path / "building_state_output_matrix.csv")
    building_model.control_output_matrix.to_csv(results_path / "building_control_output_matrix.csv")
    building_model.disturbance_output_matrix.to_csv(results_path / "building_disturbance_output_matrix.csv")
    building_model.disturbance_timeseries.to_csv(results_path / "building_disturbance_timeseries.csv")

    # Define control timeseries, which is required as an input for the simulation.
    control_vector = pd.DataFrame(
        np.ones((len(building_model.timesteps), len(building_model.controls))),
        index=building_model.timesteps,
        columns=building_model.controls,
    )

    # Execute simulation and obtain results.
    (state_vector, output_vector) = building_model.simulate(control_vector)

    # Print results.
    print(f"control_vector = \n{control_vector}")
    print(f"state_vector = \n{state_vector}")
    print(f"output_vector = \n{output_vector}")

    # Store results to CSV.
    control_vector.to_csv(results_path / "control_vector.csv")
    state_vector.to_csv(results_path / "state_vector.csv")
    output_vector.to_csv(results_path / "output_vector.csv")

    # Plot results.
    for output in building_model.outputs:

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=building_model.output_maximum_timeseries.index,
                y=building_model.output_maximum_timeseries.loc[:, output].values,
                name="Maximum",
                line=go.scatter.Line(shape="hv"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=building_model.output_minimum_timeseries.index,
                y=building_model.output_minimum_timeseries.loc[:, output].values,
                name="Minimum",
                line=go.scatter.Line(shape="hv"),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=output_vector.index,
                y=output_vector.loc[:, output].values,
                name="Optimal",
                line=go.scatter.Line(shape="hv"),
            )
        )
        figure.update_layout(
            title=f"Output: {output}",
            xaxis=go.layout.XAxis(tickformat="%H:%M"),
            legend=go.layout.Legend(x=0.99, xanchor="auto", y=0.99, yanchor="auto"),
        )
        # figure.show()
        cobmo.utils.write_figure_plotly(figure, results_path / output)

    # Print results path.
    cobmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == "__main__":
    main()
