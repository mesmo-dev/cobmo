"""
Building model main function definitions
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import cobmo.building
import cobmo.controller
import cobmo.utils
import cobmo.config
import datetime as dt
import pyomo.environ as pyo


def connect_database(
        data_path=cobmo.config.data_path,
        overwrite_database=True
):
    # Create database, if none
    if overwrite_database or not os.path.isfile(os.path.join(data_path, 'data.sqlite')):
        cobmo.utils.create_database(
            sqlite_path=os.path.join(data_path, 'data.sqlite'),
            sql_path=os.path.join(cobmo.config.cobmo_path, 'cobmo', 'database_schema.sql'),
            csv_path=data_path
        )

    conn = sqlite3.connect(os.path.join(data_path, 'data.sqlite'))
    return conn


def get_building_model(
        scenario_name='scenario_default',
        conn=connect_database()
):
    building = cobmo.building.Building(conn, scenario_name)
    return building


def example():
    """
    Example script
    """

    conn = connect_database()

    building_storage_types = pd.read_sql(
        """
        select * from building_storage_types
        """,
        conn,
        index_col='building_storage_type'
        # Indexing to allow precise modification of the dataframe.
        # If this is used you need to reindex as pandas when using to_sql (meaning NOT using "index=False")
    )

    # Here make the changes to the data in the sql
    building_storage_types.at['sensible_thermal_storage_default', 'storage_round_trip_efficiency'] = 0.6

    # print('\nbuilding_storage_types in main = ')
    # print(building_storage_types)

    building_storage_types.to_sql(
        'building_storage_types',
        con=conn,
        if_exists='replace'
        # index=False
    )

    # NB: All the changes to the sql need to be done BEFORE getting the building_model
    building = get_building_model(conn=conn)

    # Define initial state and control timeseries
    state_initial = pd.Series(
        np.concatenate([
            26.0  # in °C
            * np.ones(sum(building.set_states.str.contains('temperature'))),
            100.0  # in ppm
            * np.ones(sum(building.set_states.str.contains('co2_concentration'))),
            0.013  # in kg(water)/kg(air)
            * np.ones(sum(building.set_states.str.contains('absolute_humidity'))),
            0.0  # in all the storage units (sensible: m3 | PCM: kg | battery: kWh)
            * np.ones(sum(building.set_states.str.contains('state_of_charge'))),
            0.0  # Mass factor must be coherent with initial volume of bottom layer
            * np.ones(sum(building.set_states.str.contains('storage_mass_factor')))
        ]),
        building.set_states
    )  # TODO: Move intial state defintion to building model
    control_timeseries_simulation = pd.DataFrame(
        np.random.rand(len(building.set_timesteps), len(building.set_controls)),
        building.set_timesteps,
        building.set_controls
    )

    # Define augemented state space model matrices
    building.define_augmented_model()

    # Run simulation
    (
        state_timeseries_simulation,
        output_timeseries_simulation
    ) = building.simulate(
        state_initial=state_initial,
        control_timeseries=control_timeseries_simulation
    )

    # Outputs for debugging
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.state_matrix=")
    # print(building.state_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.control_matrix=")
    # print(building.control_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.disturbance_matrix=")
    # print(building.disturbance_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.state_output_matrix=")
    # print(building.state_output_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.control_output_matrix=")
    # print(building.control_output_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.disturbance_output_matrix=")
    # print(building.disturbance_output_matrix)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("control_timeseries_simulation=")
    # print(control_timeseries_simulation)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("building.disturbance_timeseries=")
    # print(building.disturbance_timeseries)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("state_timeseries_simulation=")
    # print(state_timeseries_simulation)
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("output_timeseries_simulation=")
    # print(output_timeseries_simulation)
    # print("-----------------------------------------------------------------------------------------------------------")

    # file_output_text = open("my_file_out.txt", "w")
    # file_output_text.write(building.state_matrix)

    # Run controller
    controller = cobmo.controller.Controller(
        conn=conn,
        building=building
    )
    (
        control_timeseries_controller,
        state_timeseries_controller,
        output_timeseries_controller,
        storage_size,
        optimum_obj
    ) = controller.solve()

    # Creating the higher level problem
    problem = pyo.ConcreteModel()
    solver = pyo.SolverFactory('gurobi')

    # Parameters
    storage_capex = 300.0
    problem.param_baseline_yearly_cost = pyo.Param(
        initialize=3.8341954e+02 * 260.0
    )
    problem.param_storage_capex = pyo.Param(
        initialize=storage_capex
    )

    # Variables
    problem.var_storage_size = pyo.Var(
        domain=pyo.Reals,
        bounds=(0.0, np.inf),
        initialize=0.0
    )
    problem.var_storage_yearly_cost = pyo.Var(
        domain=pyo.Reals,
        bounds=(0.0, np.inf),
        initialize=0.0
    )
    problem.var_years = pyo.Var(
        domain=pyo.Reals,
        bounds=(0.0, 70.0),
        initialize=0.0
    )
    problem.var_yearly_savings = pyo.Var(
        domain=pyo.Reals,
        bounds=(0.0, np.inf),
        initialize=0.0
    )

    # Rules
    def rule_storage_size(p):

        (_, _, _, size, _) = controller.solve()
        return p.var_storage_size == size

    def rule_storage_yearly_cost(p):
        (_, _, _, _, cost) = controller.solve()
        return p.var_storage_yearly_cost == cost*260.0

    def rule_yearly_savings(p):
        return (
            p.var_yearly_savings == problem.param_baseline_yearly_cost - p.var_storage_yearly_cost
        )

    def rule_obj_g_zero(p):
        return (
            (
                    p.var_years * p.var_yearly_savings
                    - p.param_storage_capex * p.var_storage_size
                    + 100.0  # fixed cost
            ) >= 0.0
        )

    # Constraints
    problem.constraint_storage_size = pyo.Constraint(
        rule=rule_storage_size
    )
    problem.constraint_storage_yearly_cost = pyo.Constraint(
        rule=rule_storage_yearly_cost
    )
    problem.constraint_yearly_savings = pyo.Constraint(
        rule=rule_yearly_savings
    )
    problem.constraint_obj_g_zero = pyo.Constraint(
        rule=rule_obj_g_zero
    )
    problem.constraint_obj_g_zero.activate()

    # Objective rule
    def rule_obj(p):
        obj = (
            p.var_years * p.var_yearly_savings
            - p.param_storage_capex * p.var_storage_size
            + 100.0  # fixed cost
        )
        return obj

    # Objective definition + solve
    problem.objective = pyo.Objective(
        rule=rule_obj,
        sense=1
    )
    results = solver.solve(
        problem,
        tee=True
    )
    print(results)

    # Retrieve variables
    storage_size = problem.var_storage_size.value
    storage_yearly_cost = problem.var_storage_yearly_cost.value
    years = problem.var_years.value
    yearly_savings = problem.var_yearly_savings.value

    objective = (
                years * yearly_savings
                - storage_capex * storage_size
                + 100.0  # fixed cost
            )

    print('============= RESULTS')
    print('\nsize = %.1f'
          '  |  storage yearly cost = %.1f'
          '  |  years = %.1f'
          '  |  yearly_savings = %.1f'
          '  |  obj = %.1f\n' % (storage_size, storage_yearly_cost, years, yearly_savings, objective))


if __name__ == "__main__":
    example()


