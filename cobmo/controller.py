"""Controller class definition."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import time as time


class Controller(object):
    """Controller object to store the model predictive control problem."""

    def __init__(
            self,
            conn,
            building
    ):
        """Initialize Controller object.

        - Obtains building model information.
        - Creates Pyomo problem.
        """
        time_start = time.clock()
        self.building = building
        self.problem = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('gurobi')
        self.result = None

        # Define sets
        self.problem.set_states = pyo.Set(
            initialize=self.building.set_states
        )
        self.problem.set_controls = pyo.Set(
            initialize=self.building.set_controls
        )
        self.problem.set_disturbances = pyo.Set(
            initialize=self.building.set_disturbances
        )
        self.problem.set_outputs = pyo.Set(
            initialize=self.building.set_outputs
        )
        self.problem.set_outputs_power = pyo.Set(
            initialize=self.building.set_outputs[self.building.set_outputs.str.contains('power')]
        )
        self.problem.set_outputs_temperature = pyo.Set(
            initialize=self.building.set_outputs[self.building.set_outputs.str.contains('temperature')]
        )
        self.problem.set_timesteps = pyo.Set(
            initialize=self.building.set_timesteps
        )
        self.problem.set_timestep_first = pyo.Set(
            initialize=self.building.set_timesteps[0:1]
        )
        self.problem.set_timesteps_without_first = pyo.Set(
            initialize=self.building.set_timesteps[1:]
        )
        self.problem.set_timesteps_without_last = pyo.Set(
            initialize=self.building.set_timesteps[:-1]
        )

        # Store timestep
        self.problem.timestep_delta = self.building.set_timesteps[1] - self.building.set_timesteps[0]

        # Define parameters
        self.problem.parameter_state_matrix = pyo.Param(
            self.problem.set_states,
            self.problem.set_states,
            initialize=self.building.state_matrix.stack().to_dict()
        )
        self.problem.parameter_state_output_matrix = pyo.Param(
            self.problem.set_outputs,
            self.problem.set_states,
            initialize=self.building.state_output_matrix.stack().to_dict()
        )
        self.problem.parameter_control_matrix = pyo.Param(
            self.problem.set_states,
            self.problem.set_controls,
            initialize=self.building.control_matrix.stack().to_dict()
        )
        self.problem.parameter_control_output_matrix = pyo.Param(
            self.problem.set_outputs,
            self.problem.set_controls,
            initialize=self.building.control_output_matrix.stack().to_dict()
        )
        self.problem.parameter_disturbance_matrix = pyo.Param(
            self.problem.set_states,
            self.problem.set_disturbances,
            initialize=self.building.disturbance_matrix.stack().to_dict()
        )
        self.problem.parameter_disturbance_output_matrix = pyo.Param(
            self.problem.set_outputs,
            self.problem.set_disturbances,
            initialize=self.building.disturbance_output_matrix.stack().to_dict()
        )
        self.problem.parameter_disturbance_timeseries = pyo.Param(
            self.problem.set_timesteps,
            self.problem.set_disturbances,
            initialize=self.building.disturbance_timeseries.stack().to_dict()
        )
        self.problem.parameter_output_timeseries_minimum = pyo.Param(
            self.problem.set_timesteps,
            self.problem.set_outputs,
            initialize=self.building.output_constraint_timeseries_minimum.transpose().stack().to_dict()
        )  # TODO: Transpose output_constraint_timeseries_minimum.
        self.problem.parameter_output_timeseries_maximum = pyo.Param(
            self.problem.set_timesteps,
            self.problem.set_outputs,
            initialize=self.building.output_constraint_timeseries_maximum.transpose().stack().to_dict()
        )  # TODO: Transpose output_constraint_timeseries_maximum.

        # Define initial state
        self.problem.parameter_state_initial = pyo.Param(
            self.problem.set_states,
            initialize=pd.Series(
                np.concatenate([
                    0.0  # in all units (m3, kg, kwh)  |  Initialize the storage as empty
                    * np.ones(sum(self.building.set_states.str.contains('state_of_charge'))),
                    26.0  # in °C
                    * np.ones(sum(self.building.set_states.str.contains('temperature'))),
                    100.0  # in ppm
                    * np.ones(sum(self.building.set_states.str.contains('co2_concentration'))),
                    0.013  # in kg(water)/kg(air)
                    * np.ones(sum(self.building.set_states.str.contains('absolute_humidity')))
                ]),
                self.building.set_states
            ).to_dict()
        )  # TODO: Move intial state defintion to building model

        # Define variable bound rules
        def rule_output_bounds(
                problem,
                timestep,
                output
        ):
            return (
                problem.parameter_output_timeseries_minimum[timestep, output],
                problem.parameter_output_timeseries_maximum[timestep, output]
            )

        # Define variables - they are defined as matrixes
        self.problem.variable_state_timeseries = pyo.Var(
            self.problem.set_timesteps,
            self.problem.set_states,
            domain=pyo.Reals
        )
        self.problem.variable_control_timeseries = pyo.Var(
            self.problem.set_timesteps,
            self.problem.set_controls,
            domain=pyo.Reals
        )
        self.problem.variable_output_timeseries = pyo.Var(
            self.problem.set_timesteps,
            self.problem.set_outputs,
            domain=pyo.Reals,
            bounds=rule_output_bounds
        )

# =================================================================================================

        # Define constraint rules
        def rule_state_initial(
                problem,
                timestep_first,
                state
        ):
            # Equality constraint
            return problem.variable_state_timeseries[timestep_first, state] == (
                problem.parameter_state_initial[state]
            )

        def rule_state_equation(
                problem,
                timestep,
                state
        ):
            # State equation
            state_value = 0.0
            for state_other in problem.set_states:  # @contraint@temperature@zone
                state_value += (
                        problem.parameter_state_matrix[state, state_other]
                        * problem.variable_state_timeseries[timestep, state_other]
                )
            for control in problem.set_controls:
                state_value += (
                        problem.parameter_control_matrix[state, control]
                        * problem.variable_control_timeseries[timestep, control]
                )
            for disturbance in problem.set_disturbances:
                state_value += (
                        problem.parameter_disturbance_matrix[state, disturbance]
                        * problem.parameter_disturbance_timeseries[timestep, disturbance]
                )

            # Equality constraint
            return problem.variable_state_timeseries[timestep + problem.timestep_delta, state] == state_value

        def rule_output_equation(
                problem,
                timestep,
                output
        ):
            # State equation
            output_value = 0.0
            for state in problem.set_states:
                output_value += (
                        problem.parameter_state_output_matrix[output, state]
                        * problem.variable_state_timeseries[timestep, state]
                )
            for control in problem.set_controls:
                output_value += (
                        problem.parameter_control_output_matrix[output, control]
                        * problem.variable_control_timeseries[timestep, control]
                )
            for disturbance in problem.set_disturbances:
                output_value += (
                        problem.parameter_disturbance_output_matrix[output, disturbance]
                        * problem.parameter_disturbance_timeseries[timestep, disturbance]
                )

            # Equality constraint
            return problem.variable_output_timeseries[timestep, output] == output_value
# =================================================================================================


# =================================================================================================
        # Define constraints
        self.problem.constraint_state_initial = pyo.Constraint(
            self.problem.set_timestep_first,
            self.problem.set_states,
            rule=rule_state_initial
        )
        self.problem.constraint_state_equation = pyo.Constraint(
            self.problem.set_timesteps_without_last,
            self.problem.set_states,
            rule=rule_state_equation
        )
        self.problem.constraint_output_equation = pyo.Constraint(
            self.problem.set_timesteps,
            self.problem.set_outputs,
            rule=rule_output_equation
        )
# =================================================================================================
        # Define objective rule
        def objective_rule(problem):
            objective_value = 0.0
            for timestep in problem.set_timesteps:
                for output_power in problem.set_outputs_power:  # TODO: Differentiate between thermal and electric.
                    objective_value += problem.variable_output_timeseries[timestep, output_power]
            return objective_value

# =================================================================================================

# =================================================================================================
        # Define objective
        self.problem.objective = pyo.Objective(
            rule=objective_rule,
            sense=1  # Minimize
        )
# =================================================================================================



# =================================================================================================
        # Print setup time for debugging
        print("Controller setup time: {:.2f} seconds".format(time.clock() - time_start))

    def solve(self):
        """Invoke solver on Pyomo problem."""

        # Solve problem
        time_start = time.clock()
        self.result = self.solver.solve(
            self.problem,
            tee=True  # Verbose solver outputs
        )
        print("Controller solve time: {:.2f} seconds".format(time.clock() - time_start))



# =================================================================================================
        # Retrieve results
        time_start = time.clock()
        control_timeseries = pd.DataFrame(
            0.0,
            self.building.set_timesteps,
            self.building.set_controls
        )
        state_timeseries = pd.DataFrame(
            0.0,
            self.building.set_timesteps,
            self.building.set_states
        )
        output_timeseries = pd.DataFrame(
            0.0,
            self.building.set_timesteps,
            self.building.set_outputs
        )
        for timestep in self.building.set_timesteps:
            for control in self.building.set_controls:
                control_timeseries.at[timestep, control] = (
                    self.problem.variable_control_timeseries[timestep, control].value
                )
            for state in self.building.set_states:
                state_timeseries.at[timestep, state] = (
                    self.problem.variable_state_timeseries[timestep, state].value
                )
            for output in self.building.set_outputs:
                output_timeseries.at[timestep, output] = (
                    self.problem.variable_output_timeseries[timestep, output].value
                )
        print("Controller results compilation time: {:.2f} seconds".format(time.clock() - time_start))

        return (
            control_timeseries,
            state_timeseries,
            output_timeseries
        )
