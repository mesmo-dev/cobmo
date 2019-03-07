"""Controller class definition."""

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
        self.problem = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('gurobi')

        # Define sets
        self.problem.set_states = pyo.Set(
            initialize=building.set_states
        )
        self.problem.set_controls = pyo.Set(
            initialize=building.set_controls
        )
        self.problem.set_disturbances = pyo.Set(
            initialize=building.set_disturbances
        )
        self.problem.set_outputs = pyo.Set(
            initialize=building.set_outputs
        )
        self.problem.set_outputs_electric = pyo.Set(
            initialize=building.set_outputs[building.set_outputs.str.contains('electric_power')]
        )
        self.problem.set_timesteps = pyo.Set(
            initialize=building.set_timesteps
        )
        self.problem.set_timesteps_without_first = pyo.Set(
            initialize=building.set_timesteps[1:]
        )
        self.problem.set_timesteps_without_last = pyo.Set(
            initialize=building.set_timesteps[:-1]
        )

        # Store timestep
        self.problem.timestep_delta = building.set_timesteps[1] - building.set_timesteps[0]

        # Define parameters
        self.problem.parameter_state_matrix = pyo.Param(
            self.problem.set_states,
            self.problem.set_states,
            initialize=building.state_matrix.stack().to_dict()
        )
        self.problem.parameter_state_output_matrix = pyo.Param(
            self.problem.set_outputs,
            self.problem.set_states,
            initialize=building.state_output_matrix.stack().to_dict()
        )
        self.problem.parameter_control_matrix = pyo.Param(
            self.problem.set_states,
            self.problem.set_controls,
            initialize=building.control_matrix.stack().to_dict()
        )
        self.problem.parameter_control_output_matrix = pyo.Param(
            self.problem.set_outputs,
            self.problem.set_controls,
            initialize=building.control_output_matrix.stack().to_dict()
        )
        self.problem.parameter_disturbance_matrix = pyo.Param(
            self.problem.set_states,
            self.problem.set_disturbances,
            initialize=building.disturbance_matrix.stack().to_dict()
        )
        self.problem.parameter_disturbance_output_matrix = pyo.Param(
            self.problem.set_outputs,
            self.problem.set_disturbances,
            initialize=building.disturbance_output_matrix.stack().to_dict()
        )
        self.problem.parameter_disturbance_timeseries = pyo.Param(
            self.problem.set_timesteps,
            self.problem.set_disturbances,
            initialize=building.disturbance_timeseries.stack().to_dict()
        )
        self.problem.parameter_output_timeseries_minimum = pyo.Param(
            self.problem.set_timesteps,
            self.problem.set_outputs,
            initialize=building.output_constraint_timeseries_minimum.transpose().stack().to_dict()
        )  # TODO: Transpose output_constraint_timeseries_minimum.
        self.problem.parameter_output_timeseries_maximum = pyo.Param(
            self.problem.set_timesteps,
            self.problem.set_outputs,
            initialize=building.output_constraint_timeseries_maximum.transpose().stack().to_dict()
        )  # TODO: Transpose output_constraint_timeseries_maximum.

        # Define variables
        self.problem.variable_state_timeseries = pyo.Var(
            self.problem.set_timesteps,
            self.problem.set_states
        )
        self.problem.variable_control_timeseries = pyo.Var(
            self.problem.set_timesteps,
            self.problem.set_controls
        )
        self.problem.variable_output_timeseries = pyo.Var(
            self.problem.set_timesteps,
            self.problem.set_outputs,
        )

        # Define constraint rules
        def rule_state_equation(
                problem,
                timestep,
                state
        ):
            # State equation
            state_value = 0.0
            for state_other in problem.set_states:
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

        def rule_output_bounds(
                problem,
                timestep,
                output
        ):
            return (
                problem.parameter_output_timeseries_minimum[timestep, output],
                problem.variable_output_timeseries[timestep, output],
                problem.parameter_output_timeseries_maximum[timestep, output]
            )

        # Define constraints
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
        self.problem.constraint_output_bounds = pyo.Constraint(
            self.problem.set_timesteps_without_first,
            self.problem.set_outputs,
            rule=rule_output_bounds
        )

        # Define objective rule
        def objective_rule(problem):
            objective_value = 0.0
            for timestep in problem.set_timesteps:
                for output in problem.set_outputs_electric:
                    objective_value += problem.variable_output_timeseries[timestep, output]
            return objective_value

        # Define objective
        self.problem.objective = pyo.Objective(
            rule=objective_rule,
            sense=1  # Minimize
        )

        # Print setup time for debugging
        print("Controller setup time: {:.2f} seconds".format(time.clock() - time_start))

    def solve(self):
        """Invoke solver on Pyomo problem."""
        time_start = time.clock()
        result = self.solver.solve(self.problem)
        print("Controller solve time: {:.2f} seconds".format(time.clock() - time_start))
        return result
