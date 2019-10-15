"""Controller class definition."""

import pandas as pd
import pyomo.environ as pyo
import time as time


class ControllerBaseline(object):
    """Controller object to store the model predictive control problem."""

    def __init__(
            self,
            conn,
            building
    ):
        """Initialize controller object based on given `building` object.

        - The optimization problem is formulated with the Pyomo toolbox.
        """

        time_start = time.clock()
        self.building = building
        self.problem = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory('gurobi')
        self.result = None

        # Define variables.
        self.problem.variable_state_timeseries = pyo.Var(
            self.building.set_timesteps,
            self.building.set_states,
            domain=pyo.Reals
        )
        self.problem.variable_control_timeseries = pyo.Var(
            self.building.set_timesteps,
            self.building.set_controls,
            domain=pyo.NonNegativeReals  # TODO: Workaround for unrealistic storage behavior.
        )
        self.problem.variable_output_timeseries = pyo.Var(
            self.building.set_timesteps,
            self.building.set_outputs,
            domain=pyo.Reals
        )

        # Define constraints.
        self.problem.constraints = pyo.ConstraintList()

        # Initial state constraint.
        for state in self.building.set_states:
            self.problem.constraints.add(
                self.problem.variable_state_timeseries[self.building.set_timesteps[0], state]
                ==
                self.building.set_state_initial[state]
            )

        # State equation constraint.
        # TODO: Move timestep_delta into building model.
        timestep_delta = self.building.set_timesteps[1] - self.building.set_timesteps[0]
        for state in self.building.set_states:
            for timestep in self.building.set_timesteps[:-1]:
                self.problem.constraints.add(
                    self.problem.variable_state_timeseries[timestep + timestep_delta, state]
                    ==
                    (
                        sum(
                            self.building.state_matrix.loc[state, state_other]
                            * self.problem.variable_state_timeseries[timestep, state_other]
                            for state_other in self.building.set_states
                        )
                        + sum(
                            self.building.control_matrix.loc[state, control]
                            * self.problem.variable_control_timeseries[timestep, control]
                            for control in self.building.set_controls
                        )
                        + sum(
                            self.building.disturbance_matrix.loc[state, disturbance]
                            * self.building.disturbance_timeseries.loc[timestep, disturbance]
                            for disturbance in self.building.set_disturbances
                        )
                    )
                )

        # Output equation constraint.
        for output in self.building.set_outputs:
            for timestep in self.building.set_timesteps:
                self.problem.constraints.add(
                    self.problem.variable_output_timeseries[timestep, output]
                    ==
                    (
                        sum(
                            self.building.state_output_matrix.loc[output, state]
                            * self.problem.variable_state_timeseries[timestep, state]
                            for state in self.building.set_states
                        )
                        + sum(
                            self.building.control_output_matrix.loc[output, control]
                            * self.problem.variable_control_timeseries[timestep, control]
                            for control in self.building.set_controls
                        )
                        + sum(
                            self.building.disturbance_output_matrix.loc[output, disturbance]
                            * self.building.disturbance_timeseries.loc[timestep, disturbance]
                            for disturbance in self.building.set_disturbances
                        )
                    )
                )

        # Output minimum / maximum bounds constraint.
        for output in self.building.set_outputs:
            for timestep in self.building.set_timesteps:
                # Minimum.
                self.problem.constraints.add(
                    self.problem.variable_output_timeseries[timestep, output]
                    >=
                    self.building.output_constraint_timeseries_minimum.loc[timestep, output]
                )

                # Maximum.
                if '_ahu_cool_electric_power' in output:
                    # TODO: Putting this for symmetry to storage problem.
                    self.problem.constraints.add(
                        self.problem.variable_output_timeseries[timestep, output]
                        <=
                        20000.0
                    )
                else:
                    self.problem.constraints.add(
                        self.problem.variable_output_timeseries[timestep, output]
                        <=
                        self.building.output_constraint_timeseries_maximum.loc[timestep, output]
                    )

        # Define objective.
        objective_value = 0.0
        for timestep in self.building.set_timesteps:
            for output in self.building.set_outputs:
                if 'electric_power' in output:
                    objective_value += (
                        self.problem.variable_output_timeseries[timestep, output]
                        * timestep_delta.seconds / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
                        * self.building.electricity_prices.loc[timestep, 'price']
                        * 14.0 * 260.0 * float(building.building_parameters['storage_lifetime'])
                        # OPEX for storage lifetime (14 levels at CREATE Tower; 260 working days per year).
                    )

        self.problem.objective = pyo.Objective(
            expr=objective_value,
            sense=pyo.minimize
        )

        # Print setup time for debugging.
        print("Controller setup time: {:.2f} seconds".format(time.clock() - time_start))

    def solve(self):
        """Invoke solver on Pyomo problem."""

        # Solve problem.
        time_start = time.clock()
        self.result = self.solver.solve(
            self.problem,
            tee=True  # Verbose solver outputs.
        )

        # Print solve time for debugging.
        print("Controller solve time: {:.2f} seconds".format(time.clock() - time_start))

        # Retrieve results.
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

        # Retrieve objective value.
        # - Objective value here only for OPEX and in SGD/day.
        objective_value = 0.0
        # TODO: Move timestep_delta into building model.
        timestep_delta = self.building.set_timesteps[1] - self.building.set_timesteps[0]
        for timestep in self.building.set_timesteps:
            for output_power in self.building.set_outputs:
                if 'electric_power' in output_power:
                    objective_value += (
                        self.problem.variable_output_timeseries[timestep, output_power].value
                        * timestep_delta.seconds / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
                        * self.building.electricity_prices.loc[timestep, 'price']
                        * 14.0  # 14 levels at CREATE Tower.
                    )

        # Print results compilation time for debugging.
        print("Controller results compilation time: {:.2f} seconds".format(time.clock() - time_start))

        return (
            control_timeseries,
            state_timeseries,
            output_timeseries,
            objective_value
        )
