"""Controller class definition."""

import pandas as pd
import pyomo.environ as pyo
import time as time

import cobmo.config


class Controller(object):
    """Controller object to store the model predictive control problem."""

    def __init__(
            self,
            conn,
            building,
            problem_type='operation'  # Choices: 'operation', 'storage_planning'
    ):
        """Initialize controller object based on given `building` object.

        - The optimization problem is formulated with the Pyomo toolbox.
        """
        time_start = time.clock()
        self.building = building
        self.problem_type = problem_type
        self.problem = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory(cobmo.config.solver_name)
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
            domain=pyo.NonNegativeReals  # TODO: Workaround for proper behavior of battery storage.
        )
        self.problem.variable_output_timeseries = pyo.Var(
            self.building.set_timesteps,
            self.building.set_outputs,
            domain=pyo.Reals
        )
        if problem_type == 'storage_planning':
            self.problem.variable_storage_size = pyo.Var(
                [0],
                domain=pyo.NonNegativeReals
            )
            if 'battery' in self.building.building_scenarios['building_storage_type'][0]:
                self.problem.variable_storage_peak_power = pyo.Var(
                    [0],
                    domain=pyo.NonNegativeReals
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
                if (self.problem_type == 'storage_planning') and ('state_of_charge' in output):
                    if 'sensible' in self.building.building_scenarios['building_storage_type'][0]:
                        self.problem.constraints.add(
                            self.problem.variable_output_timeseries[timestep, output]
                            <=
                            self.building.output_constraint_timeseries_maximum.loc[timestep, output]
                            * self.problem.variable_storage_size[0]
                        )
                    elif 'battery' in self.building.building_scenarios['building_storage_type'][0]:
                        self.problem.constraints.add(
                            self.problem.variable_output_timeseries[timestep, output]
                            <=
                            self.building.output_constraint_timeseries_maximum.loc[timestep, output]
                            * self.problem.variable_storage_size[0]
                            * float(self.building.building_scenarios['storage_depth_of_discharge'][0])
                        )
                elif '_ahu_cool_electric_power' in output:
                    # TODO: Workaround for avoiding unrealistic storage charging demand. Move this to building model.
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

        # Storage peak demand constraint.
        # TODO: Add output variable in building to simplify this.
        if (
            (self.problem_type == 'storage_planning')
            and ('battery' in self.building.building_scenarios['building_storage_type'][0])
        ):
            for timestep in self.building.set_timesteps:
                self.problem.constraints.add(
                    sum(
                        self.problem.variable_output_timeseries[timestep, output]
                        if 'battery_storage_to_zone' in output else 0.0
                        for output in self.building.set_outputs
                    )
                    <=
                    self.problem.variable_storage_peak_power[0]
                )

        # Define components of the objective.
        self.operation_cost = 0.0
        self.investment_cost = 0.0

        # Operation cost factor.
        if self.problem_type == 'storage_planning':
            # Define operation cost factor to scale operation cost to the lifetime of storage.
            self.operation_cost_factor = (
                (pd.to_timedelta('1y') / pd.to_timedelta(timestep_delta))  # Theoretical number of time steps in a year.
                / len(self.building.set_timesteps)  # Actual number of time steps.
                * float(self.building.building_parameters['storage_lifetime'])  # Storage lifetime in years.
                * 14.0  # 14 levels at CREATE Tower. # TODO: Check if considered properly in storage size.
            )
        else:
            # No scaling needed if not running planning problem.
            self.operation_cost_factor = 1.0

        # Operation cost (OPEX).
        for timestep in self.building.set_timesteps:
            for output_power in self.building.set_outputs:
                if 'electric_power' in output_power:
                    self.operation_cost += (
                        self.problem.variable_output_timeseries[timestep, output_power]
                        * timestep_delta.seconds / 3600.0 / 1000.0  # Ws in kWh.
                        * self.building.electricity_prices.loc[timestep, 'price']
                        * self.operation_cost_factor
                    )

        # Investment cost (CAPEX).
        if self.problem_type == 'storage_planning':
            if 'sensible' in self.building.building_scenarios['building_storage_type'][0]:
                if self.building.building_scenarios['investment_sgd_per_X'][0] == 'kwh':
                    self.investment_cost += (
                        # TODO: Make storage temp. difference dynamic based on building model data.
                        self.problem.variable_storage_size[0] * 1000.0 * 4186.0 * 8.0  # m^3 (water; 8 K delta) in Ws (= J).
                        / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
                        * float(self.building.building_scenarios['storage_energy_installation_cost'][0])
                    )
                elif self.building.building_scenarios['investment_sgd_per_X'][0] == 'm3':
                    self.investment_cost += (
                        self.problem.variable_storage_size[0]
                        * float(self.building.building_scenarios['storage_energy_installation_cost'][0])
                    )
            elif 'battery' in self.building.building_scenarios['building_storage_type'][0]:
                self.investment_cost += (
                    self.problem.variable_storage_size[0] / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
                    * float(self.building.building_scenarios['storage_energy_installation_cost'][0])
                    + self.problem.variable_storage_peak_power[0] / 1000.0  # W in kW. # TODO: Check unit of power cost.
                    * float(self.building.building_scenarios['storage_power_installation_cost'][0])
                    # + float(self.building.building_scenarios['storage_fixed_cost'][0])
                    # TODO: Add integer variable to consider fixed storage cost.
                )
            else:
                # Workaround to ensure `variable_storage_size` is zero if building doesn't have storage defined.
                self.investment_cost += self.problem.variable_storage_size[0]

        # Define objective.
        self.problem.objective = pyo.Objective(
            expr=(self.operation_cost + self.investment_cost),
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
            tee=cobmo.config.solver_output  # If True, activate verbose solver output.
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

        # Retrieve objective / cost values.
        if type(self.operation_cost) is float:
            operation_cost = self.operation_cost
        else:
            operation_cost = pyo.value(self.operation_cost)
        if type(self.investment_cost) is float:
            investment_cost = self.investment_cost
        else:
            investment_cost = pyo.value(self.investment_cost)

        # Scale back operation cost.
        # TODO: Revise storage run scripts / payback functions to sort this out internally.
        operation_cost /= self.operation_cost_factor
        if self.problem_type == 'storage_planning':
            operation_cost *= 14.0  # 14 levels at CREATE Tower. # TODO: Check if considered properly in storage size.

        # Retrieve storage size.
        if self.problem_type == 'storage_planning':
            storage_size = self.problem.variable_storage_size[0].value
        else:
            storage_size = None

        # Print results compilation time for debugging.
        print("Controller results compilation time: {:.2f} seconds".format(time.clock() - time_start))

        return (
            control_timeseries,
            state_timeseries,
            output_timeseries,
            operation_cost,
            investment_cost,
            storage_size
        )
