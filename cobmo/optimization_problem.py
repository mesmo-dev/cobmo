"""Optimization problem module."""

import pandas as pd
import pyomo.environ as pyo
import time as time
import typing

import cobmo.building_model
import cobmo.config

logger = cobmo.config.get_logger(__name__)


class OptimizationProblem(object):
    """Optimization problem object consisting of the pyomo problem object which encapsulates optimization variables,
    constraints and objective function definitions for given building model object and problem type.

    - The `problem_type` keyword argument can be utilized to differentiate constraint and objective function definitions
      depending on the optimization problem type, e.g., optimal operation vs. optimal storage planning.

    Syntax
        ``OptimizationProblem(building_model)``: Instantiate optimization problem for given `scenario_name`.
        the problem type defaults to `operation`, if not specified as keyword argument.

    Parameters:
        building_model (cobmo.building_model.BuildingModel): Building model object.

    Keyword Arguments:
        problem_type (str): Optimization problem type. Choices: `operation`, `storage_planning`,
            `storage_planning_baseline`, `load_reduction`, `price_sensitivity`, `load_maximization`,
            `load_minimization`, `robust_optimization`, `price_scenario`, `rolling_forecast`. Default: `operation`
        output_vector_reference (pd.DataFrame): Only for problem type `load_reduction`. Defines the reference
            for which the load reduction is calculated.
        load_reduction_start_time (pd.Timestamp): Only for problem type `load_reduction`. Defines the start time
            for which the load reduction is calculated.
        load_reduction_end_time (pd.Timestamp): Only for problem type `load_reduction`. Defines the end time
            for which the load reduction is calculated.
        price_sensitivity_factor: np.float: Only for problem type `price_sensitivity`. Defines the factor with which
            the price value is modified.
        price_sensitivity_timestep (pd.Timedelta): Only for problem type `price_sensitivity`. Defines the timstep for
            which the price value is modified.
        load_maximization_time (pd.Timestamp): Only for problem type `load_maximization`. Defines the timestep at
            which the load is maximized.
    """

    def __init__(
            self,
            building: cobmo.building_model.BuildingModel,
            problem_type='operation',
            # Choices: `operation`, `storage_planning`, `storage_planning_baseline`, `load_reduction`,
            # `price_sensitivity`, `load_maximization`, `load_minimization`, `robust_optimization`,
            # `price_scenario`, `rolling_forecast`.
            output_vector_reference=None,
            load_reduction_start_time=None,
            load_reduction_end_time=None,
            price_sensitivity_factor=None,
            price_sensitivity_timestep=None,
            load_maximization_time=None,
            price_scenario_timestep=None,
            price_point=None,
            price_forecast=None,
            actual_dispatch=None
    ):

        # TODO: Only store absolutely necessary parameters.
        time_start = time.time()
        self.building = building
        self.problem_type = problem_type
        # self.output_vector_reference = output_vector_reference
        # self.load_reduction_start_time = load_reduction_start_time
        # self.load_reduction_end_time = load_reduction_end_time
        self.price_sensitivity_factor = price_sensitivity_factor
        self.price_sensitivity_timestep = price_sensitivity_timestep
        self.load_maximization_time = load_maximization_time
        self.price_scenario_timestep = price_scenario_timestep
        self.price_point = price_point
        self.price_forecast = price_forecast
        self.actual_dispatch = actual_dispatch

        # Copy `electricity_price_timeseries` to allow local modifications.
        self.electricity_price_timeseries = self.building.electricity_price_timeseries.copy()
        self.electricity_price_distribution_timeseries = self.building.electricity_price_distribution_timeseries.copy()

        self.problem = pyo.ConcreteModel()
        self.solver = pyo.SolverFactory(cobmo.config.config['optimization']['solver_name'])
        self.result = None

        # Define variables.
        self.problem.variable_state_vector = pyo.Var(
            self.building.timesteps,
            self.building.states,
            domain=pyo.Reals
        )
        self.problem.variable_control_vector = pyo.Var(
            self.building.timesteps,
            self.building.controls,
            domain=pyo.NonNegativeReals  # TODO: Workaround for proper behavior of battery storage.
        )
        self.problem.variable_output_vector = pyo.Var(
            self.building.timesteps,
            self.building.outputs,
            domain=pyo.Reals
        )
        if self.problem_type == 'storage_planning':
            self.problem.variable_storage_capacity = pyo.Var(
                [0],
                domain=pyo.NonNegativeReals
            )
            self.problem.variable_storage_peak_power = pyo.Var(
                [0],
                domain=pyo.NonNegativeReals
            )
            # Variable to describe if storage exists (= 1) or not (= 0).
            self.problem.variable_storage_exists = pyo.Var(
                [0],
                domain=pyo.Binary
            )
        if self.problem_type == 'storage_planning_baseline':
            # Force storage size to zero for baseline case.
            self.problem.variable_storage_capacity = [0.0]
        if self.problem_type == 'load_reduction':
            self.problem.variable_load_reduction = pyo.Var(
                [0],
                domain=pyo.NonNegativeReals
            )
        # TODO: rename variables, move gamma to input
        if self.problem_type == 'robust_optimization':
            self.problem.variable_z = pyo.Var(domain=pyo.NonNegativeReals)
            self.problem.variable_gamma = len(self.building.timesteps)
            self.problem.variable_q = pyo.Var(self.building.timesteps, domain=pyo.NonNegativeReals)
            self.problem.variable_y = pyo.Var(self.building.timesteps, domain=pyo.NonNegativeReals)

        if self.problem_type == 'rolling_forecast':
            self.problem.variable_z = pyo.Var(domain=pyo.NonNegativeReals)
            self.problem.variable_gamma = len(self.building.timesteps)/2
            self.problem.variable_q = pyo.Var(self.building.timesteps, domain=pyo.NonNegativeReals)
            self.problem.variable_y = pyo.Var(self.building.timesteps, domain=pyo.NonNegativeReals)

        # Define constraints.
        self.problem.constraints = pyo.ConstraintList()

        # Initial state constraint.
        for state in self.building.states:
            self.problem.constraints.add(
                self.problem.variable_state_vector[self.building.timesteps[0], state]
                ==
                self.building.state_vector_initial[state]
            )

        # State equation constraint.
        for state in self.building.states:
            for timestep in self.building.timesteps[:-1]:
                self.problem.constraints.add(
                    self.problem.variable_state_vector[timestep + self.building.timestep_interval, state]
                    ==
                    (
                        sum(
                            self.building.state_matrix.loc[state, state_other]
                            * self.problem.variable_state_vector[timestep, state_other]
                            for state_other in self.building.states
                        )
                        + sum(
                            self.building.control_matrix.loc[state, control]
                            * self.problem.variable_control_vector[timestep, control]
                            for control in self.building.controls
                        )
                        + sum(
                            self.building.disturbance_matrix.loc[state, disturbance]
                            * self.building.disturbance_timeseries.loc[timestep, disturbance]
                            for disturbance in self.building.disturbances
                        )
                    )
                )

        # Output equation constraint.
        for output in self.building.outputs:
            for timestep in self.building.timesteps:
                self.problem.constraints.add(
                    self.problem.variable_output_vector[timestep, output]
                    ==
                    (
                        sum(
                            self.building.state_output_matrix.loc[output, state]
                            * self.problem.variable_state_vector[timestep, state]
                            for state in self.building.states
                        )
                        + sum(
                            self.building.control_output_matrix.loc[output, control]
                            * self.problem.variable_control_vector[timestep, control]
                            for control in self.building.controls
                        )
                        + sum(
                            self.building.disturbance_output_matrix.loc[output, disturbance]
                            * self.building.disturbance_timeseries.loc[timestep, disturbance]
                            for disturbance in self.building.disturbances
                        )
                    )
                )

        # Output minimum / maximum bounds constraint.
        for output in self.building.outputs:
            for timestep in self.building.timesteps:
                # Minimum.
                if ('temperature' in output) and (self.problem_type == 'load_maximization'):
                    if timestep == (self.load_maximization_time + self.building.timestep_interval):
                        self.problem.constraints.add(
                            self.problem.variable_output_vector[timestep, output]
                            ==
                            self.building.output_minimum_timeseries.loc[timestep, output]
                        )
                else:
                    self.problem.constraints.add(
                        self.problem.variable_output_vector[timestep, output]
                        >=
                        self.building.output_minimum_timeseries.loc[timestep, output]
                    )

                # Maximum.
                if (
                    ((self.problem_type == 'storage_planning') or (self.problem_type == 'storage_planning_baseline'))
                    and ('state_of_charge' in output)
                ):
                    # Storage planning constraints.
                    if 'sensible' in self.building.building_data.scenarios['storage_type']:
                        self.problem.constraints.add(
                            self.problem.variable_output_vector[timestep, output]
                            <=
                            self.problem.variable_storage_capacity[0]
                            * self.building.parse_parameter('water_density')  # TODO: Replace parse_parameter.
                        )
                    elif 'battery' in self.building.building_data.scenarios['storage_type']:
                        self.problem.constraints.add(
                            self.problem.variable_output_vector[timestep, output]
                            <=
                            self.problem.variable_storage_capacity[0]
                            * self.building.building_data.scenarios['storage_battery_depth_of_discharge']
                        )
                else:
                    self.problem.constraints.add(
                        self.problem.variable_output_vector[timestep, output]
                        <=
                        self.building.output_maximum_timeseries.loc[timestep, output]
                    )

        # Storage planning auxiliary constraints.
        if self.problem_type == 'storage_planning':
            for timestep in self.building.timesteps:
                # Storage peak demand constraint.
                self.problem.constraints.add(
                    sum(
                        self.problem.variable_output_vector[timestep, output]
                        if ('storage_charge' in output) and ('electric_power' in output) else 0.0
                        for output in self.building.outputs
                    )
                    <=
                    self.problem.variable_storage_peak_power[0]
                )

                # Storage existence constraint.
                self.problem.constraints.add(
                    self.problem.variable_storage_capacity[0]
                    <=
                    self.problem.variable_storage_exists[0]
                    * 1.0e100  # Large constant as replacement for infinity.
                )

        # Demand side flexibility auxiliary constraints.
        elif self.problem_type == 'load_reduction':
            self.define_load_reduction_constraints(
                output_vector_reference,
                load_reduction_start_time,
                load_reduction_end_time
            )

        # Robust optimization additional constraints.
        elif self.problem_type == 'robust_optimization':
            for timestep in self.building.timesteps:
                self.problem.constraints.add(
                    self.problem.variable_z + self.problem.variable_q[timestep]
                    >=
                    (self.electricity_price_distribution_timeseries.loc[timestep, 'delta_lower']
                     * self.problem.variable_y[timestep])
                )
                self.problem.constraints.add(
                    (
                        self.problem.variable_output_vector[timestep, 'grid_electric_power']
                        * self.building.timestep_interval.seconds / 3600.0 / 1000.0
                    )
                    <=
                    self.problem.variable_y[timestep]
                )

        elif self.problem_type == 'rolling_forecast':
            for timestep in self.building.timesteps:
                if timestep < self.price_scenario_timestep:
                    self.problem.constraints.add(
                        self.problem.variable_output_vector[timestep, 'grid_electric_power'] == \
                        self.actual_dispatch.at[timestep, 'actual_dispatch']
                    )
                    continue
                elif timestep == self.price_scenario_timestep:
                    continue
                self.problem.constraints.add(
                    self.problem.variable_z + self.problem.variable_q[timestep]
                    >=
                    ((self.price_forecast.loc[timestep, 'upper_limit']-self.price_forecast.loc[timestep, 'expected_price'])
                     * self.problem.variable_y[timestep])
                )
                self.problem.constraints.add(
                    (self.problem.variable_output_vector[timestep, 'grid_electric_power']
                    * self.building.timestep_interval.seconds / 3600.0 / 1000.0)
                    <=
                    self.problem.variable_y[timestep]
                )

        # Define components of the objective.
        self.operation_cost = 0.0
        self.investment_cost = 0.0

        # Operation cost factor.
        if (self.problem_type == 'storage_planning') or (self.problem_type == 'storage_planning_baseline'):
            # Define operation cost factor to scale operation cost to the lifetime of storage.
            self.operation_cost_factor = (
                (pd.to_timedelta('1y') / pd.to_timedelta(self.building.timestep_interval))  # Time steps per year.
                / len(self.building.timesteps)  # Actual number of time steps.
                * self.building.building_data.scenarios['storage_lifetime']  # Storage lifetime in years.
                * 14.0  # 14 levels at CREATE Tower. # TODO: Check if considered properly in storage size.
            )
        elif self.problem_type == 'load_reduction':
            # Adjust weight of operation cost when running load reduction problem.
            # - Workaround for unrealistic demand when not considering operation cost at all.
            # - This is a tuning parameter (has impact on load reduction result).
            self.operation_cost_factor = 1.0e-6
        else:
            # No scaling needed if not running planning problem.
            self.operation_cost_factor = 1.0

        # Modify price for price sensitivity evaluation.
        if self.problem_type == 'price_sensitivity':
            self.electricity_price_timeseries.at[self.price_sensitivity_timestep, 'price'] *= (
                self.price_sensitivity_factor
            )

        # Operation cost (OPEX).
        for timestep in self.building.timesteps:
            for output in self.building.outputs:
                if self.problem_type == 'load_minimization':
                    if output == 'grid_electric_power':
                        self.operation_cost += self.problem.variable_output_vector[timestep, output]
                elif self.problem_type == 'robust_optimization':
                    if output == 'grid_electric_power':
                        self.operation_cost += (
                            self.problem.variable_output_vector[timestep, output]
                            * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                            * self.electricity_price_distribution_timeseries.loc[timestep, 'price_mean']
                            * self.operation_cost_factor
                            + self.problem.variable_q[timestep]
                        )
                    if timestep == self.building.timesteps[-1]:
                        self.operation_cost += self.problem.variable_z * self.problem.variable_gamma
                elif self.problem_type == 'price_scenario':
                    if output == 'grid_electric_power':
                        if timestep == self.price_scenario_timestep:
                            self.operation_cost += (
                                self.problem.variable_output_vector[timestep, output]
                                * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                                * self.price_point
                                * self.operation_cost_factor
                            )
                        else:
                            self.operation_cost += (
                                self.problem.variable_output_vector[timestep, output]
                                * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                                * self.electricity_price_distribution_timeseries.loc[timestep, 'price_mean']
                                * self.operation_cost_factor
                            )
                elif self.problem_type == 'rolling_forecast':
                    if output == 'grid_electric_power':
                        if timestep < self.price_scenario_timestep:
                            self.operation_cost += (
                                self.problem.variable_output_vector[timestep, output]
                                * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                                * self.actual_dispatch.loc[timestep, 'clearing_price']
                                * self.operation_cost_factor
                            )
                        elif timestep == self.price_scenario_timestep:
                            self.operation_cost += (
                                self.problem.variable_output_vector[timestep, output]
                                * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                                * self.price_point
                                * self.operation_cost_factor
                            )
                        else:
                            self.operation_cost += (
                                self.problem.variable_output_vector[timestep, output]
                                * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                                * self.price_forecast.loc[timestep, 'expected_price']
                                * self.operation_cost_factor
                                + self.problem.variable_q[timestep]
                            )
                    if timestep == self.building.timesteps[-1]:
                        self.operation_cost += self.problem.variable_z * self.problem.variable_gamma
                else:
                    if output == 'grid_electric_power':
                        self.operation_cost += (
                            self.problem.variable_output_vector[timestep, output]
                            * self.building.timestep_interval.seconds / 3600.0 / 1000.0  # W in kWh.
                            * self.electricity_price_timeseries.loc[timestep, 'price']
                            * self.operation_cost_factor
                        )

        # Investment cost (CAPEX).
        if self.problem_type == 'storage_planning':
            if 'sensible' in self.building.building_data.scenarios['storage_type']:
                self.investment_cost += (
                    self.problem.variable_storage_capacity[0]  # In m3.
                    * self.building.building_data.scenarios['storage_planning_energy_installation_cost']  # In SGD/m3.
                    # TODO: Currently, power / fixed cost are set to zero for sensible thermal storage in the database.
                    + self.problem.variable_storage_peak_power[0] / 1000.0  # W in kW.
                    * self.building.building_data.scenarios['storage_planning_power_installation_cost']  # In SGD/kW.
                    + self.problem.variable_storage_exists[0]  # No unit.
                    * self.building.building_data.scenarios['storage_planning_fixed_installation_cost']  # In SGD.
                )
            elif 'battery' in self.building.building_data.scenarios['storage_type']:
                self.investment_cost += (
                    self.problem.variable_storage_capacity[0] / 3600.0 / 1000.0  # Ws in kWh (J in kWh).
                    * self.building.building_data.scenarios['storage_planning_energy_installation_cost']
                    # TODO: Validate unit of power cost.
                    + self.problem.variable_storage_peak_power[0] / 1000.0  # W in kW.
                    * self.building.building_data.scenarios['storage_planning_power_installation_cost']  # In SGD/kW
                    + self.problem.variable_storage_exists[0]  # No unit.
                    * self.building.building_data.scenarios['storage_planning_fixed_installation_cost']  # In SGD.
                )
        elif self.problem_type == 'load_reduction':
            # TODO: Introduce dedicated cost for demand side flexibility indicators.
            self.investment_cost -= (
                1e6  # Large weight for this part of the objective, to obtain maximum theoretical load reduction.
                * self.problem.variable_load_reduction[0]  # In percent.
            )

        # Define objective.
        self.problem.objective = pyo.Objective(
            expr=(self.operation_cost + self.investment_cost),
            sense=pyo.minimize
        )

        # Print setup time for debugging.
        logger.debug("OptimizationProblem setup time: {:.2f} seconds".format(time.time() - time_start))

    def define_load_reduction_constraints(
            self,
            output_vector_reference,
            load_reduction_start_time,
            load_reduction_end_time
    ):
        """Define additional constraints for load reduction problem."""

        # Remove existing load reduction constraints, if any.
        if self.problem.find_component('load_reduction_constraints') is not None:
            self.problem.del_component('load_reduction_constraints')
            self.problem.del_component('load_reduction_constraints_index')

        # Define load reduction constraints.
        self.problem.load_reduction_constraints = pyo.ConstraintList()
        for timestep in self.building.timesteps:
            if (
                (timestep >= load_reduction_start_time)
                and (timestep < load_reduction_end_time)
            ):
                self.problem.load_reduction_constraints.add(
                    self.problem.variable_output_vector[timestep, 'grid_electric_power']
                    ==
                    (
                        (1.0 - (self.problem.variable_load_reduction[0] / 100.0))
                        * output_vector_reference.at[timestep, 'grid_electric_power']
                    )
                )

    def solve(self):
        """Solve the optimization and return the optimal solution results, i.e., control vector timeseries,
        state vector timeseries, output vector timeseries, operation cost and storage size.

        - Invokes the optimization solver defined in `config.yml` or `config_default.yml` on the pyomo problem object.
        - If problem type is `load_reduction`, the returned investment cost value is the load reduction value.
        - If problem type is not `storage_planning` or `storage_planning_baseline`, storage size is returned as None.

        Returns:
            typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.float, np.float, np.float]: Tuple of
            control vector timeseries, state vector timeseries, output vector timeseries, operation cost,
            and storage size.
        """

        # Solve problem.
        time_start = time.time()
        self.result = self.solver.solve(
            self.problem,
            tee=cobmo.config.config['optimization']['show_solver_output']  # If True, activate verbose solver output.
        )

        # Print solve time for debugging.
        logger.debug("OptimizationProblem solve time: {:.2f} seconds".format(time.time() - time_start))

        # Retrieve results.
        time_start = time.time()
        control_vector = pd.DataFrame(
            0.0,
            self.building.timesteps,
            self.building.controls
        )
        state_vector = pd.DataFrame(
            0.0,
            self.building.timesteps,
            self.building.states
        )
        output_vector = pd.DataFrame(
            0.0,
            self.building.timesteps,
            self.building.outputs
        )
        for timestep in self.building.timesteps:
            for control in self.building.controls:
                control_vector.at[timestep, control] = (
                    self.problem.variable_control_vector[timestep, control].value
                )
            for state in self.building.states:
                state_vector.at[timestep, state] = (
                    self.problem.variable_state_vector[timestep, state].value
                )
            for output in self.building.outputs:
                output_vector.at[timestep, output] = (
                    self.problem.variable_output_vector[timestep, output].value
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

        # Retrieve storage size.
        if self.problem_type == 'storage_planning':
            storage_capacity = self.problem.variable_storage_capacity[0].value
        elif self.problem_type == 'storage_planning_baseline':
            storage_capacity = self.problem.variable_storage_capacity[0]
        else:
            storage_capacity = None

        # Print results compilation time for debugging.
        logger.debug("OptimizationProblem results compilation time: {:.2f} seconds".format(time.time() - time_start))

        return (
            control_vector,
            state_vector,
            output_vector,
            operation_cost,
            investment_cost,
            storage_capacity
        )
