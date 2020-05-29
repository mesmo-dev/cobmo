"""Building model module."""

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.interpolate

import cobmo.config
import cobmo.database_interface
import cobmo.utils

logger = cobmo.config.get_logger(__name__)


class BuildingModel(object):
    """Building model object consisting of the state space model for the given scenario. The model includes
    index sets for states / controls / disturbances / outputs, the state / control / disturbance / state-output /
    control-output / disturbance-output matrices and disturbance / electricity price / output constraint timeseries.

    - The building model object constructs the state space model matrices and index sets
      according to the building model equations which are documented CoBMo's technical documentation.
    - The required `building_data` object for the given scenario is obtained from the database
      through `cobmo.database_interface`.
    - The building can be connected to the electric grid, the thermal grid or both, which is controlled through the
      keyword arguments `connect_electric_grid` / `connect_thermal_grid_cooling` / `connect_thermal_grid_heating`
      as explained below.

    Syntax
        ``BuildingModel(scenario_name)``: Instantiate building model for given `scenario_name`.

    Arguments:
        scenario_name (str): CoBMo building scenario name, as defined in the data table `scenarios`.

    Keyword Arguments:
        timestep_start (pd.Timestamp): If provided, will used in place of `timestep_start` in the scenario definition.
        timestep_end (pd.Timestamp): If provided, will used in place of `timestep_end` in the scenario definition.
        timestep_delta (pd.Timedelta): If provided, will used in place of `timestep_delta` in the scenario definition.
        connect_electric_grid (bool): If true, the output variable `grid_electric_power` will be defined to express the
            total electric power demand at the electric grid connection point. Additionally, the control variables
            `plant_thermal_power_cooling` / `plant_thermal_power_heating` will be defined to enable controlling how much
            of the thermal demand is supplied through a local cooling / heating plant, hence translating the
            thermal demand into electric demand.
        connect_thermal_grid_cooling (bool): If true, the output variable `grid_thermal_power_cooling` will be defined
            to express the thermal power cooling demand at the thermal cooling grid (district cooling system)
            connection point. Additionally, the control variable `grid_thermal_power_cooling` will be defined to
            allow controlling how much of the thermal demand is supplied through the thermal grid connection (as
            opposed to supplying it through a local cooling plant.
        connect_thermal_grid_heating (bool): If true, the output variable `grid_thermal_power_cooling` will be defined
            to express the thermal power heating demand at the thermal heating grid (district heating system)
            connection point. Additionally, the control variable `grid_thermal_power_heating` will be defined to
            allow controlling how much of the thermal demand is supplied through the thermal grid connection (as
            opposed to supplying it through a local heating plant.
        with_validation_outputs (bool): If true, additional validation output variables for the surface temperature,
            surface exterior irradiation heat transfer and surface interior convection heat transfer will be defined.

    Attributes:
        scenario_name (str): CoBMo building scenario name.
        building_data (cobmo.database_interface.BuildingData): CoBMo scenario data object.
        states (pd.Index): Index set of the state variables.
        controls (pd.Index): Index set of the control variables.
        disturbances (pd.Index): Index set of the disturbance variables.
        outputs (pd.Index): Index set of the output variables.
        timesteps (pd.Index): Index set of the timesteps.
        timestep_delta (pd.Timedelta): Timestep interval, assuming a constant interval between all timesteps.
        state_matrix (pd.DataFrame): State matrix.
        control_matrix (pd.DataFrame): Control matrix.
        disturbance_matrix (pd.DataFrame): Disturbance matrix.
        state_output_matrix (pd.DataFrame): State output matrix.
        control_output_matrix (pd.DataFrame): Control output matrix.
        disturbance_output_matrix (pd.DataFrame): Disturbance output matrix.
        state_vector_initial (pd.Series): Initial state vector, describing the state variable values at
            the first timestep.
        disturbance_timeseries (pd.DataFrame): Disturbance timeseries.
        electricity_price_timeseries (pd.DataFrame): Electricity price timeseries.
        electricity_price_distribution_timeseries (pd.DataFrame): Electricity price value distribution timeseries.
        output_constraint_timeseries_maximum (pd.DataFrame): Maximum output constraint timeseries.
        output_constraint_timeseries_minimum (pd.DataFrame): Minimum output constraint timeseries.
    """

    scenario_name: str
    building_data: cobmo.database_interface.BuildingData
    states: pd.Index
    controls: pd.Index
    disturbances: pd.Index
    outputs: pd.Index
    timesteps: pd.Index
    timestep_delta: pd.Timedelta
    state_matrix: pd.DataFrame
    control_matrix: pd.DataFrame
    disturbance_matrix: pd.DataFrame
    state_output_matrix: pd.DataFrame
    control_output_matrix: pd.DataFrame
    disturbance_output_matrix: pd.DataFrame
    state_vector_initial: pd.Series
    disturbance_timeseries: pd.DataFrame
    electricity_price_timeseries: pd.DataFrame
    electricity_price_distribution_timeseries: pd.DataFrame
    output_constraint_timeseries_maximum: pd.DataFrame
    output_constraint_timeseries_minimum: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            timestep_start=None,
            timestep_end=None,
            timestep_delta=None,
            connect_electric_grid=True,
            connect_thermal_grid_cooling=False,
            connect_thermal_grid_heating=False,
            with_validation_outputs=False
    ):

        # Store scenario name.
        self.scenario_name = scenario_name

        # Obtain building data.
        self.building_data = (
            cobmo.database_interface.BuildingData(
                self.scenario_name,
                timestep_start=timestep_start,
                timestep_end=timestep_end,
                timestep_delta=timestep_delta
            )
        )

        # Add constant timeseries in disturbance vector, if any CO2 model or HVAC or window.
        self.define_constant = (
            pd.notnull(self.building_data.scenarios['co2_model_type'])
            | pd.notnull(self.building_data.zones['hvac_ahu_type']).any()
            | pd.notnull(self.building_data.zones['window_type']).any()
        )

        # Define sets.

        # State variables.
        self.states = pd.Index(
            pd.concat([
                # Zone temperature.
                self.building_data.zones['zone_name'] + '_temperature',

                # Surface temperature.
                self.building_data.surfaces_adiabatic['surface_name'][
                    self.building_data.surfaces_adiabatic['heat_capacity'] != 0.0
                ] + '_temperature',
                self.building_data.surfaces_exterior['surface_name'][
                    self.building_data.surfaces_exterior['heat_capacity'] != 0.0
                ] + '_temperature',
                self.building_data.surfaces_interior['surface_name'][
                    self.building_data.surfaces_interior['heat_capacity'] != 0.0
                ] + '_temperature',

                # Zone CO2 concentration.
                self.building_data.zones['zone_name'][
                    (
                        pd.notnull(self.building_data.zones['hvac_ahu_type'])
                        | pd.notnull(self.building_data.zones['window_type'])
                    )
                    & pd.notnull(self.building_data.scenarios['co2_model_type'])
                ] + '_co2_concentration',

                # Zone absolute humidity.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & pd.notnull(self.building_data.scenarios['humidity_model_type'])
                ] + '_absolute_humidity',

                # Radiator temperatures.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                ] + '_radiator_water_mean_temperature',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                ] + '_radiator_hull_front_temperature',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                ] + '_radiator_hull_rear_temperature',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                    & (self.building_data.zones['radiator_panel_number'] == '2')
                ] + '_radiator_panel_1_hull_rear_temperature',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                    & (self.building_data.zones['radiator_panel_number'] == '2')
                ] + '_radiator_panel_2_hull_front_temperature',

                # Sensible storage state of charge.
                pd.Series(['sensible_thermal_storage_state_of_charge']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_sensible_thermal_storage'
                ) else None,

                # Battery storage state of charge.
                pd.Series(['battery_storage_state_of_charge']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_battery_storage'
                ) else None
            ]),
            name='state_name'
        )

        # Control variables.
        self.controls = pd.Index(
            pd.concat([
                # Generic HVAC system.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_generic_type'])
                ] + '_generic_heat_thermal_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_generic_type'])
                ] + '_generic_cool_thermal_power',

                # Radiator thermal power.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                ] + '_radiator_thermal_power',

                # AHU.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_air_flow',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_air_flow',

                # TU.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_heat_air_flow',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_cool_air_flow',

                # Windows.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['window_type'])
                ] + '_window_air_flow',

                # Sensible storage charge.
                # TODO: Add consideration for sensible storage heating / cooling.
                pd.Series(['sensible_storage_charge_cool_thermal_power']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_sensible_thermal_storage'
                ) else None,

                # Battery storage charge.
                pd.Series(['battery_storage_charge_electric_power']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_battery_storage'
                ) else None,

                # Sensible storage discharge to AHU.
                # TODO: Add consideration for sensible storage heating / cooling.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_sensible_thermal_storage'
                    )
                ] + '_sensible_storage_to_zone_ahu_cool_thermal_power',

                # Sensible storage discharge to TU.
                # TODO: Add consideration for sensible storage heating / cooling.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_sensible_thermal_storage'
                    )
                ] + '_sensible_storage_to_zone_tu_cool_thermal_power',

                # Battery storage discharge to AHU.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_ahu_heat_electric_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_ahu_cool_electric_power',

                # Battery storage discharge to TU.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_tu_heat_electric_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_tu_cool_electric_power',

                # Electric / thermal grid connection.
                pd.Series([
                    'plant_thermal_power_cooling',
                    'plant_thermal_power_heating'
                ]) if connect_electric_grid else None,
                pd.Series(['grid_thermal_power_cooling']) if connect_thermal_grid_cooling else None,
                pd.Series(['grid_thermal_power_heating']) if connect_thermal_grid_heating else None
            ]),
            name='control_name'
        )

        # Disturbance variables.
        self.disturbances = pd.Index(
            pd.concat([
                # Weather.
                pd.Series([
                    'ambient_air_temperature',
                    'sky_temperature',
                    'irradiation_horizontal',
                    'irradiation_east',
                    'irradiation_south',
                    'irradiation_west',
                    'irradiation_north'
                ]),

                # Internal gains.
                pd.Series(
                    self.building_data.zones['internal_gain_type'].dropna().unique() + '_internal_gain_occupancy'
                ),
                pd.Series(
                    self.building_data.zones['internal_gain_type'].dropna().unique() + '_internal_gain_appliances'
                ),

                # Constant (workaround for constant model terms).
                (pd.Series(['constant']) if self.define_constant else None)
            ]),
            name='disturbance_name'
        )

        # Output variables.
        self.outputs = pd.Index(
            pd.concat([
                # Zone temperature.
                self.building_data.zones['zone_name'] + '_temperature',

                # Zone CO2 concentration.
                self.building_data.zones['zone_name'][
                    (
                        pd.notnull(self.building_data.zones['hvac_ahu_type'])
                        | pd.notnull(self.building_data.zones['window_type'])
                    )
                    & pd.notnull(self.building_data.scenarios['co2_model_type'])
                ] + '_co2_concentration',

                # Zone absolute humidity.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & pd.notnull(self.building_data.scenarios['humidity_model_type'])
                ] + '_absolute_humidity',

                # Zone fresh air flow.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    | pd.notnull(self.building_data.zones['window_type'])
                ] + '_total_fresh_air_flow',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_fresh_air_flow',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['window_type'])
                ] + '_window_fresh_air_flow',

                # Generic HVAC system.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_generic_type'])
                ] + '_generic_cool_thermal_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_generic_type'])
                ] + '_generic_cool_electric_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_generic_type'])
                ] + '_generic_heat_thermal_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_generic_type'])
                ] + '_generic_heat_electric_power_heating',

                # Radiator thermal power.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                ] + '_radiator_thermal_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                ] + '_radiator_electric_power_heating',

                # AHU electric power.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_thermal_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_thermal_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_electric_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_electric_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_thermal_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_thermal_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_electric_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_electric_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_electric_power_fan',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_electric_power_fan',

                # TU electric power.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_cool_thermal_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_cool_electric_power_cooling',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_heat_thermal_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_heat_electric_power_heating',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_heat_electric_power_fan',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                ] + '_tu_cool_electric_power_fan',

                # Sensible storage state of charge.
                pd.Series(['sensible_thermal_storage_state_of_charge']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_sensible_thermal_storage'
                ) else None,

                # Battery storage state of charge.
                pd.Series(['battery_storage_state_of_charge']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_battery_storage'
                ) else None,

                # Sensible storage discharge.
                # TODO: Add consideration for sensible storage heating / cooling.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_sensible_thermal_storage'
                    )
                ] + '_sensible_storage_to_zone_tu_heat_thermal_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_sensible_thermal_storage'
                    )
                ] + '_sensible_storage_to_zone_ahu_cool_thermal_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_sensible_thermal_storage'
                    )
                ] + '_sensible_storage_to_zone_tu_cool_thermal_power',

                # Battery storage discharge.
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_ahu_heat_electric_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_tu_heat_electric_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_ahu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_ahu_cool_electric_power',
                self.building_data.zones['zone_name'][
                    pd.notnull(self.building_data.zones['hvac_tu_type'])
                    & (
                        self.building_data.scenarios['building_storage_type']
                        == 'default_battery_storage'
                    )
                ] + '_battery_storage_to_zone_tu_cool_electric_power',

                # Sensible storage charge.
                # TODO: Add consideration for sensible storage heating / cooling.
                pd.Series([
                    'sensible_storage_charge_cool_thermal_power',
                    'sensible_storage_charge_cool_electric_power'
                ]) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_sensible_thermal_storage'
                ) else None,

                # Battery storage charge.
                pd.Series(['battery_storage_charge_electric_power']) if (
                    self.building_data.scenarios['building_storage_type']
                    == 'default_battery_storage'
                ) else None,

                # Validation outputs.
                pd.concat([
                    self.building_data.surfaces_adiabatic['surface_name'][
                        self.building_data.surfaces_adiabatic['heat_capacity'] != 0.0
                    ] + '_temperature',
                    self.building_data.surfaces_exterior['surface_name'][
                        self.building_data.surfaces_exterior['heat_capacity'] != 0.0
                    ] + '_temperature',
                    self.building_data.surfaces_interior['surface_name'][
                        self.building_data.surfaces_interior['heat_capacity'] != 0.0
                    ] + '_temperature',
                    self.building_data.surfaces_exterior['surface_name'] + '_irradiation_gain_exterior',
                    self.building_data.surfaces_exterior['surface_name'] + '_convection_interior'
                ]) if with_validation_outputs else None,

                # Electric / thermal grid connection.
                pd.Series(['grid_thermal_power_cooling_balance']) if any([
                    connect_electric_grid,
                    connect_thermal_grid_cooling
                ]) else None,
                pd.Series(['grid_thermal_power_heating_balance']) if any([
                    connect_electric_grid,
                    connect_thermal_grid_heating
                ]) else None,
                pd.Series([
                    'plant_thermal_power_cooling',
                    'plant_thermal_power_heating',
                    'grid_electric_power'
                ]) if connect_electric_grid else None,
                pd.Series(['grid_thermal_power_cooling']) if connect_thermal_grid_cooling else None,
                pd.Series(['grid_thermal_power_heating']) if connect_thermal_grid_heating else None
            ]),
            name='output_name'
        )

        # Obtain timesteps.
        self.timesteps = self.building_data.timesteps
        self.timestep_delta = self.building_data.timestep_delta

        # Instantiate state space model matrices.
        self.state_matrix = pd.DataFrame(
            0.0,
            self.states,
            self.states
        )
        self.control_matrix = pd.DataFrame(
            0.0,
            self.states,
            self.controls
        )
        self.disturbance_matrix = pd.DataFrame(
            0.0,
            self.states,
            self.disturbances
        )
        self.state_output_matrix = pd.DataFrame(
            0.0,
            self.outputs,
            self.states
        )
        self.control_output_matrix = pd.DataFrame(
            0.0,
            self.outputs,
            self.controls
        )
        self.disturbance_output_matrix = pd.DataFrame(
            0.0,
            self.outputs,
            self.disturbances
        )

        def define_initial_state():
            """Define initial value of the state vector for given definition in `initial_state_types`."""

            # Instantiate.
            self.state_vector_initial = (
                pd.Series(
                    0.0,
                    index=self.states
                )
            )

            # Zone air temperature.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(self.building_data.zones['zone_name'] + '_temperature')
            ] = (
                self.building_data.scenarios['initial_zone_temperature']
            )

            # Surface temperature.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(
                    pd.concat([
                        self.building_data.surfaces_adiabatic['surface_name'] + '_temperature',
                        self.building_data.surfaces_exterior['surface_name'] + '_temperature',
                        self.building_data.surfaces_interior['surface_name'] + '_temperature'
                    ])
                )
            ] = (
                self.building_data.scenarios['initial_surface_temperature']
            )

            # CO2 concentration.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(self.building_data.zones['zone_name'] + '_co2_concentration')
            ] = (
                self.building_data.scenarios['initial_co2_concentration']
            )

            # Zone air absolute humidity.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(self.building_data.zones['zone_name'] + '_absolute_humidity')
            ] = (
                self.building_data.scenarios['initial_absolute_humidity']
            )

            # Sensible storage state of charge.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.str.contains('_sensible_thermal_storage_state_of_charge')
            ] = (
                self.building_data.scenarios['initial_sensible_thermal_storage_state_of_charge']
            )

            # Battery storage state of charge.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.str.contains('_battery_storage_state_of_charge')
            ] = (
                self.building_data.scenarios['initial_battery_storage_state_of_charge']
            )

        def calculate_coefficients_zone():
            """Calculate zone parameters / heat transfer coefficients for use in, e.g., surface and radiator models."""

            # Calculate absolute heat capacity from specific heat capacity.
            self.building_data.zones['heat_capacity'] = (
                self.building_data.zones['zone_area']
                * self.building_data.zones['zone_height']
                * self.building_data.zones['heat_capacity']
            )

            # Instantiate columns for parameters / heat transfer coefficients.
            self.building_data.zones['zone_surfaces_wall_area'] = None
            self.building_data.zones['zone_surfaces_window_area'] = None
            self.building_data.zones['zone_surfaces_wall_emissivity'] = None
            self.building_data.zones['zone_surfaces_window_emissivity'] = None

            # Calculate zone parameters / heat transfer coefficients.
            for zone_name, zone_data in self.building_data.zones.iterrows():
                # Collect all surfaces adjacent to the zone.
                zone_surfaces = (
                    pd.concat(
                        [
                            self.building_data.surfaces_exterior.loc[
                                self.building_data.surfaces_exterior['zone_name'].isin([zone_name]),
                                :
                            ],
                            self.building_data.surfaces_interior.loc[
                                self.building_data.surfaces_interior['zone_name'].isin([zone_name]),
                                :
                            ],
                            self.building_data.surfaces_interior.loc[
                                self.building_data.surfaces_interior['zone_adjacent_name'].isin([zone_name]),
                                :
                            ],
                            self.building_data.surfaces_adiabatic.loc[
                                self.building_data.surfaces_adiabatic['zone_name'].isin([zone_name]),
                                :
                            ]
                        ],
                        sort=False
                    )
                )

                # Calculate parameters / heat transfer coefficients.
                self.building_data.zones.at[zone_name, 'zone_surfaces_wall_area'] = (
                    (
                        zone_surfaces['surface_area']
                        * (1 - zone_surfaces['window_wall_ratio'])
                    ).sum()
                )
                self.building_data.zones.at[zone_name, 'zone_surfaces_window_area'] = (
                    (
                        zone_surfaces['surface_area']
                        * zone_surfaces['window_wall_ratio']
                    ).sum()
                )
                self.building_data.zones.at[zone_name, 'zone_surfaces_wall_emissivity'] = (
                    zone_surfaces['emissivity'].mean()
                )
                # TODO: Ignore surfaces with no windows.
                self.building_data.zones.at[zone_name, 'zone_surfaces_window_emissivity'] = (
                    zone_surfaces['emissivity_window'].mean()
                )

        def calculate_coefficients_surface():
            """Calculate heat transfer coefficients for the surface models."""

            # Calculate absolute heat capacity from specific heat capacity.
            self.building_data.surfaces_adiabatic['heat_capacity'] = (
                self.building_data.surfaces_adiabatic['surface_area']
                * self.building_data.surfaces_adiabatic['heat_capacity']
            )
            self.building_data.surfaces_exterior['heat_capacity'] = (
                self.building_data.surfaces_exterior['surface_area']
                * self.building_data.surfaces_exterior['heat_capacity']
            )
            self.building_data.surfaces_interior['heat_capacity'] = (
                self.building_data.surfaces_interior['surface_area']
                * self.building_data.surfaces_interior['heat_capacity']
            )

            # Instantiate columns for heat transfer coefficients.
            self.building_data.surfaces_exterior['heat_transfer_coefficient_surface_sky'] = None
            self.building_data.surfaces_exterior['heat_transfer_coefficient_surface_ground'] = None
            self.building_data.surfaces_exterior['heat_transfer_coefficient_window_sky'] = None
            self.building_data.surfaces_exterior['heat_transfer_coefficient_window_ground'] = None

            # Calculate heat transfer coefficients.
            for surface_name, surface_data in self.building_data.surfaces_exterior.iterrows():
                surface_data['heat_transfer_coefficient_surface_sky'] = (
                    4.0
                    * self.building_data.parameters['stefan_boltzmann_constant']
                    * surface_data['emissivity']
                    * surface_data['sky_view_factor']
                    * (
                        self.building_data.scenarios['linearization_exterior_surface_temperature']
                        / 2.0
                        + self.building_data.scenarios['linearization_sky_temperature']
                        / 2.0
                        + 273.15
                    ) ** 3
                )
                surface_data['heat_transfer_coefficient_surface_ground'] = (
                    4.0
                    * self.building_data.parameters['stefan_boltzmann_constant']
                    * surface_data['emissivity']
                    * (1.0 - surface_data['sky_view_factor'])
                    * (
                        self.building_data.scenarios['linearization_exterior_surface_temperature']
                        / 2.0
                        + self.building_data.scenarios['linearization_ambient_air_temperature']
                        / 2.0
                        + 273.15
                    ) ** 3
                )
                if pd.notnull(surface_data['window_type']):
                    surface_data['heat_transfer_coefficient_window_sky'] = (
                        4.0
                        * self.building_data.parameters['stefan_boltzmann_constant']
                        * surface_data['emissivity_window']
                        * surface_data['sky_view_factor']
                        * (
                            self.building_data.scenarios['linearization_exterior_surface_temperature']
                            / 2.0
                            + self.building_data.scenarios['linearization_sky_temperature']
                            / 2.0
                            + 273.15
                        ) ** 3
                    )
                    surface_data['heat_transfer_coefficient_window_ground'] = (
                        4.0
                        * self.building_data.parameters['stefan_boltzmann_constant']
                        * surface_data['emissivity_window']
                        * (1.0 - surface_data['sky_view_factor'])
                        * (
                            self.building_data.scenarios['linearization_exterior_surface_temperature']
                            / 2.0
                            + self.building_data.scenarios['linearization_ambient_air_temperature']
                            / 2.0
                            + 273.15
                        ) ** 3
                    )

        def calculate_coefficients_radiator():
            """Calculate heat transfer coefficients for the radiator model."""

            if pd.notnull(self.building_data.zones['hvac_radiator_type']).any():
                # Instantiate columns for heat transfer coefficients.
                self.building_data.zones['heat_capacitance_hull'] = None
                self.building_data.zones['thermal_resistance_radiator_hull_conduction'] = None
                self.building_data.zones['thermal_resistance_radiator_front_zone'] = None
                self.building_data.zones['thermal_resistance_radiator_front_surfaces'] = None
                self.building_data.zones['thermal_resistance_radiator_front_zone_surfaces'] = None
                self.building_data.zones['thermal_resistance_radiator_rear_zone'] = None
                self.building_data.zones['thermal_resistance_radiator_rear_surfaces'] = None
                self.building_data.zones['thermal_resistance_radiator_rear_zone_surfaces'] = None

                # Instantiate additional columns for multi-panel radiators.
                if (self.building_data.zones['radiator_panel_number'] == '2').any():
                    self.building_data.zones['thermal_resistance_radiator_panel_1_rear_zone'] = None
                    self.building_data.zones['thermal_resistance_radiator_panel_2_front_zone'] = None

                # Calculate heat transfer coefficients.
                for zone_name, zone_data in self.building_data.zones.iterrows():
                    if pd.notnull(zone_data['hvac_radiator_type']):
                        # Calculate geometric parameters and heat capacity.
                        thickness_water_layer = (
                            zone_data['radiator_water_volume']
                            / zone_data['radiator_panel_area']
                        )
                        thickness_hull_layer = (
                            # Thickness for hull on one side of the panel.
                            0.5 * (
                                zone_data['radiator_panel_thickness']
                                - thickness_water_layer
                            )
                        )
                        radiator_hull_volume = (
                            # Volume for hull on one side of the panel.
                            thickness_hull_layer
                            * zone_data['radiator_panel_area']
                        )
                        self.building_data.zones.at[zone_name, 'heat_capacitance_hull'] = (
                            radiator_hull_volume
                            * zone_data['radiator_hull_heat_capacity']
                        )
                        self.building_data.zones.at[zone_name, 'heat_capacitance_water'] = (
                            zone_data['radiator_water_volume']
                            * self.building_data.parameters['water_specific_heat']
                        )

                        # Calculate fundamental thermal resistances.
                        thermal_resistance_conduction = (
                            thickness_hull_layer
                            / (
                                zone_data['radiator_hull_conductivity']
                                * zone_data['radiator_panel_area']
                            )
                        )
                        thermal_resistance_convection = (
                            1.0
                            / (
                                zone_data['radiator_convection_coefficient']
                                * zone_data['radiator_panel_area']
                            )
                        )
                        temperature_radiator_surfaces_mean = (
                            0.5
                            * (
                                0.5
                                * (
                                    zone_data['radiator_supply_temperature_nominal']
                                    + zone_data['radiator_return_temperature_nominal']
                                )
                                + self.building_data.scenarios['linearization_surface_temperature']
                            )
                        )
                        thermal_resistance_radiation_front = (
                            (
                                (1.0 / zone_data['radiator_panel_area'])
                                + (
                                    (1.0 - zone_data['radiator_emissivity'])
                                    / (
                                        zone_data['radiator_panel_area']
                                        * zone_data['radiator_emissivity']
                                    )
                                )
                                + (
                                    # TODO: Use total zone surface area and emissivity?
                                    (1.0 - zone_data['zone_surfaces_wall_emissivity'])
                                    / (
                                        zone_data['zone_surfaces_wall_area']
                                        * zone_data['zone_surfaces_wall_emissivity']
                                    )
                                )
                            )
                            / (
                                (
                                    4.0 * self.building_data.parameters['stefan_boltzmann_constant']
                                    * (temperature_radiator_surfaces_mean ** 3.0)
                                )
                            )
                        )
                        thermal_resistance_radiation_rear = (
                            (
                                (1.0 / zone_data['radiator_panel_area'])
                                + (
                                    (1.0 - zone_data['radiator_emissivity'])
                                    / (
                                        zone_data['radiator_panel_area']
                                        * zone_data['radiator_emissivity']
                                    )
                                )
                                + (
                                    # TODO: Use total zone surface area and emissivity?
                                    (1.0 - zone_data['zone_surfaces_wall_emissivity'])
                                    / (
                                        zone_data['radiator_panel_area']
                                        * zone_data['zone_surfaces_wall_emissivity']
                                    )
                                )
                            )
                            / (
                                (
                                    4.0 * self.building_data.parameters['stefan_boltzmann_constant']
                                    * (temperature_radiator_surfaces_mean ** 3.0)
                                )
                            )
                        )
                        thermal_resistance_star_sum_front = (
                            0.5 * thermal_resistance_conduction * thermal_resistance_convection
                            + 0.5 * thermal_resistance_conduction * thermal_resistance_radiation_front
                            + thermal_resistance_convection * thermal_resistance_radiation_front
                        )
                        thermal_resistance_star_sum_rear = (
                            0.5 * thermal_resistance_conduction * thermal_resistance_convection
                            + 0.5 * thermal_resistance_conduction * thermal_resistance_radiation_rear
                            + thermal_resistance_convection * thermal_resistance_radiation_rear
                        )

                        # Calculate transformed thermal resistances.
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_hull_conduction'] = (
                            thermal_resistance_conduction
                        )
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_front_zone'] = (
                            thermal_resistance_star_sum_front / thermal_resistance_radiation_front
                        )
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_front_surfaces'] = (
                            thermal_resistance_star_sum_front / thermal_resistance_convection
                        )
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_front_zone_surfaces'] = (
                            thermal_resistance_star_sum_front / (0.5 * thermal_resistance_conduction)
                        )
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_rear_zone'] = (
                            thermal_resistance_star_sum_rear / thermal_resistance_radiation_rear
                        )
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_rear_surfaces'] = (
                            thermal_resistance_star_sum_rear / thermal_resistance_convection
                        )
                        self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_rear_zone_surfaces'] = (
                            thermal_resistance_star_sum_rear / (0.5 * thermal_resistance_conduction)
                        )

                        if (self.building_data.zones['radiator_panel_number'] == '2').any():
                            thermal_resistance_convection_fin = (
                                1.0
                                / (
                                    thermal_resistance_convection
                                    * zone_data['radiator_fin_effectiveness']
                                )
                            )

                            self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_panel_1_rear_zone'] = (
                                0.5 * thermal_resistance_conduction
                                + thermal_resistance_convection
                            )
                            self.building_data.zones.at[zone_name, 'thermal_resistance_radiator_panel_2_front_zone'] = (
                                0.5 * thermal_resistance_conduction
                                + thermal_resistance_convection_fin
                            )

        def define_heat_transfer_surfaces_exterior():
            """Thermal model: Exterior surfaces"""
            # TODO: Rename thermal_resistance_surface
            # TODO: Exterior window transmission factor

            for surface_name, surface_data in self.building_data.surfaces_exterior.iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    # Conductive heat transfer from the exterior towards the core of surface
                    self.disturbance_matrix.at[
                        surface_name + '_temperature',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        surface_name + '_temperature',
                        'ambient_air_temperature'
                    ] += (
                        (
                            self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_surface_ground']
                        )
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        surface_name + '_temperature',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_surface_sky']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )

                    # Conductive heat transfer from the interior towards the core of surface
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / self.building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )

                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / self.building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1))
                            / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )
                    self.state_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                else:  # Surfaces with neglected heat capacity
                    # Complete convective heat transfer from surface to zone
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'ambient_air_temperature'
                    ] += (
                        (
                            self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_surface_ground']
                        )
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_surface_sky']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            + 1.0
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + 1.0
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / self.building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / (surface_data['thermal_resistance_surface'] ** (- 1))
                            ) ** (- 1))
                            / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )

                # Windows for each exterior surface - Modelled as surfaces with neglected heat capacity
                if surface_data['window_wall_ratio'] != 0.0:
                    # Complete convective heat transfer from surface to zone
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_window']
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / (surface_data['thermal_resistance_window'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'ambient_air_temperature'
                    ] += (
                        (
                            self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_window_ground']
                        )
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / (surface_data['thermal_resistance_window'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_window_sky']
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / (surface_data['thermal_resistance_window'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            + 1.0
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + 1.0
                            / (surface_data['thermal_resistance_window'] ** (- 1))
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / self.building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_window']
                            * surface_data['surface_area']
                            * surface_data['window_wall_ratio']
                            * (1.0 - (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_window_ground']
                                    + surface_data['heat_transfer_coefficient_window_sky']
                                )
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_window_ground']
                                    + surface_data['heat_transfer_coefficient_window_sky']
                                )
                                / (surface_data['thermal_resistance_window'] ** (- 1))
                            ) ** (- 1))
                            / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )

        def define_heat_transfer_surfaces_interior():
            """Thermal model: Interior surfaces"""

            for surface_name, surface_data in self.building_data.surfaces_interior.iterrows():
                for zone_name in [surface_data['zone_name'], surface_data['zone_adjacent_name']]:
                    if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                        # Conductive heat transfer from the interior towards the core of surface
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                self.building_data.surfaces_exterior[
                                    self.building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            self.disturbance_matrix.at[
                                surface_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / self.building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (
                                    1.0
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / (
                                        2.0
                                        * surface_data['thermal_resistance_surface'] ** (- 1)
                                    )
                                ) ** (- 1)
                                / surface_data['heat_capacity']
                            )
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_name + '_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            zone_name + '_temperature'
                        ] += (
                            surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )

                        # Convective heat transfer from the surface towards zone
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                self.building_data.surfaces_exterior[
                                    self.building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            self.disturbance_matrix.at[
                                zone_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / self.building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (1.0 - (
                                    1.0
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / (
                                        2.0
                                        * surface_data['thermal_resistance_surface'] ** (- 1)
                                    )
                                ) ** (- 1))
                                / self.building_data.zones.at[zone_name, 'heat_capacity']
                            )
                        self.state_matrix.at[
                            zone_name + '_temperature',
                            surface_name + '_temperature'
                        ] += (
                            surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1)
                            / self.building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        self.state_matrix.at[
                            zone_name + '_temperature',
                            zone_name + '_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1)
                            / self.building_data.zones.at[zone_name, 'heat_capacity']
                        )
                    else:  # Surfaces with neglected heat capacity
                        # Get adjacent / opposite zone_name
                        if zone_name == surface_data['zone_name']:
                            zone_adjacent_name = surface_data['zone_adjacent_name']
                        else:
                            zone_adjacent_name = surface_data['zone_name']

                        # Total adjacent zone surface area for calculating share of interior (indirect) irradiation.
                        zone_adjacent_surface_area = sum(
                            zone_surface_data['surface_area']
                            * (1 - zone_surface_data['window_wall_ratio'])
                            for zone_surface_name, zone_surface_data in pd.concat(
                                [
                                    self.building_data.surfaces_exterior[
                                        self.building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                    ],
                                    self.building_data.surfaces_interior[
                                        self.building_data.surfaces_interior['zone_name'] == zone_adjacent_name
                                    ],
                                    self.building_data.surfaces_interior[
                                        self.building_data.surfaces_interior['zone_adjacent_name'] == zone_adjacent_name
                                    ],
                                    self.building_data.surfaces_adiabatic[
                                        self.building_data.surfaces_adiabatic['zone_name'] == zone_adjacent_name
                                    ]
                                ],
                                sort=False
                            ).iterrows()  # For all surfaces adjacent to the zone
                        )

                        # Complete convective heat transfer from adjacent zone to zone
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                self.building_data.surfaces_exterior[
                                    self.building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            self.disturbance_matrix.at[
                                zone_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / zone_adjacent_surface_area
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (
                                    1.0
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['thermal_resistance_surface'] ** (- 1)
                                ) ** (- 1)
                                / self.building_data.zones.at[zone_name, 'heat_capacity']
                            )
                        self.state_matrix.at[
                            zone_name + '_temperature',
                            zone_adjacent_name + '_temperature'
                        ] += (
                            surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['thermal_resistance_surface'] ** (- 1)
                            ) ** (- 1)
                            / self.building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        self.state_matrix.at[
                            zone_name + '_temperature',
                            zone_name + '_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['thermal_resistance_surface'] ** (- 1)
                            ) ** (- 1)
                            / self.building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                self.building_data.surfaces_exterior[
                                    self.building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            self.disturbance_matrix.at[
                                zone_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / self.building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (1.0 - (
                                    1.0
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['thermal_resistance_surface'] ** (- 1)
                                ) ** (- 1))
                                / self.building_data.zones.at[zone_name, 'heat_capacity']
                            )

                    # Windows for each interior surface - Modelled as surfaces with neglected heat capacity
                    if surface_data['window_wall_ratio'] != 0.0:
                        # Get adjacent / opposite zone_name
                        if zone_name == surface_data['zone_name']:
                            zone_adjacent_name = surface_data['zone_adjacent_name']
                        else:
                            zone_adjacent_name = surface_data['zone_name']

                        # Total adjacent zone surface area for calculating share of interior (indirect) irradiation
                        zone_adjacent_surface_area = sum(
                            zone_surface_data['surface_area']
                            * (1 - zone_surface_data['window_wall_ratio'])
                            for zone_surface_name, zone_surface_data in pd.concat(
                                [
                                    self.building_data.surfaces_exterior[
                                        self.building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                    ],
                                    self.building_data.surfaces_interior[
                                        self.building_data.surfaces_interior['zone_name'] == zone_adjacent_name
                                    ],
                                    self.building_data.surfaces_interior[
                                        self.building_data.surfaces_interior['zone_adjacent_name'] == zone_adjacent_name
                                    ],
                                    self.building_data.surfaces_adiabatic[
                                        self.building_data.surfaces_adiabatic['zone_name'] == zone_adjacent_name
                                    ]
                                ],
                                sort=False
                            ).iterrows()  # For all surfaces adjacent to the zone
                        )

                        # Complete convective heat transfer from adjacent zone to zone
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                self.building_data.surfaces_exterior[
                                    self.building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            self.disturbance_matrix.at[
                                zone_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / zone_adjacent_surface_area
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_window']
                                * surface_data['surface_area']
                                * surface_data['window_wall_ratio']
                                * (
                                    1.0
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['thermal_resistance_window'] ** (- 1)
                                ) ** (- 1)
                                / self.building_data.zones.at[zone_name, 'heat_capacity']
                            )
                        self.state_matrix.at[
                            zone_name + '_temperature',
                            zone_adjacent_name + '_temperature'
                        ] += (
                            surface_data['surface_area']
                            * surface_data['window_wall_ratio']
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['thermal_resistance_window'] ** (- 1)
                            ) ** (- 1)
                            / self.building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        self.state_matrix.at[
                            zone_name + '_temperature',
                            zone_name + '_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * surface_data['window_wall_ratio']
                            * (
                                1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['thermal_resistance_window'] ** (- 1)
                            ) ** (- 1)
                            / self.building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                self.building_data.surfaces_exterior[
                                    self.building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            self.disturbance_matrix.at[
                                zone_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / self.building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_window']
                                * surface_data['surface_area']
                                * surface_data['window_wall_ratio']
                                * (1.0 - (
                                    1.0
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['thermal_resistance_window'] ** (- 1)
                                ) ** (- 1))
                                / self.building_data.zones.at[zone_name, 'heat_capacity']
                            )

        def define_heat_transfer_surfaces_adiabatic():
            """Thermal model: Adiabatic surfaces"""

            for surface_name, surface_data in self.building_data.surfaces_adiabatic.iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    # Conductive heat transfer from the interior towards the core of surface
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / self.building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )

                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / self.building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['thermal_resistance_surface'] ** (- 1)
                                )
                            ) ** (- 1))
                            / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )
                    self.state_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    self.state_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                        / self.building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                else:  # Surfaces with neglected heat capacity
                    logger.warn(f"Adiabatic surfaces with zero heat capacity have no effect: {surface_name}")

        def define_heat_transfer_infiltration():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                self.state_matrix.at[
                    zone_name + '_temperature',
                    zone_name + '_temperature'
                ] += (
                    - zone_data['infiltration_rate']
                    * self.building_data.parameters['heat_capacity_air']
                    * zone_data['zone_area']
                    * zone_data['zone_height']
                    / zone_data['heat_capacity']
                )
                self.disturbance_matrix.at[
                    zone_name + '_temperature',
                    'ambient_air_temperature'
                ] += (
                    zone_data['infiltration_rate']
                    * self.building_data.parameters['heat_capacity_air']
                    * zone_data['zone_area']
                    * zone_data['zone_height']
                    / zone_data['heat_capacity']
                )

        def define_heat_transfer_window_air_flow():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['window_type']):
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_window_air_flow'
                    ] += (
                        self.building_data.parameters['heat_capacity_air']
                        * (
                            self.building_data.scenarios['linearization_ambient_air_temperature']
                            - self.building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                        / zone_data['heat_capacity']
                    )

        def define_heat_transfer_internal_gains():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data.at['internal_gain_type']):
                    self.disturbance_matrix.at[
                        zone_name + '_temperature',
                        zone_data.at['internal_gain_type'] + '_internal_gain_occupancy'
                    ] += (
                        zone_data.at['internal_gain_occupancy_factor']
                        * zone_data.at['zone_area']
                        / zone_data.at['heat_capacity']
                    )
                    self.disturbance_matrix.at[
                        zone_name + '_temperature',
                        zone_data.at['internal_gain_type'] + '_internal_gain_appliances'
                    ] += (
                        zone_data.at['internal_gain_appliances_factor']
                        * zone_data.at['zone_area']
                        / zone_data.at['heat_capacity']
                    )

        def define_heat_transfer_hvac_generic():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_generic_type']):
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_generic_heat_thermal_power'
                    ] += (
                        1.0
                        / zone_data['heat_capacity']
                    )
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_generic_cool_thermal_power'
                    ] += (
                        - 1.0
                        / zone_data['heat_capacity']
                    )

        def define_heat_transfer_hvac_radiator():
            """Define state equations describing the heat transfer occurring due to radiators."""

            if pd.notnull(self.building_data.zones['hvac_radiator_type']).any():
                zones_index = (
                    pd.notnull(self.building_data.zones['hvac_radiator_type'])
                )
                zones_2_panels_index = (
                    zones_index
                    & (self.building_data.zones['radiator_panel_number'] == '2')
                )

                # Thermal power input to water.
                self.control_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power'
                ] += (
                    1.0
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_water']
                ).values

                # Heat transfer between radiator hull front and water.
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_water']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_water']
                ).values

                if zones_2_panels_index.any():
                    # Heat transfer between radiator panel 1 hull rear and water.
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                    ] += (
                        - 1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                    ] += (
                        1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                    ] += (
                        1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                    ] += (
                        - 1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                    ).values

                    # Heat transfer between radiator panel 2 hull front and water.
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                    ] += (
                        - 1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                    ] += (
                        1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                    ] += (
                        1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                    ] += (
                        - 1.0
                        / (0.5 * self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                    ).values

                # Heat transfer between radiator hull rear and water.
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_water']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_water']
                ).values

                # Heat transfer between radiator hull front and zone air.
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
                ] += (
                    - 1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature'
                ] += (
                    1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
                ] += (
                    1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacity']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature'
                ] += (
                    - 1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacity']
                ).values

                if zones_2_panels_index.any():
                    # Heat transfer between radiator panel 1 hull rear and zone air.
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                    ] += (
                        - 1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                    ] += (
                        1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                    ] += (
                        1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacity']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacity']
                    ).values

                    # Heat transfer between radiator panel 2 hull front and zone air.
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                    ] += (
                        - 1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                    ] += (
                        1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                    ] += (
                        1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacity']
                    ).values
                    self.state_matrix.loc[
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                        self.building_data.zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        / self.building_data.zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                        / self.building_data.zones.loc[zones_2_panels_index, 'heat_capacity']
                    ).values

                # Heat transfer between radiator hull rear and zone air.
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
                ] += (
                    - 1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature'
                ] += (
                    1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
                ] += (
                    1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacity']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_temperature'
                ] += (
                    - 1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                    / self.building_data.zones.loc[zones_index, 'heat_capacity']
                ).values

                # Heat transfer between radiator hull front / rear and zone surfaces.
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
                ] += (
                    - 1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_front_surfaces']
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
                ] += (
                    - 1.0
                    / self.building_data.zones.loc[zones_index, 'thermal_resistance_radiator_rear_surfaces']
                    / self.building_data.zones.loc[zones_index, 'heat_capacitance_hull']
                ).values
                for zone_name, zone_data in self.building_data.zones.loc[zones_index, :].iterrows():
                    for surface_name, surface_data in (
                        pd.concat(
                            [
                                self.building_data.surfaces_exterior.loc[
                                    self.building_data.surfaces_exterior['zone_name'].isin([zone_name]),
                                    :
                                ],
                                self.building_data.surfaces_interior.loc[
                                    self.building_data.surfaces_interior['zone_name'].isin([zone_name]),
                                    :
                                ],
                                self.building_data.surfaces_interior.loc[
                                    self.building_data.surfaces_interior['zone_adjacent_name'].isin([zone_name]),
                                    :
                                ],
                                self.building_data.surfaces_adiabatic.loc[
                                    self.building_data.surfaces_adiabatic['zone_name'].isin([zone_name]),
                                    :
                                ]
                            ],
                            sort=False
                        ).iterrows()  # For all surfaces adjacent to the zone.
                    ):
                        # Front.
                        self.state_matrix.at[
                            zone_name + '_radiator_hull_front_temperature',
                            surface_name + '_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_front_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / zone_data['heat_capacitance_hull']
                        )
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            zone_name + '_radiator_hull_front_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_front_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_name + '_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_front_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )

                        # Back.
                        self.state_matrix.at[
                            zone_name + '_radiator_hull_rear_temperature',
                            surface_name + '_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_rear_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / zone_data['heat_capacitance_hull']
                        )
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            zone_name + '_radiator_hull_rear_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_rear_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_name + '_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_rear_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )

        def define_heat_transfer_hvac_ahu():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_ahu_type']):
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_ahu_heat_air_flow'
                    ] += (
                        self.building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['ahu_supply_air_temperature_setpoint']
                            - self.building_data.scenarios['linearization_zone_air_temperature_heat']
                        )
                        / zone_data['heat_capacity']
                    )
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_ahu_cool_air_flow'
                    ] += (
                        self.building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['ahu_supply_air_temperature_setpoint']
                            - self.building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                        / zone_data['heat_capacity']
                    )

        def define_heat_transfer_hvac_tu():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_tu_type']):
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_tu_heat_air_flow'
                    ] += (
                        self.building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['tu_supply_air_temperature_setpoint']
                            - self.building_data.scenarios['linearization_zone_air_temperature_heat']
                        )
                        / zone_data['heat_capacity']
                    )
                    self.control_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_tu_cool_air_flow'
                    ] += (
                        self.building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['tu_supply_air_temperature_setpoint']
                            - self.building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                        / zone_data['heat_capacity']
                    )

        def define_co2_transfer_hvac_ahu():

            if pd.notnull(self.building_data.scenarios['co2_model_type']):
                for zone_name, zone_data in self.building_data.zones.iterrows():
                    if pd.notnull(zone_data.at['hvac_ahu_type']) | pd.notnull(zone_data.at['window_type']):
                        self.state_matrix.at[
                            zone_name + '_co2_concentration',
                            zone_name + '_co2_concentration'
                        ] += (
                            - 1.0
                            * (
                                self.building_data.scenarios['linearization_ventilation_rate_per_square_meter']
                                / zone_data.at['zone_height']
                            )
                        )
                        if pd.notnull(zone_data.at['hvac_ahu_type']):
                            self.control_matrix.at[
                                zone_name + '_co2_concentration',
                                zone_name + '_ahu_heat_air_flow'
                            ] += (
                                - 1.0
                                * (
                                    self.building_data.scenarios['linearization_co2_concentration']
                                    / zone_data.at['zone_height']
                                    / zone_data.at['zone_area']
                                )
                            )
                            self.control_matrix.at[
                                zone_name + '_co2_concentration',
                                zone_name + '_ahu_cool_air_flow'
                            ] += (
                                - 1.0
                                * (
                                    self.building_data.scenarios['linearization_co2_concentration']
                                    / zone_data.at['zone_height']
                                    / zone_data.at['zone_area']
                                )
                            )
                        if pd.notnull(zone_data.at['window_type']):
                            self.control_matrix.at[
                                zone_name + '_co2_concentration',
                                zone_name + '_window_air_flow'
                            ] += (
                                - 1.0
                                * (
                                    self.building_data.scenarios['linearization_co2_concentration']
                                    / zone_data.at['zone_height']
                                    / zone_data.at['zone_area']
                                )
                            )
                        # self.disturbance_matrix.at[
                        #     zone_name + '_co2_concentration',
                        #     'constant'
                        # ] += (
                        #     - 1.0
                        #     * self.building_data.scenarios['linearization_co2_concentration']
                        #     * zone_data.at['infiltration_rate'])
                        # )  # TODO: Revise infiltration
                        if pd.notnull(zone_data.at['internal_gain_type']):
                            self.disturbance_matrix.at[
                                zone_name + '_co2_concentration',
                                zone_data.at['internal_gain_type'] + '_internal_gain_occupancy'
                            ] += (
                                self.building_data.parameters['co2_generation_rate_per_person']
                                / zone_data.at['zone_height']
                                / zone_data.at['zone_area']
                            )
                            # division by zone_area since the occupancy here is in p
                            # if iterative and robust BM, no division by zone_area since the occupancy there is in p/m2
                        self.disturbance_matrix.at[
                            zone_name + '_co2_concentration',
                            'constant'
                        ] += (
                            self.building_data.scenarios['linearization_ventilation_rate_per_square_meter']
                            * self.building_data.scenarios['linearization_co2_concentration']
                            / zone_data.at['zone_height']
                        )

        def define_humidity_transfer_hvac_ahu():

            if pd.notnull(self.building_data.scenarios['humidity_model_type']):
                for zone_name, zone_data in self.building_data.zones.iterrows():
                    if pd.notnull(zone_data.at['hvac_ahu_type']):
                        self.state_matrix.at[
                            zone_name + '_absolute_humidity',
                            zone_name + '_absolute_humidity'
                        ] += (
                            - 1.0
                            * (
                                self.building_data.scenarios['linearization_ventilation_rate_per_square_meter']
                                / zone_data.at['zone_height']
                            )
                        )
                        self.control_matrix.at[
                            zone_name + '_absolute_humidity',
                            zone_name + '_ahu_heat_air_flow'
                        ] += (
                            - 1.0
                            * (
                                (
                                    self.building_data.scenarios['linearization_zone_air_humidity_ratio']
                                    - cobmo.utils.calculate_absolute_humidity_humid_air(
                                        zone_data.at['ahu_supply_air_temperature_setpoint'],
                                        zone_data.at['ahu_supply_air_relative_humidity_setpoint']
                                    )
                                )
                                / zone_data.at['zone_height']
                                / zone_data.at['zone_area']
                            )
                        )
                        self.control_matrix.at[
                            zone_name + '_absolute_humidity',
                            zone_name + '_ahu_cool_air_flow'
                        ] += (
                            - 1.0
                            * (
                                (
                                    self.building_data.scenarios['linearization_zone_air_humidity_ratio']
                                    - cobmo.utils.calculate_absolute_humidity_humid_air(
                                        zone_data.at['ahu_supply_air_temperature_setpoint'],
                                        zone_data.at['ahu_supply_air_relative_humidity_setpoint']
                                    )
                                )
                                / zone_data.at['zone_height']
                                / zone_data.at['zone_area']
                            )
                        )
                        if pd.notnull(zone_data.at['window_type']):
                            self.control_matrix.at[
                                zone_name + '_absolute_humidity',
                                zone_name + '_window_air_flow'
                            ] += (
                                - 1.0
                                * (
                                    (
                                        self.building_data.scenarios['linearization_zone_air_humidity_ratio']
                                        - self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                                    )
                                    / zone_data.at['zone_height']
                                    / zone_data.at['zone_area']
                                )
                            )
                        self.disturbance_matrix.at[
                            zone_name + '_absolute_humidity',
                            'constant'
                        ] += (
                            - 1.0
                            * (
                                (
                                    self.building_data.scenarios['linearization_zone_air_humidity_ratio']
                                    - self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                                )
                                * zone_data.at['infiltration_rate']
                            )
                        )  # TODO: Revise infiltration
                        if pd.notnull(zone_data.at['internal_gain_type']):
                            self.disturbance_matrix.at[
                                zone_name + '_absolute_humidity',
                                zone_data.at['internal_gain_type'] + '_internal_gain_occupancy'
                            ] += (
                                self.building_data.parameters['moisture_generation_rate_per_person']
                                / zone_data.at['zone_height']
                                / zone_data.at['zone_area']
                                / self.building_data.parameters['density_air']
                            )
                        self.disturbance_matrix.at[
                            zone_name + '_absolute_humidity',
                            'constant'
                        ] += (
                            self.building_data.scenarios['linearization_ventilation_rate_per_square_meter']
                            * self.building_data.scenarios['linearization_zone_air_humidity_ratio']
                            / zone_data.at['zone_height']
                        )

        def define_storage_state_of_charge():

            # Sensible thermal storage.
            if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':
                # Storage charge.
                self.control_matrix.at[
                    'sensible_thermal_storage_state_of_charge',
                    'sensible_storage_charge_cool_thermal_power',
                ] += (
                    1.0
                    / (
                        self.building_data.parameters['water_specific_heat']
                        * self.building_data.scenarios['storage_sensible_temperature_delta']
                    )
                    * self.building_data.scenarios['storage_round_trip_efficiency']
                )

                # Storage losses.
                # - Thermal losses are considered negligible, but a very small loss is added to keep the state matrix
                #   non-singular and hence invertible.
                # - TODO: For detailed losses depending on the storage size see `cobmo/README_storage.md`
                self.state_matrix.at[
                    'sensible_thermal_storage_state_of_charge',
                    'sensible_thermal_storage_state_of_charge'
                ] += (
                    - 1e-17
                )

                for zone_name, zone_data in self.building_data.zones.iterrows():
                    # TODO: Differentiate heating / cooling and define heating discharge.

                    if pd.notnull(zone_data['hvac_ahu_type']):
                        # Storage discharge to AHU for cooling.
                        self.control_matrix.at[
                            'sensible_thermal_storage_state_of_charge',
                            zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power',
                        ] += (
                            - 1.0
                            / (
                                self.building_data.parameters['water_specific_heat']
                                * self.building_data.scenarios['storage_sensible_temperature_delta']
                            )
                        )

                    if pd.notnull(zone_data['hvac_tu_type']):
                        # Storage discharge to TU for cooling.
                        self.control_matrix.at[
                            'sensible_thermal_storage_state_of_charge',
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power',
                        ] += (
                            - 1.0
                            / (
                                self.building_data.parameters['water_specific_heat']
                                * self.building_data.scenarios['storage_sensible_temperature_delta']
                            )
                        )

            # Battery storage.
            if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':
                # Storage charge.
                self.control_matrix.at[
                    'battery_storage_state_of_charge',
                    'battery_storage_charge_electric_power'
                ] += (
                    self.building_data.scenarios['storage_round_trip_efficiency']
                )  # TODO: Make the battery loss dependent on the outdoor temperature.

                # Storage losses.
                # - There are no battery storage losses, but a very small loss is added to keep the state matrix
                #   non-singular and hence invertible.
                self.state_matrix.at[
                    'battery_storage_state_of_charge',
                    'battery_storage_state_of_charge'
                ] += (
                    - 1E-17
                )

                for zone_name, zone_data in self.building_data.zones.iterrows():
                    # TODO: Differentiate heating / cooling and define heating discharge.

                    if pd.notnull(zone_data['hvac_ahu_type']):
                        # Storage discharge to AHU for cooling.
                        self.control_matrix.at[
                            'battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_ahu_cool_electric_power',
                        ] += (
                            - 1.0
                        )

                        # Storage discharge to AHU for heating.
                        self.control_matrix.at[
                            'battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_ahu_heat_electric_power',
                        ] += (
                            - 1.0
                        )

                    if pd.notnull(zone_data['hvac_tu_type']):
                        # Storage discharge to TU for cooling.
                        self.control_matrix.at[
                            'battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_tu_cool_electric_power',
                        ] += (
                            - 1.0
                        )

                        # Storage discharge to TU for heating.
                        self.control_matrix.at[
                            'battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_tu_heat_electric_power',
                        ] += (
                            - 1.0
                        )

        def define_output_zone_temperature():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                self.state_output_matrix.at[
                    zone_name + '_temperature',
                    zone_name + '_temperature'
                ] = 1.0

        def define_output_zone_co2_concentration():

            if pd.notnull(self.building_data.scenarios['co2_model_type']):
                for zone_name, zone_data in self.building_data.zones.iterrows():
                    if pd.notnull(zone_data['hvac_ahu_type']) | pd.notnull(zone_data['window_type']):
                        self.state_output_matrix.at[
                            zone_name + '_co2_concentration',
                            zone_name + '_co2_concentration'
                        ] = 1.0

        def define_output_zone_humidity():

            if pd.notnull(self.building_data.scenarios['humidity_model_type']):
                for zone_name, zone_data in self.building_data.zones.iterrows():
                    if pd.notnull(zone_data['hvac_ahu_type']):
                        self.state_output_matrix.at[
                            zone_name + '_absolute_humidity',
                            zone_name + '_absolute_humidity'
                        ] = 1.0

        def define_output_hvac_generic_power():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_generic_type']):

                    # Cooling power.
                    self.control_output_matrix.at[
                        zone_name + '_generic_cool_thermal_power_cooling',
                        zone_name + '_generic_cool_thermal_power'
                    ] = 1.0
                    self.control_output_matrix.at[
                        zone_name + '_generic_cool_electric_power_cooling',
                        zone_name + '_generic_cool_thermal_power'
                    ] = (
                        1.0
                        / zone_data['generic_cooling_efficiency']
                    )

                    # Heating power.
                    self.control_output_matrix.at[
                        zone_name + '_generic_heat_thermal_power_heating',
                        zone_name + '_generic_heat_thermal_power'
                    ] = 1.0
                    self.control_output_matrix.at[
                        zone_name + '_generic_heat_electric_power_heating',
                        zone_name + '_generic_heat_thermal_power'
                    ] = (
                        1.0
                        / zone_data['generic_heating_efficiency']
                    )

        def define_output_hvac_radiator_power():
            """Define output equations for the thermal and electric power demand due to radiators."""

            if pd.notnull(self.building_data.zones['hvac_radiator_type']).any():
                zones_index = pd.notnull(self.building_data.zones['hvac_radiator_type'])

                # Heating power (radiators only require heating power).
                self.control_output_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power_heating',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power'
                ] = 1.0
                # TODO: Define heating plant COP for radiators.
                self.control_output_matrix.loc[
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_electric_power_heating',
                    self.building_data.zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power'
                ] = (
                    1.0
                    / 1.0
                )

        def define_output_hvac_ahu_power():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_ahu_type']):
                    # Obtain parameters.
                    ahu_supply_air_absolute_humidity_setpoint = (
                        cobmo.utils.calculate_absolute_humidity_humid_air(
                            zone_data['ahu_supply_air_temperature_setpoint'],
                            zone_data['ahu_supply_air_relative_humidity_setpoint']
                        )
                    )
                    # TODO: Define zone linearization temperature and humidity in database.
                    linearization_zone_air_temperature = (
                        0.5
                        * (
                            self.building_data.scenarios['linearization_zone_air_temperature_heat']
                            + self.building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                    )
                    linearization_zone_air_absolute_humidity = (
                        0.5
                        * (
                            self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                            + ahu_supply_air_absolute_humidity_setpoint
                        )
                    )
                    delta_enthalpy_ahu_recovery = (
                        cobmo.utils.calculate_enthalpy_humid_air(
                            linearization_zone_air_temperature,
                            linearization_zone_air_absolute_humidity
                        )
                        - cobmo.utils.calculate_enthalpy_humid_air(
                            self.building_data.scenarios['linearization_ambient_air_temperature'],
                            linearization_zone_air_absolute_humidity
                        )
                    )

                    # Obtain enthalpies.
                    if (
                        self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                        <= ahu_supply_air_absolute_humidity_setpoint
                    ):
                        delta_enthalpy_ahu_cooling = min(
                            0.0,
                            cobmo.utils.calculate_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                self.building_data.scenarios['linearization_ambient_air_temperature'],
                                self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                            )
                        )
                        delta_enthalpy_ahu_heating = max(
                            0.0,
                            cobmo.utils.calculate_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                self.building_data.scenarios['linearization_ambient_air_temperature'],
                                self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                            )
                        )
                        delta_enthalpy_ahu_recovery_cooling = max(
                            delta_enthalpy_ahu_cooling,
                            min(
                                0.0,
                                zone_data['ahu_return_air_heat_recovery_efficiency']
                                * delta_enthalpy_ahu_recovery
                            )
                        )
                        delta_enthalpy_ahu_recovery_heating = min(
                            delta_enthalpy_ahu_heating,
                            max(
                                0.0,
                                zone_data['ahu_return_air_heat_recovery_efficiency']
                                * delta_enthalpy_ahu_recovery
                            )
                        )
                    else:
                        delta_enthalpy_ahu_cooling = (
                            cobmo.utils.calculate_dew_point_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                zone_data['ahu_supply_air_relative_humidity_setpoint']
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                self.building_data.scenarios['linearization_ambient_air_temperature'],
                                self.building_data.scenarios['linearization_ambient_air_humidity_ratio']
                            )
                        )
                        delta_enthalpy_ahu_heating = (
                            cobmo.utils.calculate_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                ahu_supply_air_absolute_humidity_setpoint
                            )
                            - cobmo.utils.calculate_dew_point_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                zone_data['ahu_supply_air_relative_humidity_setpoint']
                            )
                        )
                        delta_enthalpy_ahu_recovery_cooling = max(
                            delta_enthalpy_ahu_cooling,
                            min(
                                0.0,
                                zone_data['ahu_return_air_heat_recovery_efficiency']
                                * delta_enthalpy_ahu_recovery
                            )
                        )
                        delta_enthalpy_ahu_recovery_heating = 0.0

                    # TODO: Revise code to remove `abs(delta_enthalpy...)` and invert `ahu_cooling_efficiency`.

                    # Cooling power.
                    self.control_output_matrix.at[
                        zone_name + '_ahu_heat_thermal_power_cooling',
                        zone_name + '_ahu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_heat_electric_power_cooling',
                        zone_name + '_ahu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                        / zone_data['ahu_cooling_efficiency']
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_thermal_power_cooling',
                        zone_name + '_ahu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_electric_power_cooling',
                        zone_name + '_ahu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                        / zone_data['ahu_cooling_efficiency']
                    )

                    # Heating power.
                    self.control_output_matrix.at[
                        zone_name + '_ahu_heat_thermal_power_heating',
                        zone_name + '_ahu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_heat_electric_power_heating',
                        zone_name + '_ahu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                        / zone_data['ahu_heating_efficiency']
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_thermal_power_heating',
                        zone_name + '_ahu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_electric_power_heating',
                        zone_name + '_ahu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                        / zone_data['ahu_heating_efficiency']
                    )

                    # Fan power.
                    self.control_output_matrix.at[
                        zone_name + '_ahu_heat_electric_power_fan',
                        zone_name + '_ahu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * zone_data['ahu_fan_efficiency']
                    )
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_electric_power_fan',
                        zone_name + '_ahu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * zone_data['ahu_fan_efficiency']
                    )

                    # Sensible thermal storage discharge.
                    if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':

                        # Cooling power.
                        self.control_output_matrix.at[
                            zone_name + '_ahu_cool_thermal_power_cooling',
                            zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power'
                        ] = - 1.0
                        self.control_output_matrix.at[
                            zone_name + '_ahu_cool_electric_power_cooling',
                            zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power'
                        ] = (
                            - 1.0
                            / zone_data['ahu_cooling_efficiency']
                        )

                        # Heating power.
                        # TODO: Add consideration for sensible storage heating.

                    # Battery storage discharge.
                    if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':

                        # Cooling power.
                        self.control_output_matrix.at[
                            zone_name + '_ahu_heat_electric_power_cooling',
                            zone_name + '_battery_storage_to_zone_ahu_heat_electric_power'
                        ] = (
                            - 1.0
                            * (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_heat_electric_power_cooling',
                                    zone_name + '_ahu_heat_air_flow'
                                ]
                            ) / (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_heat_electric_power_cooling',
                                    zone_name + '_ahu_heat_air_flow'
                                ]
                                + self.control_output_matrix.at[
                                    zone_name + '_ahu_heat_electric_power_heating',
                                    zone_name + '_ahu_heat_air_flow'
                                ]
                            )
                        )
                        self.control_output_matrix.at[
                            zone_name + '_ahu_cool_electric_power_cooling',
                            zone_name + '_battery_storage_to_zone_ahu_cool_electric_power'
                        ] = (
                            - 1.0
                            * (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_cool_electric_power_cooling',
                                    zone_name + '_ahu_cool_air_flow'
                                ]
                            ) / (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_cool_electric_power_cooling',
                                    zone_name + '_ahu_cool_air_flow'
                                ]
                                + self.control_output_matrix.at[
                                    zone_name + '_ahu_cool_electric_power_heating',
                                    zone_name + '_ahu_cool_air_flow'
                                ]
                            )
                        )

                        # Heating power.
                        self.control_output_matrix.at[
                            zone_name + '_ahu_heat_electric_power_heating',
                            zone_name + '_battery_storage_to_zone_ahu_heat_electric_power'
                        ] = (
                            - 1.0
                            * (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_heat_electric_power_heating',
                                    zone_name + '_ahu_heat_air_flow'
                                ]
                            ) / (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_heat_electric_power_cooling',
                                    zone_name + '_ahu_heat_air_flow'
                                ]
                                + self.control_output_matrix.at[
                                    zone_name + '_ahu_heat_electric_power_heating',
                                    zone_name + '_ahu_heat_air_flow'
                                ]
                            )
                        )
                        self.control_output_matrix.at[
                            zone_name + '_ahu_cool_electric_power_heating',
                            zone_name + '_battery_storage_to_zone_ahu_cool_electric_power'
                        ] = (
                            - 1.0
                            * (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_cool_electric_power_heating',
                                    zone_name + '_ahu_cool_air_flow'
                                ]
                            ) / (
                                self.control_output_matrix.at[
                                    zone_name + '_ahu_cool_electric_power_cooling',
                                    zone_name + '_ahu_cool_air_flow'
                                ]
                                + self.control_output_matrix.at[
                                    zone_name + '_ahu_cool_electric_power_heating',
                                    zone_name + '_ahu_cool_air_flow'
                                ]
                            )
                        )

        def define_output_hvac_tu_power():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_tu_type']):
                    # Calculate enthalpies.
                    if zone_data['tu_air_intake_type'] == 'zone':
                        delta_enthalpy_tu_cooling = self.building_data.parameters['heat_capacity_air'] * (
                            self.building_data.scenarios['linearization_zone_air_temperature_cool']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                        delta_enthalpy_tu_heating = self.building_data.parameters['heat_capacity_air'] * (
                            self.building_data.scenarios['linearization_zone_air_temperature_heat']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                    elif zone_data['tu_air_intake_type'] == 'ahu':
                        delta_enthalpy_tu_cooling = self.building_data.parameters['heat_capacity_air'] * (
                            self.building_data.scenarios['ahu_supply_air_temperature_setpoint']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                        delta_enthalpy_tu_heating = self.building_data.parameters['heat_capacity_air'] * (
                            self.building_data.scenarios['ahu_supply_air_temperature_setpoint']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                    else:
                        logger.error(f"Unknown `tu_air_intake_type` type: {zone_data['tu_air_intake_type']}")
                        raise ValueError

                    # Cooling power.
                    self.control_output_matrix.at[
                        zone_name + '_tu_cool_thermal_power_cooling',
                        zone_name + '_tu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * abs(delta_enthalpy_tu_cooling)
                    )
                    self.control_output_matrix.at[
                        zone_name + '_tu_cool_electric_power_cooling',
                        zone_name + '_tu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * abs(delta_enthalpy_tu_cooling)
                        / zone_data['tu_cooling_efficiency']
                    )

                    # Heating power.
                    self.control_output_matrix.at[
                        zone_name + '_tu_heat_electric_power_heating',
                        zone_name + '_tu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * abs(delta_enthalpy_tu_heating)
                        / zone_data['tu_heating_efficiency']
                    )
                    self.control_output_matrix.at[
                        zone_name + '_tu_heat_thermal_power_heating',
                        zone_name + '_tu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * abs(delta_enthalpy_tu_heating)
                    )

                    # Fan power.
                    self.control_output_matrix.at[
                        zone_name + '_tu_heat_electric_power_fan',
                        zone_name + '_tu_heat_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * zone_data['tu_fan_efficiency']
                    )
                    self.control_output_matrix.at[
                        zone_name + '_tu_cool_electric_power_fan',
                        zone_name + '_tu_cool_air_flow'
                    ] = (
                        self.building_data.parameters['density_air']
                        * zone_data['tu_fan_efficiency']
                    )

                    # Sensible thermal storage discharge.
                    if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':

                        # Cooling power.
                        self.control_output_matrix.at[
                            zone_name + '_tu_cool_thermal_power_cooling',
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power'
                        ] = - 1.0
                        self.control_output_matrix.at[
                            zone_name + '_tu_cool_electric_power_cooling',
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power'
                        ] = (
                            - 1.0
                            / zone_data['tu_cooling_efficiency']
                        )

                        # Heating power.
                        # TODO: Add consideration for sensible storage heating.

                    # Battery storage discharge.
                    if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':

                        # Cooling power.
                        self.control_output_matrix.at[
                            zone_name + '_tu_cool_electric_power_cooling',
                            zone_name + '_battery_storage_to_zone_tu_cool_electric_power'
                        ] += (
                            - 1.0
                        )

                        # Heating power.
                        self.control_output_matrix.at[
                            zone_name + '_tu_heat_electric_power_heating',
                            zone_name + '_battery_storage_to_zone_tu_heat_electric_power'
                        ] += (
                            - 1.0
                        )

        def define_output_fresh_air_flow():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_ahu_type']):
                    self.control_output_matrix.at[
                        zone_name + '_total_fresh_air_flow',
                        zone_name + '_ahu_heat_air_flow'
                    ] = 1.0
                    self.control_output_matrix.at[
                        zone_name + '_total_fresh_air_flow',
                        zone_name + '_ahu_cool_air_flow'
                    ] = 1.0
                if pd.notnull(zone_data['window_type']):
                    self.control_output_matrix.at[
                        zone_name + '_total_fresh_air_flow',
                        zone_name + '_window_air_flow'
                    ] = 1.0
                if pd.notnull(zone_data['window_type']) | pd.notnull(zone_data['hvac_ahu_type']):
                    self.disturbance_output_matrix.at[
                        zone_name + '_total_fresh_air_flow',
                        'constant'
                    ] += (
                        zone_data['infiltration_rate']
                        * zone_data['zone_area']
                        * zone_data['zone_height']
                    )  # TODO: Revise infiltration

        def define_output_ahu_fresh_air_flow():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_ahu_type']):
                    self.control_output_matrix.at[
                        zone_name + '_ahu_fresh_air_flow',
                        zone_name + '_ahu_heat_air_flow'
                    ] = 1.0
                    self.control_output_matrix.at[
                        zone_name + '_ahu_fresh_air_flow',
                        zone_name + '_ahu_cool_air_flow'
                    ] = 1.0

        def define_output_window_fresh_air_flow():
            for zone_name, zone_data in self.building_data.zones.iterrows():
                if pd.notnull(zone_data['window_type']):
                    self.control_output_matrix.at[
                        zone_name + '_window_fresh_air_flow',
                        zone_name + '_window_air_flow'
                    ] = 1.0

        def define_output_storage_state_of_charge():

            # Sensible thermal storage.
            if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':
                self.state_output_matrix.at[
                    'sensible_thermal_storage_state_of_charge',
                    'sensible_thermal_storage_state_of_charge'
                ] = 1.0

            # Battery storage.
            if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':
                self.state_output_matrix.at[
                    'battery_storage_state_of_charge',
                    'battery_storage_state_of_charge'
                ] = 1.0

        def define_output_storage_charge():
            # TODO: Remove redundant charge ouputs.

            # Sensible thermal storage charge thermal power.
            if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':
                # Heating.
                # TODO: Add consideration for sensible storage heating / cooling.

                # Cooling.
                self.control_output_matrix.at[
                    'sensible_storage_charge_cool_thermal_power',
                    'sensible_storage_charge_cool_thermal_power'
                ] = 1.0

            # Sensible thermal storage charge electric power.
            if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':
                # Heating.

                # Cooling.
                self.control_output_matrix.at[
                    'sensible_storage_charge_cool_electric_power',
                    'sensible_storage_charge_cool_thermal_power'
                ] += (
                    1.0
                    / self.building_data.zones['ahu_cooling_efficiency'][0]
                    if pd.notnull(self.building_data.zones['ahu_cooling_efficiency'][0]) else 1.0
                    * 1.0000001  # TODO: Very small loss to avoid simultaneous charge and discharge still needed?
                )

            # Battery storage charge.
            if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':
                self.control_output_matrix.at[
                    'battery_storage_charge_electric_power',
                    'battery_storage_charge_electric_power'
                ] = 1.0

        def define_output_storage_discharge():

            for zone_name, zone_data in self.building_data.zones.iterrows():
                # Sensible thermal storage.
                if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':
                    # TODO: Add consideration for sensible storage heating / cooling.

                    if pd.notnull(zone_data['hvac_ahu_type']):
                        # Storage discharge to AHU for cooling.
                        self.control_output_matrix.at[
                            zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power',
                            zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power'
                        ] = 1.0

                    if pd.notnull(zone_data['hvac_tu_type']):
                        # Storage discharge to TU for cooling.
                        self.control_output_matrix.at[
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power',
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power'
                        ] = 1.0

                # Battery storage.
                if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':
                    if pd.notnull(zone_data['hvac_ahu_type']):
                        # Storage discharge to AHU for cooling.
                        self.control_output_matrix.at[
                            zone_name + '_battery_storage_to_zone_ahu_cool_electric_power',
                            zone_name + '_battery_storage_to_zone_ahu_cool_electric_power'
                        ] = 1.0

                        # Storage discharge to AHU for heating.
                        self.control_output_matrix.at[
                            zone_name + '_battery_storage_to_zone_ahu_heat_electric_power',
                            zone_name + '_battery_storage_to_zone_ahu_heat_electric_power'
                        ] = 1.0

                    if pd.notnull(zone_data['hvac_tu_type']):
                        # Storage discharge to TU for cooling.
                        self.control_output_matrix.at[
                            zone_name + '_battery_storage_to_zone_tu_cool_electric_power',
                            zone_name + '_battery_storage_to_zone_tu_cool_electric_power'
                        ] = 1.0

                    if pd.notnull(zone_data['hvac_tu_type']):
                        # Storage discharge to TU for heating.
                        self.control_output_matrix.at[
                            zone_name + '_battery_storage_to_zone_tu_heat_electric_power',
                            zone_name + '_battery_storage_to_zone_tu_heat_electric_power'
                        ] = 1.0

        def define_output_surface_temperature():

            for surface_name, surface_data in (
                    pd.concat([
                        self.building_data.surfaces_adiabatic,
                        self.building_data.surfaces_exterior,
                        self.building_data.surfaces_interior
                    ], sort=False)
            ).iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    self.state_output_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] = 1.0

        def define_output_surfaces_exterior_irradiation_gain_exterior():

            for surface_name, surface_data in self.building_data.surfaces_exterior.iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    self.disturbance_output_matrix.at[
                        surface_name + '_irradiation_gain_exterior',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity']
                        * (1 - surface_data['window_wall_ratio'])
                    )
                else:  # Surfaces with neglected heat capacity
                    self.disturbance_output_matrix.at[
                        surface_data['surface_name'] + '_irradiation_gain_exterior',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity']
                        * (1 - surface_data['window_wall_ratio'])
                    )

        def define_output_surfaces_exterior_convection_interior():

            for surface_name, surface_data in self.building_data.surfaces_exterior.iterrows():
                # Total zone surface area for later calculating share of interior (indirect) irradiation
                zone_surface_area = sum(
                    zone_surface_data['surface_area']
                    * (1 - zone_surface_data['window_wall_ratio'])
                    for zone_surface_name, zone_surface_data in pd.concat(
                        [
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ],
                            self.building_data.surfaces_interior[
                                self.building_data.surfaces_interior['zone_name'] == surface_data['zone_name']
                            ],
                            self.building_data.surfaces_interior[
                                self.building_data.surfaces_interior['zone_adjacent_name'] == surface_data['zone_name']
                            ],
                            self.building_data.surfaces_adiabatic[
                                self.building_data.surfaces_adiabatic['zone_name'] == surface_data['zone_name']
                            ]
                        ],
                        sort=False
                    ).iterrows()  # For all surfaces adjacent to the zone
                )

                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            self.building_data.surfaces_exterior[
                                self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / zone_surface_area
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * (1.0 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                - (
                                    1.0
                                    + (
                                        self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    )
                                    / (
                                        2.0
                                        * surface_data['thermal_resistance_surface'] ** (- 1)
                                    )
                                ) ** (- 1)
                            )
                        )
                    self.state_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        surface_name + '_temperature'
                    ] += (
                        (1.0 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                    )
                    self.state_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * (1.0 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['thermal_resistance_surface'] ** (- 1)
                            )
                        ) ** (- 1)
                    )
                else:  # Surfaces with neglected heat capacity
                    # Complete convective heat transfer from surface to zone
                    self.disturbance_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                    )
                    self.disturbance_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        'ambient_air_temperature'
                    ] += (
                        (
                            self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_surface_ground']
                        )
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                    )
                    self.disturbance_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_surface_sky']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                    )
                    self.state_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            + 1.0
                            / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + 1.0
                            / (surface_data['thermal_resistance_surface'] ** (- 1))
                        ) ** (- 1)
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_data.surfaces_exterior[
                        self.building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / zone_surface_area
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / self.building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + (
                                    self.building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / (surface_data['thermal_resistance_surface'] ** (- 1))
                            ) ** (- 1))
                        )

        def define_output_grid():

            # TODO: Move storage charge / discharge to grid balance equations.

            # Cooling power balance.
            if any([connect_electric_grid, connect_thermal_grid_cooling]):
                self.control_output_matrix.loc[
                    'grid_thermal_power_cooling_balance',
                    :
                ] = (
                    self.control_output_matrix.loc[
                        self.outputs.str.contains('thermal_power_cooling'),
                        :
                    ].sum(axis=0)
                )
                # TODO: This operation contains itself and does not capture storage charge power. Below may be similar.
                if connect_electric_grid:
                    self.control_output_matrix.at[
                        'grid_thermal_power_cooling_balance',
                        'plant_thermal_power_cooling'
                    ] = -1.0
                if connect_thermal_grid_cooling:
                    self.control_output_matrix.at[
                        'grid_thermal_power_cooling_balance',
                        'grid_thermal_power_cooling'
                    ] = -1.0

            # Heating power balance.
            if any([connect_electric_grid, connect_thermal_grid_heating]):
                self.control_output_matrix.loc[
                    'grid_thermal_power_heating_balance',
                    :
                ] = (
                    self.control_output_matrix.loc[
                        self.outputs.str.contains('thermal_power_heating'),
                        :
                    ].sum(axis=0)
                )
                if connect_electric_grid:
                    self.control_output_matrix.at[
                        'grid_thermal_power_heating_balance',
                        'plant_thermal_power_heating'
                    ] = -1.0
                if connect_thermal_grid_heating:
                    self.control_output_matrix.at[
                        'grid_thermal_power_heating_balance',
                        'grid_thermal_power_heating'
                    ] = -1.0

            # Electric grid power.
            if connect_electric_grid:
                self.control_output_matrix.loc[
                    'grid_electric_power',
                    :
                ] = (
                    self.control_output_matrix.loc[
                        self.outputs.str.contains('electric_power_fan'),
                        :
                    ].sum(axis=0)
                )
                self.control_output_matrix.at[
                    'grid_electric_power',
                    'plant_thermal_power_cooling'
                ] = (
                    1.0
                    / self.building_data.zones['ahu_cooling_efficiency'][0]
                    if pd.notnull(self.building_data.zones['ahu_cooling_efficiency'][0]) else 1.0
                    # TODO: Define heating / cooling plant.
                )
                self.control_output_matrix.at[
                    'grid_electric_power',
                    'plant_thermal_power_heating'
                ] = (
                    1.0
                    / self.building_data.zones['ahu_heating_efficiency'][0]
                    if pd.notnull(self.building_data.zones['ahu_heating_efficiency'][0]) else 1.0
                    # TODO: Define heating / cooling plant.
                )
                for zone_name, zone_data in self.building_data.zones.iterrows():
                    if pd.notnull(zone_data.at['internal_gain_type']):
                        self.disturbance_output_matrix.loc[
                            'grid_electric_power',
                            zone_data.at['internal_gain_type'] + '_internal_gain_appliances'
                        ] += (
                            zone_data.at['internal_gain_appliances_factor']
                            * zone_data.at['zone_area']
                        )
                self.control_output_matrix.at[
                    'plant_thermal_power_cooling',
                    'plant_thermal_power_cooling'
                ] = 1.0
                self.control_output_matrix.at[
                    'plant_thermal_power_heating',
                    'plant_thermal_power_heating'
                ] = 1.0

            # Thermal cooling grid power.
            if connect_thermal_grid_cooling:
                self.control_output_matrix.at[
                    'grid_thermal_power_cooling',
                    'grid_thermal_power_cooling'
                ] = 1.0

            # Thermal heating grid power.
            if connect_thermal_grid_heating:
                self.control_output_matrix.at[
                    'grid_thermal_power_heating',
                    'grid_thermal_power_heating'
                ] = 1.0

        def define_disturbance_timeseries():

            # Reindex, interpolate and construct full disturbance timeseries.
            self.disturbance_timeseries = pd.concat(
                [
                    self.building_data.weather_timeseries[[
                        'ambient_air_temperature',
                        'sky_temperature',
                        'irradiation_horizontal',
                        'irradiation_east',
                        'irradiation_south',
                        'irradiation_west',
                        'irradiation_north'
                    ]],
                    self.building_data.internal_gain_timeseries,
                    (
                        pd.DataFrame(
                            1.0,
                            self.timesteps,
                            ['constant']
                        ) if self.define_constant else pd.DataFrame([])  # Append constant only when needed.
                    )
                ],
                axis='columns',
            ).rename_axis('disturbance_name', axis='columns')

        def define_electricity_price_timeseries():

            if pd.isnull(self.building_data.scenarios['price_type']):
                # If no price_type defined, generate a flat price signal.
                self.electricity_price_timeseries = (
                    pd.DataFrame(
                        [[None, None, 1.0]],
                        columns=['time', 'price_type', 'price'],
                        index=self.timesteps
                    )
                )
                self.electricity_price_timeseries['time'] = self.timesteps
            else:
                self.electricity_price_timeseries = self.building_data.electricity_price_timeseries

        def define_electricity_price_distribution_timeseries():
            self.electricity_price_distribution_timeseries = (
                self.building_data.electricity_price_distribution_timeseries
            )

        def define_output_constraint_timeseries():

            # Do not define constraints, if `constraint_type` not defined for any zones.
            if any(pd.isnull(self.building_data.zones.loc[:, 'constraint_type'])):
                logger.debug('Skipping definition of constraint timeseries due to missing constraint type definition.')
                return

            # Instantiate constraint timeseries.
            self.output_constraint_timeseries_maximum = pd.DataFrame(
                +1.0 * np.infty,
                self.timesteps,
                self.outputs
            )
            self.output_constraint_timeseries_minimum = pd.DataFrame(
                -1.0 * np.infty,
                self.timesteps,
                self.outputs
            )

            # Obtain indexing shorthands.
            zones_ventilation_index = (
                pd.notnull(self.building_data.zones['hvac_ahu_type'])
                # | pd.notnull(self.building_data.zones['window_type'])  # TODO: revise window air flow.
            )
            zones_ahu_index = (
                pd.notnull(self.building_data.zones['hvac_ahu_type'])
            )

            # Minimum constraint for power outputs.
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_power')
            ] = 0.0

            # Minimum constraint for flow outputs.
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_flow')
            ] = 0.0

            # Minimum constraint for storage charge and discharge outputs.
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_storage_charge')
            ] = 0.0
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_storage_to_zone')
            ] = 0.0

            # Minimum / maximum constraint for balance outputs.
            self.output_constraint_timeseries_minimum.loc[
                :, self.outputs.str.contains('_balance')
            ] = 0.0
            self.output_constraint_timeseries_maximum.loc[
                :, self.outputs.str.contains('_balance')
            ] = 0.0

            # If a heating / cooling session is defined, the cooling / heating air flow is forced to 0.
            # TODO: Remove heating / cooling session.
            if self.building_data.scenarios['heating_cooling_session'] == 'heating':
                self.output_constraint_timeseries_maximum.loc[
                    :,  self.outputs.str.contains('_cool')
                ] = 0.0
            if self.building_data.scenarios['heating_cooling_session'] == 'cooling':
                self.output_constraint_timeseries_maximum.loc[
                    :,  self.outputs.str.contains('_heat')
                ] = 0.0

            # Minimum / maximum constraint for zone air temperature.
            self.output_constraint_timeseries_minimum.loc[
                :, self.building_data.zones['zone_name'] + '_temperature'
            ] = (
                self.building_data.constraint_timeseries.loc[
                    :, self.building_data.zones['constraint_type'] + '_minimum_air_temperature'
                ]
            ).values
            self.output_constraint_timeseries_maximum.loc[
                :, self.building_data.zones['zone_name'] + '_temperature'
            ] = (
                self.building_data.constraint_timeseries.loc[
                    :, self.building_data.zones['constraint_type'] + '_maximum_air_temperature'
                ]
            ).values

            # Minimum constraint for zone fresh air flow.
            self.output_constraint_timeseries_minimum.loc[
                :, self.building_data.zones.loc[zones_ventilation_index, 'zone_name'] + '_total_fresh_air_flow'
            ] = (
                self.building_data.constraint_timeseries.loc[
                    :, (
                        self.building_data.zones.loc[zones_ventilation_index, 'constraint_type']
                        + '_minimum_fresh_air_flow_per_area_no_dcv'
                    )
                ].values
                * self.building_data.zones.loc[zones_ventilation_index, 'zone_area'].values
            )

            # Maximum constraint for zone CO2 concentration.
            # TODO: Revise DCV implementation (check previous constraint implementation for missing constraints).
            if pd.notnull(self.building_data.scenarios['co2_model_type']):
                self.output_constraint_timeseries_minimum.loc[
                    :, self.building_data.zones.loc[zones_ventilation_index, 'zone_name'] + '_co2_concentration'
                ] = (
                    self.building_data.constraint_timeseries.loc[
                        :, (
                            self.building_data.zones.loc[zones_ventilation_index, 'constraint_type']
                            + '_maximum_co2_concentration'
                        )
                    ]
                ).values

            # Minimum / maximum constraint for zone humidity concentration.
            # TODO: Revise DCV implementation (check previous constraint implementation for missing constraints).
            if pd.notnull(self.building_data.scenarios['humidity_model_type']):
                self.output_constraint_timeseries_minimum.loc[
                    :, self.building_data.zones.loc[zones_ahu_index, 'zone_name'] + '_absolute_humidity'
                ] = (
                    np.vectorize(cobmo.utils.calculate_absolute_humidity_humid_air)(
                        self.building_data.scenarios['linearization_zone_air_temperature_cool'],
                        self.building_data.constraint_timeseries.loc[
                            :, (
                                self.building_data.zones.loc[zones_ahu_index, 'constraint_type']
                                + '_minimum_relative_humidity'
                            )
                        ]
                    )
                )
                self.output_constraint_timeseries_maximum.loc[
                    :, self.building_data.zones.loc[zones_ahu_index, 'zone_name'] + '_absolute_humidity'
                ] = (
                    np.vectorize(cobmo.utils.calculate_absolute_humidity_humid_air)(
                        self.building_data.scenarios['linearization_zone_air_temperature_cool'],
                        self.building_data.constraint_timeseries.loc[
                            :, (
                                self.building_data.zones.loc[zones_ahu_index, 'constraint_type']
                                + '_maximum_relative_humidity'
                            )
                        ]
                    )
                )

            # Minimum / maximum constraints for storage state of charge.
            # TODO: Validate storage size units.

            # Sensible thermal storage.
            if self.building_data.scenarios['building_storage_type'] == 'default_sensible_thermal_storage':
                self.output_constraint_timeseries_minimum.loc[
                    :, 'sensible_thermal_storage_state_of_charge'
                ] = 0.0
                self.output_constraint_timeseries_maximum.loc[
                    :, 'sensible_thermal_storage_state_of_charge'
                ] = (
                    self.building_data.scenarios['storage_size']
                    * self.building_data.parameters['water_density']  # Convert volume to mass.
                )

            # Battery storage.
            if self.building_data.scenarios['building_storage_type'] == 'default_battery_storage':
                self.output_constraint_timeseries_minimum.loc[
                    :, 'battery_storage_state_of_charge'
                ] = (
                    0.0
                    # TODO: Revise implementation of depth of discharge.
                    # + self.building_data.scenarios['storage_size']
                    # * (1.0 - self.building_data.scenarios['storage_battery_depth_of_discharge'])
                )
                self.output_constraint_timeseries_maximum.loc[
                    :, 'battery_storage_state_of_charge'
                ] = (
                    self.building_data.scenarios['storage_size']
                )

        def discretize_model():
            """
            - Discretization assuming zero order hold
            - Source: https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
            """

            state_matrix_discrete = scipy.linalg.expm(
                self.state_matrix.values
                * self.timestep_delta.seconds
            )
            control_matrix_discrete = (
                np.linalg.matrix_power(
                    self.state_matrix.values,
                    -1
                ).dot(
                    state_matrix_discrete
                    - np.identity(self.state_matrix.shape[0])
                ).dot(
                    self.control_matrix.values
                )
            )
            disturbance_matrix_discrete = (
                np.linalg.matrix_power(
                    self.state_matrix.values,
                    -1
                ).dot(
                    state_matrix_discrete
                    - np.identity(self.state_matrix.shape[0])
                ).dot(
                    self.disturbance_matrix.values
                )
            )

            self.state_matrix = pd.DataFrame(
                data=state_matrix_discrete,
                index=self.state_matrix.index,
                columns=self.state_matrix.columns
            )
            self.control_matrix = pd.DataFrame(
                data=control_matrix_discrete,
                index=self.control_matrix.index,
                columns=self.control_matrix.columns
            )
            self.disturbance_matrix = pd.DataFrame(
                data=disturbance_matrix_discrete,
                index=self.disturbance_matrix.index,
                columns=self.disturbance_matrix.columns
            )

        # Define initial state.
        define_initial_state()

        # Calculate parameters / coefficients.
        calculate_coefficients_zone()
        calculate_coefficients_surface()
        calculate_coefficients_radiator()

        # Define heat, CO2 and humidity transfers.
        define_heat_transfer_surfaces_exterior()
        define_heat_transfer_surfaces_interior()
        define_heat_transfer_surfaces_adiabatic()
        define_heat_transfer_infiltration()
        define_heat_transfer_internal_gains()
        define_heat_transfer_hvac_generic()
        define_heat_transfer_hvac_radiator()
        define_heat_transfer_hvac_ahu()
        define_heat_transfer_hvac_tu()
        define_co2_transfer_hvac_ahu()
        define_heat_transfer_window_air_flow()
        define_humidity_transfer_hvac_ahu()
        define_storage_state_of_charge()

        # Define outputs.
        define_output_zone_temperature()
        define_output_zone_co2_concentration()
        define_output_zone_humidity()
        define_output_hvac_generic_power()
        define_output_hvac_radiator_power()
        define_output_hvac_ahu_power()
        define_output_hvac_tu_power()
        define_output_fresh_air_flow()
        define_output_window_fresh_air_flow()
        define_output_ahu_fresh_air_flow()
        define_output_storage_state_of_charge()
        define_output_storage_charge()
        define_output_storage_discharge()
        define_output_grid()

        # Define validation outputs.
        if with_validation_outputs:
            define_output_surface_temperature()
            define_output_surfaces_exterior_irradiation_gain_exterior()
            define_output_surfaces_exterior_convection_interior()

        # Define timeseries.
        define_disturbance_timeseries()
        define_electricity_price_timeseries()
        define_electricity_price_distribution_timeseries()
        define_output_constraint_timeseries()

        # Convert to time discrete model.
        discretize_model()

    def simulate(
            self,
            state_initial,
            control_vector,
            disturbance_timeseries=None
    ):
        """Simulate building model with given initial state and control timeseries.

        - Time horizon is derived from scenario definition in database.
        - Disturbance timeseries is derived from database.
        - TODO: Automatically create initial state from database.
        """
        # Default values
        if disturbance_timeseries is None:
            disturbance_timeseries = self.disturbance_timeseries

        # Initialize state and output timeseries
        state_vector = pd.DataFrame(
            np.nan,
            self.timesteps,
            self.states
        )
        state_vector.iloc[0, :] = state_initial
        output_vector = pd.DataFrame(
            np.nan,
            self.timesteps,
            self.outputs
        )

        # Iterative simulation of state space equations
        # TODO: Check `timestep + 1` (This was added to match with EnergyPlus outputs)
        for timestep in range(len(self.timesteps) - 1):
            state_vector.iloc[timestep + 1, :] = (
                np.dot(self.state_matrix.values, state_vector.iloc[timestep, :].values)
                + np.dot(self.control_matrix.values, control_vector.iloc[timestep, :].values)
                + np.dot(self.disturbance_matrix.values, disturbance_timeseries.iloc[timestep + 1, :].values)
            )
        for timestep in range(len(self.timesteps) - 1):
            output_vector.iloc[timestep, :] = (
                np.dot(self.state_output_matrix.values, state_vector.iloc[timestep, :].values)
                + np.dot(self.control_output_matrix.values, control_vector.iloc[timestep, :].values)
                + np.dot(self.disturbance_output_matrix.values, disturbance_timeseries.iloc[timestep + 1, :].values)
            )

        return (
            state_vector,
            output_vector
        )
