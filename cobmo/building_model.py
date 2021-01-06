"""Building model module."""

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.interpolate
import typing

import cobmo.config
import cobmo.data_interface
import cobmo.utils

logger = cobmo.config.get_logger(__name__)


class BuildingModel(object):
    """Building model object consisting of the state space model for the given scenario. The model includes
    index sets for states / controls / disturbances / outputs, the state / control / disturbance / state-output /
    control-output / disturbance-output matrices and disturbance / electricity price / output constraint timeseries.

    - The building model object constructs the state space model matrices and index sets
      according to the building model equations which are documented CoBMo's technical documentation.
    - The required `building_data` object for the given scenario is obtained from the database
      through `cobmo.data_interface`.
    - The building can be connected to the electric grid, the thermal grid or both, which is controlled through the
      keyword arguments `connect_electric_grid` / `connect_thermal_grid_cooling` / `connect_thermal_grid_heating`
      as explained below.

    Syntax
        ``BuildingModel(scenario_name)``: Instantiate building model for given `scenario_name`.

    Parameters:
        scenario_name (str): CoBMo building scenario name, as defined in the data table `scenarios`.

    Keyword Arguments:
        timestep_start (pd.Timestamp): If provided, will used in place of `timestep_start` from the scenario definition.
        timestep_end (pd.Timestamp): If provided, will used in place of `timestep_end` from the scenario definition.
        timestep_interval (pd.Timedelta): If provided, will used in place of `timestep_interval` from the scenario definition.
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
        states (pd.Index): Index set of the state variables.
        controls (pd.Index): Index set of the control variables.
        disturbances (pd.Index): Index set of the disturbance variables.
        outputs (pd.Index): Index set of the output variables.
        timesteps (pd.Index): Index set of the timesteps.
        timestep_interval (pd.Timedelta): Timestep interval, assuming a constant interval between all timesteps.
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
        output_minimum_timeseries (pd.DataFrame): Minimum output constraint timeseries.
        output_maximum_timeseries (pd.DataFrame): Maximum output constraint timeseries.
    """

    scenario_name: str
    states: pd.Index
    controls: pd.Index
    disturbances: pd.Index
    outputs: pd.Index
    timesteps: pd.Index
    timestep_interval: pd.Timedelta
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
    output_minimum_timeseries: pd.DataFrame
    output_maximum_timeseries: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            timestep_start=None,
            timestep_end=None,
            timestep_interval=None,
            connect_electric_grid=True,
            connect_thermal_grid_cooling=False,
            connect_thermal_grid_heating=False,
            with_validation_outputs=False
    ):

        # Store scenario name.
        self.scenario_name = scenario_name

        # Obtain building data.
        building_data = (
            cobmo.data_interface.BuildingData(
                self.scenario_name,
                timestep_start=timestep_start,
                timestep_end=timestep_end,
                timestep_interval=timestep_interval
            )
        )

        # Store building data.
        self.building_data = building_data

        # Obtain total building zone area.
        # - This is used for scaling air flow / power values to per-square-meter values.
        self.zone_area_total = building_data.zones.loc[:, 'zone_area'].sum()

        # Add constant timeseries in disturbance vector, if any CO2 model or HVAC or window.
        self.define_constant = (
            (building_data.zones['fresh_air_flow_control_type'] == 'co2_based').any()
            | pd.notnull(building_data.zones['hvac_ahu_type']).any()
        )

        # Define sets.

        # State variables.
        self.states = pd.Index(
            pd.concat([
                # Zone temperature.
                building_data.zones['zone_name'] + '_temperature',

                # Surface temperature.
                building_data.surfaces_adiabatic['surface_name'][
                    building_data.surfaces_adiabatic['heat_capacity'] != 0.0
                ] + '_temperature',
                building_data.surfaces_exterior['surface_name'][
                    building_data.surfaces_exterior['heat_capacity'] != 0.0
                ] + '_temperature',
                building_data.surfaces_interior['surface_name'][
                    building_data.surfaces_interior['heat_capacity'] != 0.0
                ] + '_temperature',

                # Zone CO2 concentration.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                    & (building_data.zones['fresh_air_flow_control_type'] == 'co2_based')
                ] + '_co2_concentration',

                # Zone absolute humidity.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                    & (building_data.zones['humidity_control_type'] == 'humidity_based')
                ] + '_absolute_humidity',

                # Radiator temperatures.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                ] + '_radiator_water_mean_temperature',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                ] + '_radiator_hull_front_temperature',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                ] + '_radiator_hull_rear_temperature',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                    & (building_data.zones['radiator_panel_number'] == '2')
                ] + '_radiator_panel_1_hull_rear_temperature',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                    & (building_data.zones['radiator_panel_number'] == '2')
                ] + '_radiator_panel_2_hull_front_temperature',

                # Storage state of charge.
                pd.Series(['storage_state_of_charge']) if (
                    pd.notnull(building_data.scenarios['storage_type'])
                ) else None
            ]),
            name='state_name'
        )

        # Control variables.
        self.controls = pd.Index(
            pd.concat([
                # Electric / thermal grid connections and heating / cooling plants.
                pd.Series([
                    'grid_electric_power',
                    'grid_thermal_power_cooling',
                    'grid_thermal_power_heating',
                    'plant_thermal_power_cooling',
                    'plant_thermal_power_heating'
                ]),

                # Generic HVAC system.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_generic_type'])
                ] + '_generic_heat_thermal_power',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_generic_type'])
                ] + '_generic_cool_thermal_power',

                # Radiator thermal power.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                ] + '_radiator_thermal_power',

                # AHU.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_air_flow',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_air_flow',

                # TU.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_tu_type'])
                ] + '_tu_heat_air_flow',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_tu_type'])
                ] + '_tu_cool_air_flow',

                # Sensible storage cooling.
                pd.Series([
                    'storage_charge_thermal_power_cooling',
                    'storage_discharge_thermal_power_cooling',
                ]) if (
                    building_data.scenarios['storage_commodity_type'] == 'sensible_cooling'
                ) else None,

                # Sensible storage heating.
                pd.Series([
                    'storage_charge_thermal_power_heating',
                    'storage_discharge_thermal_power_heating',
                ]) if (
                    building_data.scenarios['storage_commodity_type'] == 'sensible_heating'
                ) else None,

                # Battery storage.
                pd.Series([
                    'storage_charge_electric_power',
                    'storage_discharge_electric_power'
                ]) if (
                    building_data.scenarios['storage_commodity_type'] == 'battery'
                ) else None
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
                    building_data.zones['internal_gain_type'].dropna().unique() + '_internal_gain_occupancy'
                ),
                pd.Series(
                    building_data.zones['internal_gain_type'].dropna().unique() + '_internal_gain_appliances'
                ),

                # Constant (workaround for constant model terms).
                (pd.Series(['constant']) if self.define_constant else None)
            ]),
            name='disturbance_name'
        )

        # Output variables.
        self.outputs = pd.Index(
            pd.concat([
                # Electric / thermal grid connections and heating / cooling plants.
                pd.Series([
                    'grid_electric_power',
                    'grid_thermal_power_cooling',
                    'grid_thermal_power_heating',
                    'plant_thermal_power_cooling',
                    'plant_thermal_power_heating'
                ]),

                # Generic HVAC system controls.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_generic_type'])
                ] + '_generic_heat_thermal_power',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_generic_type'])
                ] + '_generic_cool_thermal_power',

                # Radiator controls.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_radiator_type'])
                ] + '_radiator_thermal_power',

                # AHU controls.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                ] + '_ahu_heat_air_flow',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                ] + '_ahu_cool_air_flow',

                # TU controls.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_tu_type'])
                ] + '_tu_heat_air_flow',
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_tu_type'])
                ] + '_tu_cool_air_flow',

                # Sensible storage cooling controls.
                pd.Series([
                    'storage_charge_thermal_power_cooling',
                    'storage_discharge_thermal_power_cooling',
                ]) if (
                    building_data.scenarios['storage_commodity_type'] == 'sensible_cooling'
                ) else None,

                # Sensible storage heating controls.
                pd.Series([
                    'storage_charge_thermal_power_heating',
                    'storage_discharge_thermal_power_heating',
                ]) if (
                    building_data.scenarios['storage_commodity_type'] == 'sensible_heating'
                ) else None,

                # Battery storage controls.
                pd.Series([
                    'storage_charge_electric_power',
                    'storage_discharge_electric_power'
                ]) if (
                    building_data.scenarios['storage_commodity_type'] == 'battery'
                ) else None,

                # Storage state of charge.
                pd.Series(['storage_state_of_charge']) if (
                    pd.notnull(building_data.scenarios['storage_type'])
                ) else None,

                # Zone temperature.
                building_data.zones['zone_name'] + '_temperature',

                # Zone CO2 concentration.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                    & (building_data.zones['fresh_air_flow_control_type'] == 'co2_based')
                ] + '_co2_concentration',

                # Zone absolute humidity.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                    & (building_data.zones['humidity_control_type'] == 'humidity_based')
                ] + '_absolute_humidity',

                # Zone fresh air flow.
                building_data.zones['zone_name'][
                    pd.notnull(building_data.zones['hvac_ahu_type'])
                ] + '_total_fresh_air_flow',

                # Validation outputs.
                pd.concat([
                    building_data.surfaces_adiabatic['surface_name'][
                        building_data.surfaces_adiabatic['heat_capacity'] != 0.0
                    ] + '_temperature',
                    building_data.surfaces_exterior['surface_name'][
                        building_data.surfaces_exterior['heat_capacity'] != 0.0
                    ] + '_temperature',
                    building_data.surfaces_interior['surface_name'][
                        building_data.surfaces_interior['heat_capacity'] != 0.0
                    ] + '_temperature',
                    building_data.surfaces_exterior['surface_name'] + '_irradiation_gain_exterior',
                    building_data.surfaces_exterior['surface_name'] + '_convection_interior'
                ]) if with_validation_outputs else None,

                # Power balances.
                pd.Series([
                    'electric_power_balance',
                    'thermal_power_cooling_balance',
                    'thermal_power_heating_balance'
                ])
            ]),
            name='output_name'
        )

        # Obtain timesteps.
        self.timesteps = building_data.timesteps
        self.timestep_interval = building_data.timestep_interval

        # Instantiate state space model matrix constructors.
        state_matrix = cobmo.utils.MatrixConstructor(index=self.states, columns=self.states)
        control_matrix = cobmo.utils.MatrixConstructor(index=self.states, columns=self.controls)
        disturbance_matrix = cobmo.utils.MatrixConstructor(index=self.states, columns=self.disturbances)
        state_output_matrix = cobmo.utils.MatrixConstructor(index=self.outputs, columns=self.states)
        control_output_matrix = cobmo.utils.MatrixConstructor(index=self.outputs, columns=self.controls)
        disturbance_output_matrix = cobmo.utils.MatrixConstructor(index=self.outputs, columns=self.disturbances)

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
                self.state_vector_initial.index.isin(building_data.zones['zone_name'] + '_temperature')
            ] = (
                building_data.scenarios['initial_zone_temperature']
            )

            # Surface temperature.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(
                    pd.concat([
                        building_data.surfaces_adiabatic['surface_name'] + '_temperature',
                        building_data.surfaces_exterior['surface_name'] + '_temperature',
                        building_data.surfaces_interior['surface_name'] + '_temperature'
                    ])
                )
            ] = (
                building_data.scenarios['initial_surface_temperature']
            )

            # CO2 concentration.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(building_data.zones['zone_name'] + '_co2_concentration')
            ] = (
                building_data.scenarios['initial_co2_concentration']
            )

            # Zone air absolute humidity.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.isin(building_data.zones['zone_name'] + '_absolute_humidity')
            ] = (
                building_data.scenarios['initial_absolute_humidity']
            )

            # Sensible storage state of charge.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.str.contains('_storage_state_of_charge')
            ] = (
                building_data.scenarios['initial_storage_state_of_charge']
            )

            # Battery storage state of charge.
            self.state_vector_initial.loc[
                self.state_vector_initial.index.str.contains('_storage_state_of_charge')
            ] = (
                building_data.scenarios['initial_storage_state_of_charge']
            )

        def calculate_coefficients_zone():
            """Calculate zone parameters / heat transfer coefficients for use in, e.g., surface and radiator models."""

            # Calculate absolute heat capacity from specific heat capacity.
            building_data.zones['heat_capacity'] = (
                building_data.zones['zone_area']
                * building_data.zones['zone_height']
                * building_data.zones['heat_capacity']
            )

            # Instantiate columns for parameters / heat transfer coefficients.
            building_data.zones['zone_surfaces_wall_area'] = None
            building_data.zones['zone_surfaces_window_area'] = None
            building_data.zones['zone_surfaces_wall_emissivity'] = None
            building_data.zones['zone_surfaces_window_emissivity'] = None

            # Calculate zone parameters / heat transfer coefficients.
            for zone_name, zone_data in building_data.zones.iterrows():
                # Collect all surfaces adjacent to the zone.
                zone_surfaces = (
                    pd.concat(
                        [
                            building_data.surfaces_exterior.loc[
                                building_data.surfaces_exterior['zone_name'].isin([zone_name]),
                                :
                            ],
                            building_data.surfaces_interior.loc[
                                building_data.surfaces_interior['zone_name'].isin([zone_name]),
                                :
                            ],
                            building_data.surfaces_interior.loc[
                                building_data.surfaces_interior['zone_adjacent_name'].isin([zone_name]),
                                :
                            ],
                            building_data.surfaces_adiabatic.loc[
                                building_data.surfaces_adiabatic['zone_name'].isin([zone_name]),
                                :
                            ]
                        ],
                        sort=False
                    )
                )

                # Calculate parameters / heat transfer coefficients.
                building_data.zones.at[zone_name, 'zone_surfaces_wall_area'] = (
                    (
                        zone_surfaces['surface_area']
                        * (1 - zone_surfaces['window_wall_ratio'])
                    ).sum()
                )
                building_data.zones.at[zone_name, 'zone_surfaces_window_area'] = (
                    (
                        zone_surfaces['surface_area']
                        * zone_surfaces['window_wall_ratio']
                    ).sum()
                )
                building_data.zones.at[zone_name, 'zone_surfaces_wall_emissivity'] = (
                    zone_surfaces['emissivity_surface'].mean()
                )
                # TODO: Ignore surfaces with no windows.
                building_data.zones.at[zone_name, 'zone_surfaces_window_emissivity'] = (
                    zone_surfaces['emissivity_window'].mean()
                )

        def calculate_coefficients_surface():
            """Calculate heat transfer coefficients for the surface models."""

            # Calculate absolute heat capacity from specific heat capacity.
            building_data.surfaces_adiabatic['heat_capacity'] = (
                building_data.surfaces_adiabatic['surface_area']
                * building_data.surfaces_adiabatic['heat_capacity']
            )
            building_data.surfaces_exterior['heat_capacity'] = (
                building_data.surfaces_exterior['surface_area']
                * building_data.surfaces_exterior['heat_capacity']
            )
            building_data.surfaces_interior['heat_capacity'] = (
                building_data.surfaces_interior['surface_area']
                * building_data.surfaces_interior['heat_capacity']
            )

            # Instantiate columns for heat transfer coefficients.
            building_data.surfaces_exterior['heat_transfer_coefficient_surface_sky'] = None
            building_data.surfaces_exterior['heat_transfer_coefficient_surface_ground'] = None
            building_data.surfaces_exterior['heat_transfer_coefficient_window_sky'] = None
            building_data.surfaces_exterior['heat_transfer_coefficient_window_ground'] = None

            # Calculate heat transfer coefficients.
            for surface_name, surface_data in building_data.surfaces_exterior.iterrows():
                building_data.surfaces_exterior.at[
                    surface_name,
                    'heat_transfer_coefficient_surface_sky'
                ] = (
                    4.0
                    * building_data.parameters['stefan_boltzmann_constant']
                    * surface_data['emissivity_surface']
                    * surface_data['sky_view_factor']
                    * (
                        building_data.scenarios['linearization_exterior_surface_temperature']
                        / 2.0
                        + building_data.scenarios['linearization_sky_temperature']
                        / 2.0
                        + 273.15
                    ) ** 3
                )
                building_data.surfaces_exterior.at[
                    surface_name,
                    'heat_transfer_coefficient_surface_ground'
                ] = (
                    4.0
                    * building_data.parameters['stefan_boltzmann_constant']
                    * surface_data['emissivity_surface']
                    * (1.0 - surface_data['sky_view_factor'])
                    * (
                        building_data.scenarios['linearization_exterior_surface_temperature']
                        / 2.0
                        + building_data.scenarios['linearization_ambient_air_temperature']
                        / 2.0
                        + 273.15
                    ) ** 3
                )
                if pd.notnull(surface_data['window_type']):
                    building_data.surfaces_exterior.at[
                        surface_name,
                        'heat_transfer_coefficient_window_sky'
                    ] = (
                        4.0
                        * building_data.parameters['stefan_boltzmann_constant']
                        * surface_data['emissivity_window']
                        * surface_data['sky_view_factor']
                        * (
                            building_data.scenarios['linearization_exterior_surface_temperature']
                            / 2.0
                            + building_data.scenarios['linearization_sky_temperature']
                            / 2.0
                            + 273.15
                        ) ** 3
                    )
                    building_data.surfaces_exterior.at[
                        surface_name,
                        'heat_transfer_coefficient_window_ground'
                    ] = (
                        4.0
                        * building_data.parameters['stefan_boltzmann_constant']
                        * surface_data['emissivity_window']
                        * (1.0 - surface_data['sky_view_factor'])
                        * (
                            building_data.scenarios['linearization_exterior_surface_temperature']
                            / 2.0
                            + building_data.scenarios['linearization_ambient_air_temperature']
                            / 2.0
                            + 273.15
                        ) ** 3
                    )

        def calculate_coefficients_radiator():
            """Calculate heat transfer coefficients for the radiator model."""

            if pd.notnull(building_data.zones['hvac_radiator_type']).any():
                # Instantiate columns for heat transfer coefficients.
                building_data.zones['heat_capacitance_hull'] = None
                building_data.zones['thermal_resistance_radiator_hull_conduction'] = None
                building_data.zones['thermal_resistance_radiator_front_zone'] = None
                building_data.zones['thermal_resistance_radiator_front_surfaces'] = None
                building_data.zones['thermal_resistance_radiator_front_zone_surfaces'] = None
                building_data.zones['thermal_resistance_radiator_rear_zone'] = None
                building_data.zones['thermal_resistance_radiator_rear_surfaces'] = None
                building_data.zones['thermal_resistance_radiator_rear_zone_surfaces'] = None

                # Instantiate additional columns for multi-panel radiators.
                if (building_data.zones['radiator_panel_number'] == '2').any():
                    building_data.zones['thermal_resistance_radiator_panel_1_rear_zone'] = None
                    building_data.zones['thermal_resistance_radiator_panel_2_front_zone'] = None

                # Calculate heat transfer coefficients.
                for zone_name, zone_data in building_data.zones.iterrows():
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
                        building_data.zones.at[zone_name, 'heat_capacitance_hull'] = (
                            radiator_hull_volume
                            * zone_data['radiator_hull_heat_capacity']
                        )
                        building_data.zones.at[zone_name, 'heat_capacitance_water'] = (
                            zone_data['radiator_water_volume']
                            * building_data.parameters['water_specific_heat']
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
                                + building_data.scenarios['linearization_surface_temperature']
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
                                    4.0 * building_data.parameters['stefan_boltzmann_constant']
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
                                    4.0 * building_data.parameters['stefan_boltzmann_constant']
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
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_hull_conduction'] = (
                            thermal_resistance_conduction
                        )
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_front_zone'] = (
                            thermal_resistance_star_sum_front / thermal_resistance_radiation_front
                        )
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_front_surfaces'] = (
                            thermal_resistance_star_sum_front / thermal_resistance_convection
                        )
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_front_zone_surfaces'] = (
                            thermal_resistance_star_sum_front / (0.5 * thermal_resistance_conduction)
                        )
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_rear_zone'] = (
                            thermal_resistance_star_sum_rear / thermal_resistance_radiation_rear
                        )
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_rear_surfaces'] = (
                            thermal_resistance_star_sum_rear / thermal_resistance_convection
                        )
                        building_data.zones.at[zone_name, 'thermal_resistance_radiator_rear_zone_surfaces'] = (
                            thermal_resistance_star_sum_rear / (0.5 * thermal_resistance_conduction)
                        )

                        if (building_data.zones['radiator_panel_number'] == '2').any():
                            thermal_resistance_convection_fin = (
                                1.0
                                / (
                                    thermal_resistance_convection
                                    * zone_data['radiator_fin_effectiveness']
                                )
                            )

                            building_data.zones.at[zone_name, 'thermal_resistance_radiator_panel_1_rear_zone'] = (
                                0.5 * thermal_resistance_conduction
                                + thermal_resistance_convection
                            )
                            building_data.zones.at[zone_name, 'thermal_resistance_radiator_panel_2_front_zone'] = (
                                0.5 * thermal_resistance_conduction
                                + thermal_resistance_convection_fin
                            )

        def define_heat_transfer_surfaces_exterior():
            """Thermal model: Exterior surfaces"""
            # TODO: Exterior window transmission factor

            for surface_name, surface_data in building_data.surfaces_exterior.iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    # Conductive heat transfer from the exterior towards the core of surface
                    disturbance_matrix[
                        f'{surface_name}_temperature',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_surface']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    disturbance_matrix[
                        f'{surface_name}_temperature',
                        'ambient_air_temperature'
                    ] += (
                        (
                            building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_surface_ground']
                        )
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    disturbance_matrix[
                        f'{surface_name}_temperature',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_surface_sky']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    state_matrix[
                        f'{surface_name}_temperature',
                        f'{surface_name}_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )

                    # Conductive heat transfer from the interior towards the core of surface
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_matrix[
                            f'{surface_name}_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )
                    state_matrix[
                        f'{surface_name}_temperature',
                        f'{surface_name}_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    state_matrix[
                        f'{surface_name}_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )

                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_matrix[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1))
                            / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )
                    state_matrix[
                        surface_data['zone_name'] + '_temperature',
                        f'{surface_name}_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    state_matrix[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                else:  # Surfaces with neglected heat capacity
                    # Complete convective heat transfer from surface to zone
                    disturbance_matrix[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_surface']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    disturbance_matrix[
                        surface_data['zone_name'] + '_temperature',
                        'ambient_air_temperature'
                    ] += (
                        (
                            building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_surface_ground']
                        )
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    disturbance_matrix[
                        surface_data['zone_name'] + '_temperature',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_surface_sky']
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    state_matrix[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            + 1.0
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + 1.0
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_matrix[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + (
                                    building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / (surface_data['heat_transfer_coefficient_conduction_surface'])
                            ) ** (- 1))
                            / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )

                # Windows for each exterior surface - Modelled as surfaces with neglected heat capacity
                if surface_data['window_wall_ratio'] != 0.0:
                    # Complete convective heat transfer from surface to zone
                    disturbance_matrix[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_window']
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / surface_data['heat_transfer_coefficient_conduction_window']
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    disturbance_matrix[
                        surface_data['zone_name'] + '_temperature',
                        'ambient_air_temperature'
                    ] += (
                        (
                            building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_window_ground']
                        )
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / surface_data['heat_transfer_coefficient_conduction_window']
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    disturbance_matrix[
                        surface_data['zone_name'] + '_temperature',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_window_sky']
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            / surface_data['heat_transfer_coefficient_conduction_window']
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    state_matrix[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * surface_data['window_wall_ratio']
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_window_ground']
                                + surface_data['heat_transfer_coefficient_window_sky']
                            )
                            + 1.0
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + 1.0
                            / surface_data['heat_transfer_coefficient_conduction_window']
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_matrix[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_window']
                            * surface_data['surface_area']
                            * surface_data['window_wall_ratio']
                            * (1.0 - (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_window_ground']
                                    + surface_data['heat_transfer_coefficient_window_sky']
                                )
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + (
                                    building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_window_ground']
                                    + surface_data['heat_transfer_coefficient_window_sky']
                                )
                                / surface_data['heat_transfer_coefficient_conduction_window']
                            ) ** (- 1))
                            / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )

        def define_heat_transfer_surfaces_interior():
            """Thermal model: Interior surfaces"""

            for surface_name, surface_data in building_data.surfaces_interior.iterrows():
                for zone_name in [surface_data['zone_name'], surface_data['zone_adjacent_name']]:
                    if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                        # Conductive heat transfer from the interior towards the core of surface
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                building_data.surfaces_exterior[
                                    building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            disturbance_matrix[
                                f'{surface_name}_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_surface']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (
                                    1.0
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / (
                                        2.0
                                        * surface_data['heat_transfer_coefficient_conduction_surface']
                                    )
                                ) ** (- 1)
                                / surface_data['heat_capacity']
                            )
                        state_matrix[
                            f'{surface_name}_temperature',
                            f'{surface_name}_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )
                        state_matrix[
                            f'{surface_name}_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )

                        # Convective heat transfer from the surface towards zone
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                building_data.surfaces_exterior[
                                    building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            disturbance_matrix[
                                f'{zone_name}_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_surface']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (1.0 - (
                                    1.0
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / (
                                        2.0
                                        * surface_data['heat_transfer_coefficient_conduction_surface']
                                    )
                                ) ** (- 1))
                                / building_data.zones.at[zone_name, 'heat_capacity']
                            )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{surface_name}_temperature'
                        ] += (
                            surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1)
                            / building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1)
                            / building_data.zones.at[zone_name, 'heat_capacity']
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
                                    building_data.surfaces_exterior[
                                        building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                    ],
                                    building_data.surfaces_interior[
                                        building_data.surfaces_interior['zone_name'] == zone_adjacent_name
                                    ],
                                    building_data.surfaces_interior[
                                        building_data.surfaces_interior['zone_adjacent_name'] == zone_adjacent_name
                                    ],
                                    building_data.surfaces_adiabatic[
                                        building_data.surfaces_adiabatic['zone_name'] == zone_adjacent_name
                                    ]
                                ],
                                sort=False
                            ).iterrows()  # For all surfaces adjacent to the zone
                        )

                        # Complete convective heat transfer from adjacent zone to zone
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                building_data.surfaces_exterior[
                                    building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            disturbance_matrix[
                                f'{zone_name}_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / zone_adjacent_surface_area
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_surface']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (
                                    1.0
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['heat_transfer_coefficient_conduction_surface']
                                ) ** (- 1)
                                / building_data.zones.at[zone_name, 'heat_capacity']
                            )
                        state_matrix[
                            f'{zone_name}_temperature',
                            zone_adjacent_name + '_temperature'
                        ] += (
                            surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['heat_transfer_coefficient_conduction_surface']
                            ) ** (- 1)
                            / building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['heat_transfer_coefficient_conduction_surface']
                            ) ** (- 1)
                            / building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                building_data.surfaces_exterior[
                                    building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            disturbance_matrix[
                                f'{zone_name}_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_surface']
                                * surface_data['surface_area']
                                * (1 - surface_data['window_wall_ratio'])
                                * (1.0 - (
                                    1.0
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['heat_transfer_coefficient_conduction_surface']
                                ) ** (- 1))
                                / building_data.zones.at[zone_name, 'heat_capacity']
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
                                    building_data.surfaces_exterior[
                                        building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                    ],
                                    building_data.surfaces_interior[
                                        building_data.surfaces_interior['zone_name'] == zone_adjacent_name
                                    ],
                                    building_data.surfaces_interior[
                                        building_data.surfaces_interior['zone_adjacent_name'] == zone_adjacent_name
                                    ],
                                    building_data.surfaces_adiabatic[
                                        building_data.surfaces_adiabatic['zone_name'] == zone_adjacent_name
                                    ]
                                ],
                                sort=False
                            ).iterrows()  # For all surfaces adjacent to the zone
                        )

                        # Complete convective heat transfer from adjacent zone to zone
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                building_data.surfaces_exterior[
                                    building_data.surfaces_exterior['zone_name'] == zone_adjacent_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            disturbance_matrix[
                                f'{zone_name}_temperature',
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
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['heat_transfer_coefficient_conduction_window']
                                ) ** (- 1)
                                / building_data.zones.at[zone_name, 'heat_capacity']
                            )
                        state_matrix[
                            f'{zone_name}_temperature',
                            zone_adjacent_name + '_temperature'
                        ] += (
                            surface_data['surface_area']
                            * surface_data['window_wall_ratio']
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['heat_transfer_coefficient_conduction_window']
                            ) ** (- 1)
                            / building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            - 1.0
                            * surface_data['surface_area']
                            * surface_data['window_wall_ratio']
                            * (
                                1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + 1.0
                                / surface_data['heat_transfer_coefficient_conduction_window']
                            ) ** (- 1)
                            / building_data.zones.at[zone_name, 'heat_capacity']
                        )
                        for zone_exterior_surface_name, zone_exterior_surface_data in (
                                building_data.surfaces_exterior[
                                    building_data.surfaces_exterior['zone_name'] == zone_name
                                ].iterrows()
                        ):
                            # Interior irradiation through all exterior surfaces adjacent to the zone
                            disturbance_matrix[
                                f'{zone_name}_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ] += (
                                (
                                    zone_exterior_surface_data['surface_area']
                                    * zone_exterior_surface_data['window_wall_ratio']
                                    / building_data.zones.at[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * surface_data['absorptivity_window']
                                * surface_data['surface_area']
                                * surface_data['window_wall_ratio']
                                * (1.0 - (
                                    1.0
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    + building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    / surface_data['heat_transfer_coefficient_conduction_window']
                                ) ** (- 1))
                                / building_data.zones.at[zone_name, 'heat_capacity']
                            )

        def define_heat_transfer_surfaces_adiabatic():
            """Thermal model: Adiabatic surfaces"""

            for surface_name, surface_data in building_data.surfaces_adiabatic.iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    # Conductive heat transfer from the interior towards the core of surface
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_matrix[
                            f'{surface_name}_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1)
                            / surface_data['heat_capacity']
                        )
                    state_matrix[
                        f'{surface_name}_temperature',
                        f'{surface_name}_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )
                    state_matrix[
                        f'{surface_name}_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / surface_data['heat_capacity']
                    )

                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_matrix[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / building_data.zones.at[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_interior_convection']
                                )
                                / (
                                    2.0
                                    * surface_data['heat_transfer_coefficient_conduction_surface']
                                )
                            ) ** (- 1))
                            / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                        )
                    state_matrix[
                        surface_data['zone_name'] + '_temperature',
                        f'{surface_name}_temperature'
                    ] += (
                        surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                    state_matrix[
                        surface_data['zone_name'] + '_temperature',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * surface_data['surface_area']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                        / building_data.zones.at[surface_data['zone_name'], 'heat_capacity']
                    )
                else:  # Surfaces with neglected heat capacity
                    logger.warning(f"Adiabatic surfaces with zero heat capacity have no effect: {surface_name}")

        def define_heat_transfer_infiltration():

            for zone_name, zone_data in building_data.zones.iterrows():
                state_matrix[
                    f'{zone_name}_temperature',
                    f'{zone_name}_temperature'
                ] += (
                    - zone_data['infiltration_rate']
                    * building_data.parameters['heat_capacity_air']
                    * zone_data['zone_area']
                    * zone_data['zone_height']
                    / zone_data['heat_capacity']
                )
                disturbance_matrix[
                    f'{zone_name}_temperature',
                    'ambient_air_temperature'
                ] += (
                    zone_data['infiltration_rate']
                    * building_data.parameters['heat_capacity_air']
                    * zone_data['zone_area']
                    * zone_data['zone_height']
                    / zone_data['heat_capacity']
                )

        def define_heat_transfer_internal_gains():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data.at['internal_gain_type']):
                    disturbance_matrix[
                        f'{zone_name}_temperature',
                        zone_data.at['internal_gain_type'] + '_internal_gain_occupancy'
                    ] += (
                        zone_data.at['occupancy_density']
                        * zone_data.at['occupancy_heat_gain']
                        * zone_data.at['zone_area']
                        / zone_data.at['heat_capacity']
                    )
                    disturbance_matrix[
                        f'{zone_name}_temperature',
                        zone_data.at['internal_gain_type'] + '_internal_gain_appliances'
                    ] += (
                        zone_data.at['appliances_heat_gain']
                        * zone_data.at['zone_area']
                        / zone_data.at['heat_capacity']
                    )

        def define_heat_transfer_hvac_generic():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_generic_type']):
                    control_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_generic_heat_thermal_power'
                    ] += (
                        1.0
                        * zone_data['zone_area']
                        / zone_data['heat_capacity']
                    )
                    control_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_generic_cool_thermal_power'
                    ] += (
                        - 1.0
                        * zone_data['zone_area']
                        / zone_data['heat_capacity']
                    )

        def define_heat_transfer_hvac_radiator():
            """Define state equations describing the heat transfer occurring due to radiators."""

            if pd.notnull(building_data.zones['hvac_radiator_type']).any():
                for zone_name, zone_data in building_data.zones.iterrows():
                    if pd.notnull(zone_data['hvac_radiator_type']):
                        # Thermal power input to water.
                        control_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_thermal_power'
                        ] += (
                            1.0
                            * zone_data['zone_area']
                            / zone_data['heat_capacitance_water']
                        )

                        # Heat transfer between radiator hull front and water.
                        state_matrix[
                            f'{zone_name}_radiator_hull_front_temperature',
                            f'{zone_name}_radiator_hull_front_temperature'
                        ] += (
                            - 1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_hull_front_temperature',
                            f'{zone_name}_radiator_water_mean_temperature'
                        ] += (
                            1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_hull_front_temperature'
                        ] += (
                            1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_water']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_water_mean_temperature'
                        ] += (
                            - 1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_water']
                        )

                    if zone_data['radiator_panel_number'] == '2':
                        # Heat transfer between radiator panel 1 hull rear and water.
                        state_matrix[
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature',
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature'
                        ] += (
                            - 1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature',
                            f'{zone_name}_radiator_water_mean_temperature'
                        ] += (
                            1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature'
                        ] += (
                            1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_water']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_water_mean_temperature'
                        ] += (
                            - 1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_water']
                        )

                        # Heat transfer between radiator panel 2 hull front and water.
                        state_matrix[
                            f'{zone_name}_radiator_panel_2_hull_front_temperature',
                            f'{zone_name}_radiator_panel_2_hull_front_temperature'
                        ] += (
                            - 1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_panel_2_hull_front_temperature',
                            f'{zone_name}_radiator_water_mean_temperature'
                        ] += (
                            1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_panel_2_hull_front_temperature'
                        ] += (
                            1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_water']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_water_mean_temperature',
                            f'{zone_name}_radiator_water_mean_temperature'
                        ] += (
                            - 1.0
                            / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                            / zone_data['heat_capacitance_water']
                        )

                    # Heat transfer between radiator hull rear and water.
                    state_matrix[
                        f'{zone_name}_radiator_hull_rear_temperature',
                        f'{zone_name}_radiator_hull_rear_temperature'
                    ] += (
                        - 1.0
                        / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_radiator_hull_rear_temperature',
                        f'{zone_name}_radiator_water_mean_temperature'
                    ] += (
                        1.0
                        / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_radiator_water_mean_temperature',
                        f'{zone_name}_radiator_hull_rear_temperature'
                    ] += (
                        1.0
                        / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                        / zone_data['heat_capacitance_water']
                    )
                    state_matrix[
                        f'{zone_name}_radiator_water_mean_temperature',
                        f'{zone_name}_radiator_water_mean_temperature'
                    ] += (
                        - 1.0
                        / (0.5 * zone_data['thermal_resistance_radiator_hull_conduction'])
                        / zone_data['heat_capacitance_water']
                    )

                    # Heat transfer between radiator hull front and zone air.
                    state_matrix[
                        f'{zone_name}_radiator_hull_front_temperature',
                        f'{zone_name}_radiator_hull_front_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_front_zone']
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_radiator_hull_front_temperature',
                        f'{zone_name}_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_front_zone']
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_radiator_hull_front_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_front_zone']
                        / zone_data['heat_capacity']
                    )
                    state_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_front_zone']
                        / zone_data['heat_capacity']
                    )

                    if zone_data['radiator_panel_number'] == '2':
                        # Heat transfer between radiator panel 1 hull rear and zone air.
                        state_matrix[
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature',
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_panel_1_rear_zone']
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_panel_1_rear_zone']
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_radiator_panel_1_hull_rear_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_panel_1_rear_zone']
                            / zone_data['heat_capacity']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_panel_1_rear_zone']
                            / zone_data['heat_capacity']
                        )

                        # Heat transfer between radiator panel 2 hull front and zone air.
                        state_matrix[
                            f'{zone_name}_radiator_panel_2_hull_front_temperature',
                            f'{zone_name}_radiator_panel_2_hull_front_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_panel_2_front_zone']
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_radiator_panel_2_hull_front_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_panel_2_front_zone']
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_radiator_panel_2_hull_front_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_panel_2_front_zone']
                            / zone_data['heat_capacity']
                        )
                        state_matrix[
                            f'{zone_name}_temperature',
                            f'{zone_name}_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_panel_2_front_zone']
                            / zone_data['heat_capacity']
                        )

                    # Heat transfer between radiator hull rear and zone air.
                    state_matrix[
                        f'{zone_name}_radiator_hull_rear_temperature',
                        f'{zone_name}_radiator_hull_rear_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_rear_zone']
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_radiator_hull_rear_temperature',
                        f'{zone_name}_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_rear_zone']
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_radiator_hull_rear_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_rear_zone']
                        / zone_data['heat_capacity']
                    )
                    state_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_rear_zone']
                        / zone_data['heat_capacity']
                    )

                    # Heat transfer between radiator hull front / rear and zone surfaces.
                    state_matrix[
                        f'{zone_name}_radiator_hull_front_temperature',
                        f'{zone_name}_radiator_hull_front_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_front_surfaces']
                        / zone_data['heat_capacitance_hull']
                    )
                    state_matrix[
                        f'{zone_name}_radiator_hull_rear_temperature',
                        f'{zone_name}_radiator_hull_rear_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_rear_surfaces']
                        / zone_data['heat_capacitance_hull']
                    )

                    for surface_name, surface_data in (
                        pd.concat(
                            [
                                building_data.surfaces_exterior.loc[
                                    building_data.surfaces_exterior['zone_name'].isin([zone_name]),
                                    :
                                ],
                                building_data.surfaces_interior.loc[
                                    building_data.surfaces_interior['zone_name'].isin([zone_name]),
                                    :
                                ],
                                building_data.surfaces_interior.loc[
                                    building_data.surfaces_interior['zone_adjacent_name'].isin([zone_name]),
                                    :
                                ],
                                building_data.surfaces_adiabatic.loc[
                                    building_data.surfaces_adiabatic['zone_name'].isin([zone_name]),
                                    :
                                ]
                            ],
                            sort=False
                        ).iterrows()  # For all surfaces adjacent to the zone.
                    ):
                        # Front.
                        state_matrix[
                            f'{zone_name}_radiator_hull_front_temperature',
                            f'{surface_name}_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_front_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{surface_name}_temperature',
                            f'{zone_name}_radiator_hull_front_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_front_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )
                        state_matrix[
                            f'{surface_name}_temperature',
                            f'{surface_name}_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_front_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )

                        # Back.
                        state_matrix[
                            f'{zone_name}_radiator_hull_rear_temperature',
                            f'{surface_name}_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_rear_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / zone_data['heat_capacitance_hull']
                        )
                        state_matrix[
                            f'{surface_name}_temperature',
                            f'{zone_name}_radiator_hull_rear_temperature'
                        ] += (
                            1.0
                            / zone_data['thermal_resistance_radiator_rear_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )
                        state_matrix[
                            f'{surface_name}_temperature',
                            f'{surface_name}_temperature'
                        ] += (
                            - 1.0
                            / zone_data['thermal_resistance_radiator_rear_surfaces']
                            * surface_data['surface_area']
                            * (1 - surface_data['window_wall_ratio'])
                            / zone_data['zone_surfaces_wall_area']
                            / surface_data['heat_capacity']
                        )

        def define_heat_transfer_hvac_ahu():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_ahu_type']):
                    control_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] += (
                        1.0
                        / 1000  # l in m³.
                        * building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['ahu_supply_air_temperature_setpoint']
                            - building_data.scenarios['linearization_zone_air_temperature_heat']
                        )
                        * zone_data['zone_area']
                        / zone_data['heat_capacity']
                    )
                    control_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] += (
                        1.0
                        / 1000  # l in m³.
                        * building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['ahu_supply_air_temperature_setpoint']
                            - building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                        * zone_data['zone_area']
                        / zone_data['heat_capacity']
                    )

        def define_heat_transfer_hvac_tu():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_tu_type']):
                    control_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_tu_heat_air_flow'
                    ] += (
                        1.0
                        / 1000  # l in m³.
                        * building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['tu_supply_air_temperature_setpoint']
                            - building_data.scenarios['linearization_zone_air_temperature_heat']
                        )
                        * zone_data['zone_area']
                        / zone_data['heat_capacity']
                    )
                    control_matrix[
                        f'{zone_name}_temperature',
                        f'{zone_name}_tu_cool_air_flow'
                    ] += (
                        1.0
                        / 1000  # l in m³.
                        * building_data.parameters['heat_capacity_air']
                        * (
                            zone_data['tu_supply_air_temperature_setpoint']
                            - building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                        * zone_data['zone_area']
                        / zone_data['heat_capacity']
                    )

        def define_co2_transfer():

            for zone_name, zone_data in building_data.zones.iterrows():
                if (
                        pd.notnull(zone_data['hvac_ahu_type'])
                        & (zone_data['fresh_air_flow_control_type'] == 'co2_based')
                ):
                    state_matrix[
                        f'{zone_name}_co2_concentration',
                        f'{zone_name}_co2_concentration'
                    ] += (
                        - 1.0
                        * building_data.scenarios['linearization_zone_fresh_air_flow']
                        / 1000  # l in m³.
                        / zone_data.at['zone_height']
                    )
                    if pd.notnull(zone_data.at['hvac_ahu_type']):
                        control_matrix[
                            f'{zone_name}_co2_concentration',
                            f'{zone_name}_ahu_heat_air_flow'
                        ] += (
                            - 1.0
                            / 1000  # l in m³.
                            * building_data.scenarios['linearization_zone_air_co2_concentration']
                            / zone_data.at['zone_height']
                        )
                        control_matrix[
                            f'{zone_name}_co2_concentration',
                            f'{zone_name}_ahu_cool_air_flow'
                        ] += (
                            - 1.0
                            / 1000  # l in m³.
                            * building_data.scenarios['linearization_zone_air_co2_concentration']
                            / zone_data.at['zone_height']
                        )
                    # disturbance_matrix[
                    #     f'{zone_name}_co2_concentration',
                    #     'constant'
                    # ] += (
                    #     - 1.0
                    #     * building_data.scenarios['linearization_zone_air_co2_concentration']
                    #     * zone_data.at['infiltration_rate'])
                    # )  # TODO: Revise infiltration
                    if pd.notnull(zone_data.at['internal_gain_type']):
                        disturbance_matrix[
                            f'{zone_name}_co2_concentration',
                            zone_data.at['internal_gain_type'] + '_internal_gain_occupancy'
                        ] += (
                            1.0
                            * zone_data.at['occupancy_density']
                            * zone_data.at['occupancy_co2_gain']
                            / zone_data.at['zone_height']
                            / zone_data.at['zone_area']
                        )
                        # division by zone_area since the occupancy here is in p
                        # if iterative and robust BM, no division by zone_area since the occupancy there is in p/m2
                    disturbance_matrix[
                        f'{zone_name}_co2_concentration',
                        'constant'
                    ] += (
                        1.0
                        * building_data.scenarios['linearization_zone_fresh_air_flow']
                        / 1000  # l in m³.
                        * building_data.scenarios['linearization_zone_air_co2_concentration']
                        / zone_data.at['zone_height']
                    )

        def define_humidity_transfer():

            for zone_name, zone_data in building_data.zones.iterrows():
                if (
                        pd.notnull(zone_data.at['hvac_ahu_type'])
                        & (zone_data.at['humidity_control_type'] == 'humidity_based')
                ):
                    state_matrix[
                        f'{zone_name}_absolute_humidity',
                        f'{zone_name}_absolute_humidity'
                    ] += (
                        - 1.0
                        * building_data.scenarios['linearization_zone_fresh_air_flow']
                        / 1000  # l in m³.
                        / zone_data.at['zone_height']
                    )
                    control_matrix[
                        f'{zone_name}_absolute_humidity',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] += (
                        - 1.0
                        / 1000  # l in m³.
                        * (
                            building_data.scenarios['linearization_zone_air_absolute_humidity']
                            - cobmo.utils.calculate_absolute_humidity_humid_air(
                                zone_data.at['ahu_supply_air_temperature_setpoint'],
                                zone_data.at['ahu_supply_air_relative_humidity_setpoint']
                            )
                        )
                        / zone_data.at['zone_height']
                    )
                    control_matrix[
                        f'{zone_name}_absolute_humidity',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] += (
                        - 1.0
                        / 1000  # l in m³.
                        * (
                            building_data.scenarios['linearization_zone_air_absolute_humidity']
                            - cobmo.utils.calculate_absolute_humidity_humid_air(
                                zone_data.at['ahu_supply_air_temperature_setpoint'],
                                zone_data.at['ahu_supply_air_relative_humidity_setpoint']
                            )
                        )
                        / zone_data.at['zone_height']
                    )
                    disturbance_matrix[
                        f'{zone_name}_absolute_humidity',
                        'constant'
                    ] += (
                        - 1.0
                        / 1000  # l in m³.
                        * (
                            building_data.scenarios['linearization_zone_air_absolute_humidity']
                            - building_data.scenarios['linearization_ambient_air_absolute_humidity']
                        )
                        * zone_data.at['infiltration_rate']
                    )  # TODO: Revise infiltration
                    if pd.notnull(zone_data.at['internal_gain_type']):
                        disturbance_matrix[
                            f'{zone_name}_absolute_humidity',
                            zone_data.at['internal_gain_type'] + '_internal_gain_occupancy'
                        ] += (
                            building_data.parameters['moisture_generation_rate_per_person']
                            / zone_data.at['zone_height']
                            / zone_data.at['zone_area']
                            / building_data.parameters['density_air']
                        )
                    disturbance_matrix[
                        f'{zone_name}_absolute_humidity',
                        'constant'
                    ] += (
                        1.0
                        * building_data.scenarios['linearization_zone_fresh_air_flow']
                        / 1000  # l in m³.
                        * building_data.scenarios['linearization_zone_air_absolute_humidity']
                        / zone_data.at['zone_height']
                    )

        def define_storage_state_of_charge():

            # Sensible storage cooling.
            if building_data.scenarios['storage_commodity_type'] == 'sensible_cooling':

                # Storage charge.
                control_matrix[
                    'storage_state_of_charge',
                    'storage_charge_thermal_power_cooling'
                ] += (
                    100.0  # in %.
                    * building_data.scenarios['storage_round_trip_efficiency']
                    / building_data.scenarios['storage_capacity']
                    / building_data.parameters['water_density']
                    / building_data.parameters['water_specific_heat']
                    / building_data.scenarios['storage_sensible_temperature_delta']
                )

                # Storage discharge.
                control_matrix[
                    'storage_state_of_charge',
                    'storage_discharge_thermal_power_cooling'
                ] += (
                    - 100.0  # in %.
                    / building_data.scenarios['storage_capacity']
                    / building_data.parameters['water_density']
                    / building_data.parameters['water_specific_heat']
                    / building_data.scenarios['storage_sensible_temperature_delta']
                )

            # Sensible storage heating.
            if building_data.scenarios['storage_commodity_type'] == 'sensible_heating':

                # Storage charge.
                control_matrix[
                    'storage_state_of_charge',
                    'storage_charge_thermal_power_heating'
                ] += (
                    100.0  # in %.
                    * building_data.scenarios['storage_round_trip_efficiency']
                    / building_data.scenarios['storage_capacity']
                    / building_data.parameters['water_density']
                    / building_data.parameters['water_specific_heat']
                    / building_data.scenarios['storage_sensible_temperature_delta']
                )

                # Storage discharge.
                control_matrix[
                    'storage_state_of_charge',
                    'storage_discharge_thermal_power_heating'
                ] += (
                    - 100.0  # in %.
                    / building_data.scenarios['storage_capacity']
                    / building_data.parameters['water_density']
                    / building_data.parameters['water_specific_heat']
                    / building_data.scenarios['storage_sensible_temperature_delta']
                )

            # Battery storage.
            if building_data.scenarios['storage_commodity_type'] == 'battery':

                # Storage charge.
                control_matrix[
                    'storage_state_of_charge',
                    'storage_charge_electric_power'
                ] += (
                    100.0  # in %.
                    * building_data.scenarios['storage_round_trip_efficiency']
                    / building_data.scenarios['storage_capacity']
                    / 3600 / 1000  # kWh in Ws.
                    / building_data.scenarios['storage_battery_depth_of_discharge']
                )

                # Storage discharge.
                control_matrix[
                    'storage_state_of_charge',
                    'storage_discharge_electric_power'
                ] += (
                    - 100.0  # in %.
                    / building_data.scenarios['storage_capacity']
                    / 3600 / 1000  # kWh in Ws.
                    / building_data.scenarios['storage_battery_depth_of_discharge']
                )

            # Storage losses.
            if pd.notnull(building_data.scenarios['storage_type']):
                state_matrix[
                    'storage_state_of_charge',
                    'storage_state_of_charge'
                ] += (
                    - 1.0
                    * building_data.scenarios['storage_self_discharge_rate']
                    / 3600  # %/h in %/s.
                )

        def define_output_zone_temperature():

            for zone_name, zone_data in building_data.zones.iterrows():
                state_output_matrix[
                    f'{zone_name}_temperature',
                    f'{zone_name}_temperature'
                ] = 1.0

        def define_output_zone_co2_concentration():

            for zone_name, zone_data in building_data.zones.iterrows():
                if (
                        pd.notnull(zone_data['hvac_ahu_type'])
                        & (zone_data['fresh_air_flow_control_type'] == 'co2_based')
                ):
                    state_output_matrix[
                        f'{zone_name}_co2_concentration',
                        f'{zone_name}_co2_concentration'
                    ] = 1.0

        def define_output_zone_humidity():

            for zone_name, zone_data in building_data.zones.iterrows():
                if (
                        pd.notnull(zone_data.at['hvac_ahu_type'])
                        & (zone_data.at['humidity_control_type'] == 'humidity_based')
                ):
                    state_output_matrix[
                        f'{zone_name}_absolute_humidity',
                        f'{zone_name}_absolute_humidity'
                    ] = 1.0

        def define_output_internal_gain_power():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data.at['internal_gain_type']):
                    disturbance_output_matrix[
                        'electric_power_balance',
                        zone_data.at['internal_gain_type'] + '_internal_gain_appliances'
                    ] += (
                        1.0
                        * zone_data.at['appliances_heat_gain']
                        * zone_data.at['zone_area']
                        / self.zone_area_total
                    )

        def define_output_hvac_generic():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_generic_type']):

                    # Cooling power.
                    control_output_matrix[
                        f'{zone_name}_generic_cool_thermal_power',
                        f'{zone_name}_generic_cool_thermal_power'
                    ] = 1.0
                    control_output_matrix[
                        'thermal_power_cooling_balance',
                        f'{zone_name}_generic_cool_thermal_power'
                    ] = (
                        1.0
                        / zone_data['generic_cooling_efficiency']
                        * zone_data['zone_area']
                        / self.zone_area_total
                    )

                    # Heating power.
                    control_output_matrix[
                        f'{zone_name}_generic_heat_thermal_power',
                        f'{zone_name}_generic_heat_thermal_power'
                    ] = 1.0
                    control_output_matrix[
                        'thermal_power_heating_balance',
                        f'{zone_name}_generic_heat_thermal_power'
                    ] = (
                        1.0
                        / zone_data['generic_heating_efficiency']
                        * zone_data['zone_area']
                        / self.zone_area_total
                    )

        def define_output_hvac_radiator_power():

            if pd.notnull(building_data.zones['hvac_radiator_type']).any():
                for zone_name, zone_data in building_data.zones.iterrows():
                    if pd.notnull(zone_data['hvac_radiator_type']):

                        # Heating power (radiators do not require cooling power).
                        control_output_matrix[
                            f'{zone_name}_radiator_thermal_power',
                            f'{zone_name}_radiator_thermal_power'
                        ] = 1.0
                        control_output_matrix[
                            'thermal_power_heating_balance',
                            f'{zone_name}_radiator_thermal_power'
                        ] = (
                            1.0
                            / zone_data['radiator_heating_efficiency']
                            * zone_data['zone_area']
                            / self.zone_area_total
                        )

        def define_output_hvac_ahu_power():

            for zone_name, zone_data in building_data.zones.iterrows():
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
                            building_data.scenarios['linearization_zone_air_temperature_heat']
                            + building_data.scenarios['linearization_zone_air_temperature_cool']
                        )
                    )
                    linearization_zone_air_absolute_humidity = (
                        0.5
                        * (
                            building_data.scenarios['linearization_ambient_air_absolute_humidity']
                            + ahu_supply_air_absolute_humidity_setpoint
                        )
                    )
                    delta_enthalpy_ahu_recovery = (
                        cobmo.utils.calculate_enthalpy_humid_air(
                            linearization_zone_air_temperature,
                            linearization_zone_air_absolute_humidity
                        )
                        - cobmo.utils.calculate_enthalpy_humid_air(
                            building_data.scenarios['linearization_ambient_air_temperature'],
                            linearization_zone_air_absolute_humidity
                        )
                    )

                    # Obtain enthalpies.
                    if (
                        building_data.scenarios['linearization_ambient_air_absolute_humidity']
                        <= ahu_supply_air_absolute_humidity_setpoint
                    ):
                        delta_enthalpy_ahu_cooling = min(
                            0.0,
                            cobmo.utils.calculate_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                building_data.scenarios['linearization_ambient_air_absolute_humidity']
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                building_data.scenarios['linearization_ambient_air_temperature'],
                                building_data.scenarios['linearization_ambient_air_absolute_humidity']
                            )
                        )
                        delta_enthalpy_ahu_heating = max(
                            0.0,
                            cobmo.utils.calculate_enthalpy_humid_air(
                                zone_data['ahu_supply_air_temperature_setpoint'],
                                building_data.scenarios['linearization_ambient_air_absolute_humidity']
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                building_data.scenarios['linearization_ambient_air_temperature'],
                                building_data.scenarios['linearization_ambient_air_absolute_humidity']
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
                                building_data.scenarios['linearization_ambient_air_temperature'],
                                building_data.scenarios['linearization_ambient_air_absolute_humidity']
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

                    # Air flow.
                    control_output_matrix[
                        f'{zone_name}_ahu_cool_air_flow',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] = 1.0
                    control_output_matrix[
                        f'{zone_name}_ahu_heat_air_flow',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] = 1.0

                    # Cooling power.
                    control_output_matrix[
                        'thermal_power_cooling_balance',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                        / zone_data['ahu_cooling_efficiency']
                    )
                    control_output_matrix[
                        'thermal_power_cooling_balance',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                        / zone_data['ahu_cooling_efficiency']
                    )

                    # Heating power.
                    control_output_matrix[
                        'thermal_power_heating_balance',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                        / zone_data['ahu_heating_efficiency']
                    )
                    control_output_matrix[
                        'thermal_power_heating_balance',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                        / zone_data['ahu_heating_efficiency']
                    )

                    # Fan power.
                    control_output_matrix[
                        'electric_power_balance',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * zone_data['ahu_fan_efficiency']
                    )
                    control_output_matrix[
                        'electric_power_balance',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * zone_data['ahu_fan_efficiency']
                    )

        def define_output_hvac_tu_power():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_tu_type']):
                    # Calculate enthalpies.
                    if zone_data['tu_air_intake_type'] == 'zone':
                        delta_enthalpy_tu_cooling = building_data.parameters['heat_capacity_air'] * (
                            building_data.scenarios['linearization_zone_air_temperature_cool']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                        delta_enthalpy_tu_heating = building_data.parameters['heat_capacity_air'] * (
                            building_data.scenarios['linearization_zone_air_temperature_heat']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                    elif zone_data['tu_air_intake_type'] == 'ahu':
                        delta_enthalpy_tu_cooling = building_data.parameters['heat_capacity_air'] * (
                            building_data.scenarios['ahu_supply_air_temperature_setpoint']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                        delta_enthalpy_tu_heating = building_data.parameters['heat_capacity_air'] * (
                            building_data.scenarios['ahu_supply_air_temperature_setpoint']
                            - zone_data['tu_supply_air_temperature_setpoint']
                        )
                    else:
                        logger.error(f"Unknown `tu_air_intake_type` type: {zone_data['tu_air_intake_type']}")
                        raise ValueError

                    # Air flow.
                    control_output_matrix[
                        f'{zone_name}_tu_cool_air_flow',
                        f'{zone_name}_tu_cool_air_flow'
                    ] = 1.0
                    control_output_matrix[
                        f'{zone_name}_tu_heat_air_flow',
                        f'{zone_name}_tu_heat_air_flow'
                    ] = 1.0

                    # Cooling power.
                    control_output_matrix[
                        'thermal_power_cooling_balance',
                        f'{zone_name}_tu_cool_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * abs(delta_enthalpy_tu_cooling)
                        / zone_data['tu_cooling_efficiency']
                    )

                    # Heating power.
                    control_output_matrix[
                        'thermal_power_heating_balance',
                        f'{zone_name}_tu_heat_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * abs(delta_enthalpy_tu_heating)
                        / zone_data['tu_heating_efficiency']
                    )

                    # Fan power.
                    control_output_matrix[
                        'electric_power_balance',
                        f'{zone_name}_tu_cool_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * zone_data['tu_fan_efficiency']
                    )
                    control_output_matrix[
                        'electric_power_balance',
                        f'{zone_name}_tu_heat_air_flow'
                    ] = (
                        1.0
                        / 1000  # l in m³.
                        * zone_data['zone_area']
                        / self.zone_area_total
                        * building_data.parameters['density_air']
                        * zone_data['tu_fan_efficiency']
                    )

        def define_output_fresh_air_flow():

            for zone_name, zone_data in building_data.zones.iterrows():
                if pd.notnull(zone_data['hvac_ahu_type']):
                    control_output_matrix[
                        f'{zone_name}_total_fresh_air_flow',
                        f'{zone_name}_ahu_cool_air_flow'
                    ] = 1.0
                    control_output_matrix[
                        f'{zone_name}_total_fresh_air_flow',
                        f'{zone_name}_ahu_heat_air_flow'
                    ] = 1.0
                if pd.notnull(zone_data['hvac_ahu_type']):
                    disturbance_output_matrix[
                        f'{zone_name}_total_fresh_air_flow',
                        'constant'
                    ] += (
                        zone_data['infiltration_rate']
                        * zone_data['zone_height']
                    )  # TODO: Revise infiltration.

        def define_output_storage_state_of_charge():

            if pd.notnull(building_data.scenarios['storage_type']):
                state_output_matrix[
                    'storage_state_of_charge',
                    'storage_state_of_charge'
                ] = 1.0

        def define_output_storage_power():

            # Sensible storage cooling.
            if building_data.scenarios['storage_commodity_type'] == 'sensible_cooling':
                control_output_matrix[
                    'thermal_power_cooling_balance',
                    'storage_charge_thermal_power_cooling'
                ] = 1.0
                control_output_matrix[
                    'storage_charge_thermal_power_cooling',
                    'storage_charge_thermal_power_cooling'
                ] = 1.0
                control_output_matrix[
                    'thermal_power_cooling_balance',
                    'storage_discharge_thermal_power_cooling'
                ] = - 1.0
                control_output_matrix[
                    'storage_discharge_thermal_power_cooling',
                    'storage_discharge_thermal_power_cooling'
                ] = 1.0

            # Sensible storage heating.
            if building_data.scenarios['storage_commodity_type'] == 'sensible_heating':
                control_output_matrix[
                    'thermal_power_cooling_balance',
                    'storage_charge_thermal_power_heating'
                ] = 1.0
                control_output_matrix[
                    'storage_charge_thermal_power_heating',
                    'storage_charge_thermal_power_heating'
                ] = 1.0
                control_output_matrix[
                    'thermal_power_cooling_balance',
                    'storage_discharge_thermal_power_heating'
                ] = - 1.0
                control_output_matrix[
                    'storage_discharge_thermal_power_heating',
                    'storage_discharge_thermal_power_heating'
                ] = 1.0

            # Battery storage.
            if building_data.scenarios['storage_commodity_type'] == 'battery':
                control_output_matrix[
                    'electric_power_balance',
                    'storage_charge_electric_power'
                ] = 1.0
                control_output_matrix[
                    'storage_charge_electric_power',
                    'storage_charge_electric_power'
                ] = 1.0
                control_output_matrix[
                    'electric_power_balance',
                    'storage_discharge_electric_power'
                ] = - 1.0
                control_output_matrix[
                    'storage_discharge_electric_power',
                    'storage_discharge_electric_power'
                ] = 1.0

        def define_output_plant_power():

            # Cooling.
            control_output_matrix[
                'thermal_power_cooling_balance',
                'plant_thermal_power_cooling'
            ] = - 1.0
            control_output_matrix[
                'plant_thermal_power_cooling',
                'plant_thermal_power_cooling'
            ] = 1.0
            control_output_matrix[
                'grid_electric_power',
                'plant_thermal_power_cooling'
            ] = (
                1.0
                / building_data.scenarios['plant_cooling_efficiency']
            )

            # Heating.
            control_output_matrix[
                'thermal_power_heating_balance',
                'plant_thermal_power_heating'
            ] = - 1.0
            control_output_matrix[
                'plant_thermal_power_heating',
                'plant_thermal_power_heating'
            ] = 1.0
            control_output_matrix[
                'grid_electric_power',
                'plant_thermal_power_heating'
            ] = (
                1.0
                / building_data.scenarios['plant_heating_efficiency']
            )

        def define_output_grid_power():

            # Electric.
            control_output_matrix[
                'electric_power_balance',
                'grid_electric_power'
            ] = - 1.0
            control_output_matrix[
                'grid_electric_power',
                'grid_electric_power'
            ] = 1.0

            # Cooling.
            control_output_matrix[
                'thermal_power_cooling_balance',
                'grid_thermal_power_cooling'
            ] = - 1.0
            control_output_matrix[
                'grid_thermal_power_cooling',
                'grid_thermal_power_cooling'
            ] = 1.0

            # Heating.
            control_output_matrix[
                'thermal_power_heating_balance',
                'grid_thermal_power_heating'
            ] = - 1.0
            control_output_matrix[
                'grid_thermal_power_heating',
                'grid_thermal_power_heating'
            ] = 1.0

        def define_output_validation_surface_temperature():

            for surface_name, surface_data in (
                    pd.concat([
                        building_data.surfaces_adiabatic,
                        building_data.surfaces_exterior,
                        building_data.surfaces_interior
                    ], sort=False).iterrows()
            ):
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    state_output_matrix[
                        f'{surface_name}_temperature',
                        f'{surface_name}_temperature'
                    ] = 1.0

        def define_output_validation_surfaces_exterior_irradiation_gain_exterior():

            for surface_name, surface_data in building_data.surfaces_exterior.iterrows():
                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    disturbance_output_matrix[
                        f'{surface_name}_irradiation_gain_exterior',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_surface']
                        * (1 - surface_data['window_wall_ratio'])
                    )
                else:  # Surfaces with neglected heat capacity
                    disturbance_output_matrix[
                        surface_data['surface_name'] + '_irradiation_gain_exterior',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_surface']
                        * (1 - surface_data['window_wall_ratio'])
                    )

        def define_output_validation_surfaces_exterior_convection_interior():

            for surface_name, surface_data in building_data.surfaces_exterior.iterrows():
                # Total zone surface area for later calculating share of interior (indirect) irradiation
                zone_surface_area = sum(
                    zone_surface_data['surface_area']
                    * (1 - zone_surface_data['window_wall_ratio'])
                    for zone_surface_name, zone_surface_data in pd.concat(
                        [
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ],
                            building_data.surfaces_interior[
                                building_data.surfaces_interior['zone_name'] == surface_data['zone_name']
                            ],
                            building_data.surfaces_interior[
                                building_data.surfaces_interior['zone_adjacent_name'] == surface_data['zone_name']
                            ],
                            building_data.surfaces_adiabatic[
                                building_data.surfaces_adiabatic['zone_name'] == surface_data['zone_name']
                            ]
                        ],
                        sort=False
                    ).iterrows()  # For all surfaces adjacent to the zone
                )

                if surface_data['heat_capacity'] != 0.0:  # Surfaces with non-zero heat capacity
                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in (
                            building_data.surfaces_exterior[
                                building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ].iterrows()
                    ):
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_output_matrix[
                            surface_data['surface_name'] + '_convection_interior',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / zone_surface_area
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * (1.0 - surface_data['window_wall_ratio'])
                            * (
                                1.0
                                - (
                                    1.0
                                    + (
                                        building_data.parameters['heat_transfer_coefficient_interior_convection']
                                    )
                                    / (
                                        2.0
                                        * surface_data['heat_transfer_coefficient_conduction_surface']
                                    )
                                ) ** (- 1)
                            )
                        )
                    state_output_matrix[
                        surface_data['surface_name'] + '_convection_interior',
                        f'{surface_name}_temperature'
                    ] += (
                        (1.0 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                    )
                    state_output_matrix[
                        surface_data['surface_name'] + '_convection_interior',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * (1.0 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_interior_convection']
                            )
                            + 1.0
                            / (
                                2.0
                                * surface_data['heat_transfer_coefficient_conduction_surface']
                            )
                        ) ** (- 1)
                    )
                else:  # Surfaces with neglected heat capacity
                    # Complete convective heat transfer from surface to zone
                    disturbance_output_matrix[
                        surface_data['surface_name'] + '_convection_interior',
                        'irradiation_' + surface_data['direction_name']
                    ] += (
                        surface_data['absorptivity_surface']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                    )
                    disturbance_output_matrix[
                        surface_data['surface_name'] + '_convection_interior',
                        'ambient_air_temperature'
                    ] += (
                        (
                            building_data.parameters['heat_transfer_coefficient_exterior_convection']
                            + surface_data['heat_transfer_coefficient_surface_ground']
                        )
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                    )
                    disturbance_output_matrix[
                        surface_data['surface_name'] + '_convection_interior',
                        'sky_temperature'
                    ] += (
                        surface_data['heat_transfer_coefficient_surface_sky']
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                    )
                    state_output_matrix[
                        surface_data['surface_name'] + '_convection_interior',
                        surface_data['zone_name'] + '_temperature'
                    ] += (
                        - 1.0
                        * (1 - surface_data['window_wall_ratio'])
                        * (
                            1.0
                            / (
                                building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                + surface_data['heat_transfer_coefficient_surface_ground']
                                + surface_data['heat_transfer_coefficient_surface_sky']
                            )
                            + 1.0
                            / building_data.parameters['heat_transfer_coefficient_interior_convection']
                            + 1.0
                            / (surface_data['heat_transfer_coefficient_conduction_surface'])
                        ) ** (- 1)
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in building_data.surfaces_exterior[
                        building_data.surfaces_exterior['zone_name'] == surface_data['zone_name']
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        disturbance_output_matrix[
                            surface_data['surface_name'] + '_convection_interior',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] += (
                            (
                                zone_exterior_surface_data['surface_area']
                                * zone_exterior_surface_data['window_wall_ratio']
                                / zone_surface_area
                            )  # Considers the share at the respective surface
                            * surface_data['absorptivity_surface']
                            * (1 - surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                    building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / building_data.parameters['heat_transfer_coefficient_interior_convection']
                                + (
                                    building_data.parameters['heat_transfer_coefficient_exterior_convection']
                                    + surface_data['heat_transfer_coefficient_surface_ground']
                                    + surface_data['heat_transfer_coefficient_surface_sky']
                                )
                                / (surface_data['heat_transfer_coefficient_conduction_surface'])
                            ) ** (- 1))
                        )

        def define_disturbance_timeseries():

            # Reindex, interpolate and construct full disturbance timeseries.
            self.disturbance_timeseries = pd.concat(
                [
                    building_data.weather_timeseries[[
                        'ambient_air_temperature',
                        'sky_temperature',
                        'irradiation_horizontal',
                        'irradiation_east',
                        'irradiation_south',
                        'irradiation_west',
                        'irradiation_north'
                    ]],
                    building_data.internal_gain_timeseries,
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

            if pd.isnull(building_data.scenarios['price_type']):
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
                self.electricity_price_timeseries = building_data.electricity_price_timeseries

        def define_electricity_price_distribution_timeseries():

            self.electricity_price_distribution_timeseries = (
                building_data.electricity_price_distribution_timeseries
            )

        def define_output_constraint_timeseries():

            # Do not define constraints, if `constraint_type` not defined for any zones.
            if any(pd.isnull(building_data.zones.loc[:, 'constraint_type'])):
                logger.debug('Skipping definition of constraint timeseries due to missing constraint type definition.')
                return

            # Instantiate constraint timeseries.
            self.output_maximum_timeseries = pd.DataFrame(
                +1.0 * np.infty,
                self.timesteps,
                self.outputs
            )
            self.output_minimum_timeseries = pd.DataFrame(
                -1.0 * np.infty,
                self.timesteps,
                self.outputs
            )

            # Obtain indexing shorthands.
            zones_fixed_fresh_air_flow_index = (
                pd.notnull(building_data.zones['hvac_ahu_type'])
                & pd.isnull(building_data.zones['fresh_air_flow_control_type'])
            )
            zones_occupancy_based_index = (
                pd.notnull(building_data.zones['hvac_ahu_type'])
                & (building_data.zones['fresh_air_flow_control_type'] == 'occupancy_based')
            )
            zones_co2_based_index = (
                pd.notnull(building_data.zones['hvac_ahu_type'])
                & (building_data.zones['fresh_air_flow_control_type'] == 'co2_based')
            )
            zones_humidity_based_index = (
                pd.notnull(building_data.zones['hvac_ahu_type'])
                & (building_data.zones['humidity_control_type'] == 'humidity_based')
            )

            # Minimum constraint for power outputs.
            self.output_minimum_timeseries.loc[
                :, self.outputs.str.contains('_power')
            ] = 0.0

            # Minimum constraint for flow outputs.
            self.output_minimum_timeseries.loc[
                :, self.outputs.str.contains('_flow')
            ] = 0.0

            # Minimum / maximum constraint for balance outputs.
            self.output_minimum_timeseries.loc[
                :, self.outputs.str.contains('_balance')
            ] = 0.0
            self.output_maximum_timeseries.loc[
                :, self.outputs.str.contains('_balance')
            ] = 0.0

            # Minimum / maximum constraint for zone air temperature.
            self.output_minimum_timeseries.loc[
                :, building_data.zones['zone_name'] + '_temperature'
            ] = (
                building_data.constraint_timeseries.loc[
                    :, building_data.zones['constraint_type'] + '_minimum_air_temperature'
                ]
            ).values
            self.output_maximum_timeseries.loc[
                :, building_data.zones['zone_name'] + '_temperature'
            ] = (
                building_data.constraint_timeseries.loc[
                    :, building_data.zones['constraint_type'] + '_maximum_air_temperature'
                ]
            ).values

            # Minimum constraint for fixed zone fresh air flow.
            if zones_fixed_fresh_air_flow_index.any():
                self.output_minimum_timeseries.loc[
                    :, building_data.zones.loc[zones_fixed_fresh_air_flow_index, 'zone_name'] + '_total_fresh_air_flow'
                ] = (
                    building_data.constraint_timeseries.loc[
                        :, (
                            building_data.zones.loc[zones_fixed_fresh_air_flow_index, 'constraint_type']
                            + '_minimum_fresh_air_flow'
                        )
                    ].values
                )

            # Minimum constraint for occupancy-based zone fresh air flow.
            if zones_occupancy_based_index.any():
                self.output_minimum_timeseries.loc[
                    :, building_data.zones.loc[zones_occupancy_based_index, 'zone_name'] + '_total_fresh_air_flow'
                ] = (
                    building_data.constraint_timeseries.loc[
                        :, (
                            building_data.zones.loc[zones_occupancy_based_index, 'constraint_type']
                            + '_minimum_fresh_air_flow_building'
                        )
                    ].values
                    + building_data.constraint_timeseries.loc[
                        :, (
                            building_data.zones.loc[zones_occupancy_based_index, 'constraint_type']
                            + '_minimum_fresh_air_flow_occupants'
                        )
                    ].values
                    * building_data.internal_gain_timeseries.loc[
                        :, (
                            building_data.zones.loc[zones_occupancy_based_index, 'internal_gain_type']
                            + '_internal_gain_occupancy'
                        )
                    ].values
                    * building_data.zones.loc[zones_occupancy_based_index, 'occupancy_density'].values
                )

            # Maximum constraint for zone CO2 concentration.
            if zones_co2_based_index.any():
                self.output_minimum_timeseries.loc[
                    :, building_data.zones.loc[zones_co2_based_index, 'zone_name'] + '_co2_concentration'
                ] = (
                    building_data.constraint_timeseries.loc[
                        :, (
                            building_data.zones.loc[zones_co2_based_index, 'constraint_type']
                            + '_maximum_co2_concentration'
                        )
                    ]
                ).values

            # Minimum / maximum constraint for zone humidity concentration.
            if zones_humidity_based_index.any():
                self.output_minimum_timeseries.loc[
                    :, building_data.zones.loc[zones_humidity_based_index, 'zone_name'] + '_absolute_humidity'
                ] = (
                    np.vectorize(cobmo.utils.calculate_absolute_humidity_humid_air)(
                        building_data.scenarios['linearization_zone_air_temperature_cool'],
                        building_data.constraint_timeseries.loc[
                            :, (
                                building_data.zones.loc[zones_humidity_based_index, 'constraint_type']
                                + '_minimum_relative_humidity'
                            )
                        ]
                    )
                )
                self.output_maximum_timeseries.loc[
                    :, building_data.zones.loc[zones_humidity_based_index, 'zone_name'] + '_absolute_humidity'
                ] = (
                    np.vectorize(cobmo.utils.calculate_absolute_humidity_humid_air)(
                        building_data.scenarios['linearization_zone_air_temperature_cool'],
                        building_data.constraint_timeseries.loc[
                            :, (
                                building_data.zones.loc[zones_humidity_based_index, 'constraint_type']
                                + '_maximum_relative_humidity'
                            )
                        ]
                    )
                )

            # Minimum / maximum constraints for storage state of charge.
            if pd.notnull(building_data.scenarios['storage_type']):
                self.output_minimum_timeseries.loc[
                    :, 'storage_state_of_charge'
                ] = 0.0
                self.output_maximum_timeseries.loc[
                    :, 'storage_state_of_charge'
                ] = 100.0  # in %.

            # Electric / thermal grid connections.
            if not connect_electric_grid:
                self.output_maximum_timeseries.loc[
                    :, 'plant_thermal_power_cooling'
                ] = 0.0
                self.output_maximum_timeseries.loc[
                    :, 'plant_thermal_power_heating'
                ] = 0.0
            if not connect_thermal_grid_cooling:
                self.output_maximum_timeseries.loc[
                    :, 'grid_thermal_power_cooling'
                ] = 0.0
            if not connect_thermal_grid_heating:
                self.output_maximum_timeseries.loc[
                    :, 'grid_thermal_power_heating'
                ] = 0.0

        def discretize_model():

            # Discretize state space model with zero order hold.
            # - Reference: <https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models>
            state_matrix_discrete = scipy.linalg.expm(
                self.state_matrix.values
                * self.timestep_interval.seconds
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

            self.state_matrix.loc[:, :] = state_matrix_discrete
            self.control_matrix.loc[:, :] = control_matrix_discrete
            self.disturbance_matrix.loc[:, :] = disturbance_matrix_discrete

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
        define_co2_transfer()
        define_heat_transfer_window_air_flow()
        define_humidity_transfer()
        define_storage_state_of_charge()

        # Define outputs.
        define_output_zone_temperature()
        define_output_zone_co2_concentration()
        define_output_zone_humidity()
        define_output_internal_gain_power()
        define_output_window_air_flow()
        define_output_hvac_generic()
        define_output_hvac_radiator_power()
        define_output_hvac_ahu_power()
        define_output_hvac_tu_power()
        define_output_fresh_air_flow()
        define_output_storage_state_of_charge()
        define_output_storage_power()
        define_output_plant_power()
        define_output_grid_power()

        # Define validation outputs.
        if with_validation_outputs:
            define_output_validation_surface_temperature()
            define_output_validation_surfaces_exterior_irradiation_gain_exterior()
            define_output_validation_surfaces_exterior_convection_interior()

        # Define timeseries.
        define_disturbance_timeseries()
        define_electricity_price_timeseries()
        define_electricity_price_distribution_timeseries()
        define_output_constraint_timeseries()

        # Convert matrix constructors to dataframes.
        self.state_matrix = state_matrix.to_dataframe_dense()
        self.control_matrix = control_matrix.to_dataframe_dense()
        self.disturbance_matrix = disturbance_matrix.to_dataframe_dense()
        self.state_output_matrix = state_output_matrix.to_dataframe_dense()
        self.control_output_matrix = control_output_matrix.to_dataframe_dense()
        self.disturbance_output_matrix = disturbance_output_matrix.to_dataframe_dense()

        # Convert to time discrete model.
        discretize_model()

    def simulate(
            self,
            control_vector: pd.DataFrame,
            state_vector_initial=None,
            disturbance_timeseries=None
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate building model for given control vector timeseries to obtain state vector timeseries and
        output vector timeseries.

        - The simulation is based on the iterative solution of the state space equations.
        - The required initial state vector and disturbance timeseries are obtained from
        the building model definition or can be provided through keyword arguments.

        :syntax:
            `building_model.simulate(control_vector)`: Simulate `building_model` for given `control_vector`.

        Arguments:
            control_vector (pd.DataFrame): Control vector timeseries, as dataframe with control variables as columns and
                timesteps as rows.

        Keyword Arguments:
            state_vector_initial (pd.Series): Initial state vector values, as series with state variables as index.
                Defaults to the initial state vector in the building model definition.
            disturbance_timeseries (pd.DataFrame): Disturbance vector timeseries, sa dataframe with disturbance
                variables as columns and timesteps as rows. Defaults to the disturbance timeseries in the building
                model definition.

        Returns:
            typing.Tuple[pd.DataFrame, pd.DataFrame]: State vector timeseries, as dataframe with state variables as rows
                and timesteps as columns. Output vector timeseries, as dataframe with output variables as rows and
                timesteps as columns.
        """

        # Obtain initial state vector and disturbance timeseries.
        if state_vector_initial is None:
            state_vector_initial = self.state_vector_initial
        if disturbance_timeseries is None:
            disturbance_timeseries = self.disturbance_timeseries

        # Initialize state and output timeseries.
        state_vector = pd.DataFrame(
            np.nan,
            self.timesteps,
            self.states
        )
        state_vector.loc[self.timesteps[0], :] = state_vector_initial
        output_vector = pd.DataFrame(
            np.nan,
            self.timesteps,
            self.outputs
        )

        # Iterative solution of the state space equations.
        # - The following equations directly use the underlying numpy arrays for faster evaluation.
        for timestep in range(len(self.timesteps) - 1):
            state_vector.values[timestep + 1, :] = (
                self.state_matrix.values @ state_vector.values[timestep, :]
                + self.control_matrix.values @ control_vector.values[timestep, :]
                + self.disturbance_matrix.values @ disturbance_timeseries.values[timestep, :]
            )
        for timestep in range(len(self.timesteps)):
            output_vector.values[timestep, :] = (
                self.state_output_matrix.values @ state_vector.values[timestep, :]
                + self.control_output_matrix.values @ control_vector.values[timestep, :]
                + self.disturbance_output_matrix.values @ disturbance_timeseries.values[timestep, :]
            )

        return (
            state_vector,
            output_vector
        )

    def define_optimization_variables(
            self,
            optimization_problem: cobmo.utils.OptimizationProblem,
    ):

        # Define variables.
        # - Defined as dict with single entry for current DER. This is for compability of
        # `define_optimization_constraints`, etc. with `DERModelSet`.
        optimization_problem.state_vector = cp.Variable((len(self.timesteps), len(self.states)))
        optimization_problem.control_vector = cp.Variable((len(self.timesteps), len(self.controls)))
        optimization_problem.output_vector = cp.Variable((len(self.timesteps), len(self.outputs)))

    def define_optimization_constraints(
        self,
        optimization_problem: cobmo.utils.OptimizationProblem,
        initial_state_is_final_state=False
    ):

        # Initial state.
        # - If desired, initial state is set equal to final state. This enables automatic selection of the
        #   optimal initial state, assuming that the start and end timestep are the same time of day.
        if initial_state_is_final_state:
            optimization_problem.constraints.append(
                optimization_problem.state_vector[0, :]
                ==
                optimization_problem.state_vector[-1, :]
            )
        # - Otherwise, set initial state according to the initial state vector.
        else:
            optimization_problem.constraints.append(
                optimization_problem.state_vector[0, :]
                ==
                self.state_vector_initial.values
            )

        # State equation.
        optimization_problem.constraints.append(
            optimization_problem.state_vector[1:, :]
            ==
            cp.transpose(
                self.state_matrix.values
                @ cp.transpose(optimization_problem.state_vector[:-1, :])
                + self.control_matrix.values
                @ cp.transpose(optimization_problem.control_vector[:-1, :])
                + self.disturbance_matrix.values
                @ np.transpose(self.disturbance_timeseries.iloc[:-1, :].values)
            )
        )

        # Output equation.
        optimization_problem.constraints.append(
            optimization_problem.output_vector
            ==
            cp.transpose(
                self.state_output_matrix.values
                @ cp.transpose(optimization_problem.state_vector)
                + self.control_output_matrix.values
                @ cp.transpose(optimization_problem.control_vector)
                + self.disturbance_output_matrix.values
                @ np.transpose(self.disturbance_timeseries.values)
            )
        )

        # Output limits.
        optimization_problem.constraints.append(
            optimization_problem.output_vector
            >=
            self.output_minimum_timeseries.values
        )
        optimization_problem.constraints.append(
            optimization_problem.output_vector
            <=
            self.output_maximum_timeseries.values
        )

    def define_optimization_objective(
            self,
            optimization_problem: cobmo.utils.OptimizationProblem
    ):

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')

        # Define operation cost (OPEX).
        optimization_problem.operation_cost = (
            optimization_problem.output_vector[:, self.outputs.get_loc('grid_electric_power')]
            * self.zone_area_total  # W/m² in W.
            * timestep_interval_hours / 1000.0  # W in kWh.
            @ self.electricity_price_timeseries.loc[:, 'price'].values
        )

        # Add to objective.
        optimization_problem.objective += optimization_problem.operation_cost

    def get_optimization_results(
            self,
            optimization_problem: cobmo.utils.OptimizationProblem
    ) -> dict:

        # Obtain results.
        state_vector = (
            pd.DataFrame(
                optimization_problem.state_vector.value,
                index=self.timesteps,
                columns=self.states
            )
        )
        control_vector = (
            pd.DataFrame(
                optimization_problem.control_vector.value,
                index=self.timesteps,
                columns=self.controls
            )
        )
        output_vector = (
            pd.DataFrame(
                optimization_problem.output_vector.value,
                index=self.timesteps,
                columns=self.outputs
            )
        )
        operation_cost = optimization_problem.operation_cost.value

        return dict(
            state_vector=state_vector,
            control_vector=control_vector,
            output_vector=output_vector,
            operation_cost=operation_cost
        )

    def optimize(self):
        """Optimize the operation, i.e. the control vector, of the building model to minimize operation cost, subject to
        output minimum / maximum constraints. Returns results as dictionary containing state, control and output
        vector timeseries along with the operation cost.

        - The price timeseries is obtained from the building model definition.
        - The required initial state vector and disturbance timeseries are obtained from
          the building model definition.

        :syntax:
            `building_model.optimize(): Optimize the operation of `building_model` and return the results.

        Returns:
            dict: Results dictionary with keys `state_vector`, `control_vector`, `output_vector` and
                `operation_cost`. State vector timeseries `state_vector`, as dataframe with state variables as rows
                and timesteps as columns. Control vector timeseries `control_vector`, as dataframe with control
                variables as rows and timesteps as columns. Output vector timeseries `output_vector`, as dataframe
                with output variables as rows and timesteps as columns. Total operation cost as float `operation_cost`.
        """

        # Instantiate optimization problem.
        optimization_problem = cobmo.utils.OptimizationProblem()

        # Define optimization problem.
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_constraints(optimization_problem)
        self.define_optimization_objective(optimization_problem)

        # Solve optimization problem.
        optimization_problem.solve()

        # Obtain results.
        results = self.get_optimization_results(optimization_problem)

        return results
