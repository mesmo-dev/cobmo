"""Building model class definition."""

import warnings
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.interpolate
# Using CoolProp for calculating humid air properties: http://www.coolprop.org/fluid_properties/HumidAir.html
from CoolProp.HumidAirProp import HAPropsSI as humid_air_properties

import cobmo.utils


class Building(object):
    """
    Building model object.
    """

    def __init__(self, conn, scenario_name):
        """Initialize building model for given `scenario_name`."""

        # Obtain scenario name.
        self.scenario_name = scenario_name

        # Load building model definition.
        self.building_scenarios = (
            pd.read_sql(
                """
                SELECT * FROM building_scenarios 
                JOIN buildings USING (building_name) 
                JOIN building_linearization_types USING (linearization_type) 
                LEFT JOIN building_initial_state_types USING (initial_state_type) 
                LEFT JOIN building_storage_types USING (building_storage_type) 
                WHERE scenario_name='{}'
                """.format(self.scenario_name),
                conn
            )
        )
        self.building_parameters = (
            pd.read_sql(
                """
                SELECT * FROM building_parameter_sets 
                WHERE parameter_set IN ('constants', '{}')
                """.format(self.building_scenarios['parameter_set'][0]),
                conn
            )
        )
        self.building_parameters = (
            pd.Series(  # Convert to series for shorter indexing.
                self.building_parameters['parameter_value'].values,
                self.building_parameters['parameter_name'].values
            )
        )
        self.building_surfaces_adiabatic = (
            pd.read_sql(
                """
                SELECT * FROM building_surfaces_adiabatic 
                JOIN building_surface_types USING (surface_type) 
                LEFT JOIN building_window_types USING (window_type) 
                JOIN building_zones USING (zone_name, building_name) 
                WHERE building_name='{}'
                """.format(self.building_scenarios['building_name'][0]),
                conn
            )
        )
        self.building_surfaces_adiabatic.index = self.building_surfaces_adiabatic['surface_name']
        self.building_surfaces_exterior = (
            pd.read_sql(
                """
                SELECT * FROM building_surfaces_exterior 
                JOIN building_surface_types USING (surface_type) 
                LEFT JOIN building_window_types USING (window_type) 
                JOIN building_zones USING (zone_name, building_name) 
                WHERE building_name='{}'
                """.format(self.building_scenarios['building_name'][0]),
                conn
            )
        )
        self.building_surfaces_exterior.index = self.building_surfaces_exterior['surface_name']
        self.building_surfaces_interior = (
            pd.read_sql(
                """
                SELECT * FROM building_surfaces_interior 
                JOIN building_surface_types USING (surface_type) 
                LEFT JOIN building_window_types USING (window_type) 
                JOIN building_zones USING (zone_name, building_name) 
                WHERE building_name='{}'
                """.format(self.building_scenarios['building_name'][0]),
                conn
            )
        )
        self.building_surfaces_interior.index = self.building_surfaces_interior['surface_name']
        self.building_zones = (
            pd.read_sql(
                """
                SELECT * FROM building_zones 
                JOIN building_zone_types USING (zone_type) 
                JOIN building_internal_gain_types USING (internal_gain_type) 
                LEFT JOIN building_blind_types USING (blind_type) 
                LEFT JOIN building_hvac_generic_types USING (hvac_generic_type) 
                LEFT JOIN building_hvac_radiator_types USING (hvac_radiator_type) 
                LEFT JOIN building_hvac_ahu_types USING (hvac_ahu_type) 
                LEFT JOIN building_hvac_tu_types USING (hvac_tu_type) 
                WHERE building_name='{}'
                """.format(self.building_scenarios['building_name'][0]),
                conn
            )
        )
        self.building_zones.index = self.building_zones['zone_name']

        # Add constant timeseries in disturbance vector, if any CO2 model or HVAC or window.
        self.define_constant = (
            pd.notnull(self.building_scenarios['co2_model_type'][0])
            | pd.notnull(self.building_zones['hvac_ahu_type']).any()
            | pd.notnull(self.building_zones['window_type']).any()
        )

        # Define sets.

        # State variables.
        self.set_states = pd.Index(  # Create an series / array / set of indexes.
            pd.concat([  # Concatenate pandas series objects along the (default) row axis.
                # Zone temperature.
                self.building_zones['zone_name'] + '_temperature',

                # Surface temperature.
                self.building_surfaces_adiabatic['surface_name'][
                    self.building_surfaces_adiabatic['heat_capacity'].apply(self.parse_parameter) != 0.0
                ] + '_temperature',
                self.building_surfaces_exterior['surface_name'][
                    self.building_surfaces_exterior['heat_capacity'].apply(self.parse_parameter) != 0.0
                ] + '_temperature',
                self.building_surfaces_interior['surface_name'][
                    self.building_surfaces_interior['heat_capacity'].apply(self.parse_parameter) != 0.0
                ] + '_temperature',

                # Zone CO2 concentration.
                self.building_zones['zone_name'][
                    (pd.notnull(self.building_zones['hvac_ahu_type']) | pd.notnull(self.building_zones['window_type']))
                    & pd.notnull(self.building_scenarios['co2_model_type'][0])
                ] + '_co2_concentration',

                # Zone absolute humidity.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & pd.notnull(self.building_scenarios['humidity_model_type'][0])
                ] + '_absolute_humidity',

                # Radiator temperatures.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                ] + '_radiator_water_mean_temperature',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                ] + '_radiator_hull_front_temperature',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                ] + '_radiator_hull_rear_temperature',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                    & (self.building_zones['radiator_panel_number'] == '2')
                ] + '_radiator_panel_1_hull_rear_temperature',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                    & (self.building_zones['radiator_panel_number'] == '2')
                ] + '_radiator_panel_2_hull_front_temperature',

                # Sensible storage state of charge.
                self.building_scenarios['building_name'][
                    self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage'
                ] + '_sensible_thermal_storage_state_of_charge',

                # Battery storage state of charge.
                self.building_scenarios['building_name'][
                    self.building_scenarios['building_storage_type'] == 'default_battery_storage'
                ] + '_battery_storage_state_of_charge'
            ]),
            name='state_name'
        )

        # Control variables.
        self.set_controls = pd.Index(
            pd.concat([
                # Generic HVAC system.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_generic_type'])
                ] + '_generic_heat_thermal_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_generic_type'])
                ] + '_generic_cool_thermal_power',

                # Radiator thermal power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                ] + '_radiator_thermal_power',

                # AHU.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_heat_air_flow',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_cool_air_flow',

                # TU.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                ] + '_tu_heat_air_flow',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                ] + '_tu_cool_air_flow',

                # Windows.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['window_type'])
                ] + '_window_air_flow',

                # Sensible storage charge.
                # TODO: Add consideration for sensible storage heating / cooling.
                # self.building_scenarios['building_name'][
                #     self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage'
                # ] + '_sensible_storage_charge_heat_thermal_power',
                self.building_scenarios['building_name'][
                    self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage'
                ] + '_sensible_storage_charge_cool_thermal_power',

                # Battery storage charge.
                self.building_scenarios['building_name'][
                    self.building_scenarios['building_storage_type'] == 'default_battery_storage'
                ] + '_battery_storage_charge_electric_power',

                # Sensible storage discharge to AHU.
                # TODO: Add consideration for sensible storage heating / cooling.
                # self.building_zones['zone_name'][
                #     pd.notnull(self.building_zones['hvac_ahu_type'])
                #     & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                # ] + '_sensible_storage_to_zone_ahu_heat_thermal_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_to_zone_ahu_cool_thermal_power',

                # Sensible storage discharge to TU.
                # TODO: Add consideration for sensible storage heating / cooling.
                # self.building_zones['zone_name'][
                #     pd.notnull(self.building_zones['hvac_tu_type'])
                #     & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                # ] + '_sensible_storage_to_zone_tu_heat_thermal_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_to_zone_tu_cool_thermal_power',

                # Battery storage discharge to AHU.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_ahu_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_ahu_cool_electric_power',

                # Battery storage discharge to TU.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_tu_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_tu_cool_electric_power',
            ]),
            name='control_name'
        )

        # Disturbance variables.
        self.set_disturbances = pd.Index(
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
                    # 'storage_ambient_air_temperature'  # TODO: Check if storage_ambient_air_temperature still needed.
                ]),

                # Internal gains.
                pd.Series(self.building_zones['internal_gain_type'].unique() + '_occupancy'),
                pd.Series(self.building_zones['internal_gain_type'].unique() + '_appliances'),

                # Constant (workaround for constant model terms).
                (pd.Series(['constant']) if self.define_constant else None)
            ]),
            name='disturbance_name'
        )

        # Output variables.
        self.set_outputs = pd.Index(
            pd.concat([
                # Zone temperature.
                self.building_zones['zone_name'] + '_temperature',

                # Zone CO2 concentration.
                self.building_zones['zone_name'][
                    (pd.notnull(self.building_zones['hvac_ahu_type']) | pd.notnull(self.building_zones['window_type']))
                    & pd.notnull(self.building_scenarios['co2_model_type'][0])
                ] + '_co2_concentration',

                # Zone absolute humidity.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & pd.notnull(self.building_scenarios['humidity_model_type'][0])
                ] + '_absolute_humidity',

                # Zone fresh air flow.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    | pd.notnull(self.building_zones['window_type'])
                ] + '_total_fresh_air_flow',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_fresh_air_flow',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['window_type'])
                ] + '_window_fresh_air_flow',

                # Generic HVAC system electric power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_generic_type'])
                ] + '_generic_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_generic_type'])
                ] + '_generic_cool_electric_power',

                # Radiator thermal power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                ] + '_radiator_thermal_power',

                # Radiator electric power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_radiator_type'])
                ] + '_radiator_electric_power',

                # AHU electric power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_cool_electric_power_cooling',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_cool_electric_power_heating',

                # AHU thermal power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_cool_thermal_power_cooling',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                ] + '_ahu_cool_thermal_power_heating',

                # TU electric power.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                ] + '_tu_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                ] + '_tu_cool_electric_power',

                # Sensible storage state of charge.
                self.building_scenarios['building_name'][
                    (self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage')
                ] + '_sensible_thermal_storage_state_of_charge',

                # Battery storage state of charge.
                self.building_scenarios['building_name'][
                    (self.building_scenarios['building_storage_type'] == 'default_battery_storage')
                ] + '_battery_storage_state_of_charge',

                # Sensible storage discharge.
                # TODO: Add consideration for sensible storage heating / cooling.
                # self.building_zones['zone_name'][
                #     pd.notnull(self.building_zones['hvac_ahu_type'])
                #     & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                # ] + '_sensible_storage_to_zone_ahu_heat_thermal_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_to_zone_tu_heat_thermal_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_to_zone_ahu_cool_thermal_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_to_zone_tu_cool_thermal_power',

                # Battery storage discharge.
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_ahu_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_tu_heat_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_ahu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_ahu_cool_electric_power',
                self.building_zones['zone_name'][
                    pd.notnull(self.building_zones['hvac_tu_type'])
                    & (self.building_scenarios['building_storage_type'][0] == 'default_battery_storage')
                ] + '_battery_storage_to_zone_tu_cool_electric_power',

                # Sensible storage charge.
                self.building_scenarios['building_name'][
                    (self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_charge_heat_thermal_power',
                self.building_scenarios['building_name'][
                    (self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage')
                ] + '_sensible_storage_charge_cool_thermal_power',
                # TODO: Add consideration for sensible storage heating / cooling.
                # self.building_scenarios['building_name'][
                #     self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage'
                # ] + '_sensible_storage_charge_heat_electric_power',
                self.building_scenarios['building_name'][
                    self.building_scenarios['building_storage_type'] == 'default_sensible_thermal_storage'
                ] + '_sensible_storage_charge_cool_electric_power',

                # Battery storage charge.
                self.building_scenarios['building_name'][
                    (self.building_scenarios['building_storage_type'] == 'default_battery_storage')
                ] + '_battery_storage_charge_electric_power',

                # Validation outputs.
                self.building_surfaces_exterior['surface_name'] + '_irradiation_gain_exterior',
                self.building_surfaces_exterior['surface_name'] + '_convection_interior'
            ]),
            name='output_name'
        )

        # Define timesteps.
        self.timestep_delta = pd.to_timedelta(self.building_scenarios['time_step'][0])
        self.set_timesteps = pd.Index(
            pd.date_range(
                start=pd.to_datetime(self.building_scenarios['time_start'][0]),
                end=pd.to_datetime(self.building_scenarios['time_end'][0]),
                freq=self.timestep_delta
            ),
            name='time'
        )

        # Instantiate empty state space model matrices.
        self.state_matrix = pd.DataFrame(
            0.0,
            self.set_states,
            self.set_states
        )
        self.control_matrix = pd.DataFrame(
            0.0,
            self.set_states,
            self.set_controls
        )
        self.disturbance_matrix = pd.DataFrame(
            0.0,
            self.set_states,
            self.set_disturbances
        )
        self.state_output_matrix = pd.DataFrame(
            0.0,
            self.set_outputs,
            self.set_states
        )
        self.control_output_matrix = pd.DataFrame(
            0.0,
            self.set_outputs,
            self.set_controls
        )
        self.disturbance_output_matrix = pd.DataFrame(
            0.0,
            self.set_outputs,
            self.set_disturbances
        )

        # Define heat capacity vector.
        # TODO: Move heat capacity calculation to `coefficients define_coefficients_...`.
        self.heat_capacity_vector = pd.Series(
            0.0,
            self.set_states
        )
        for index, row in self.building_zones.iterrows():
            self.heat_capacity_vector.at[index] = (
                    self.parse_parameter(row['zone_area'])
                    * self.parse_parameter(row['zone_height'])
                    * self.parse_parameter(row['heat_capacity'])
            )
        for index, row in self.building_surfaces_adiabatic.iterrows():
            self.heat_capacity_vector.at[index] = (
                    self.parse_parameter(row['surface_area'])
                    * self.parse_parameter(row['heat_capacity'])
            )
        for index, row in self.building_surfaces_exterior.iterrows():
            self.heat_capacity_vector.at[index] = (
                    self.parse_parameter(row['surface_area'])
                    * self.parse_parameter(row['heat_capacity'])
            )
        for index, row in self.building_surfaces_interior.iterrows():
            self.heat_capacity_vector.at[index] = (
                    self.parse_parameter(row['surface_area'])
                    * self.parse_parameter(row['heat_capacity'])
            )

        # Instantiate other model properties / variables.
        self.set_state_initial = pd.Series(
            0.0,
            index=self.set_states
        )
        self.disturbance_timeseries = pd.DataFrame()
        self.electricity_price_timeseries = pd.DataFrame()

        # Define initial state.
        self.define_initial_state()

        # Calculate parameters / coefficients.
        self.calculate_coefficients_zone()
        self.calculate_coefficients_surface()
        self.calculate_coefficients_radiator()

        # Define heat, CO2 and humidity transfers.
        self.define_heat_transfer_surfaces_exterior()
        self.define_heat_transfer_surfaces_interior()
        self.define_heat_transfer_surfaces_adiabatic()
        self.define_heat_transfer_infiltration()
        self.define_heat_transfer_internal_gains()
        self.define_heat_transfer_hvac_generic()
        self.define_heat_transfer_hvac_radiator()
        self.define_heat_transfer_hvac_ahu()
        self.define_heat_transfer_hvac_tu()
        self.define_co2_transfer_hvac_ahu()
        self.define_heat_transfer_window_air_flow()
        self.define_humidity_transfer_hvac_ahu()
        self.define_storage_state_of_charge()

        # Define outputs.
        self.define_output_zone_temperature()
        self.define_output_zone_co2_concentration()
        self.define_output_zone_humidity()
        self.define_output_hvac_generic_power()
        self.define_output_hvac_radiator_power()
        self.define_output_hvac_ahu_electric_power()
        self.define_output_hvac_tu_electric_power()
        self.define_output_fresh_air_flow()
        self.define_output_window_fresh_air_flow()
        self.define_output_ahu_fresh_air_flow()
        self.define_output_storage_state_of_charge()
        self.define_output_storage_charge()
        self.define_output_storage_discharge()

        # Define validation outputs.
        self.define_output_surfaces_exterior_irradiation_gain_exterior()
        self.define_output_surfaces_exterior_convection_interior()

        # Define timeseries.
        self.load_disturbance_timeseries(conn)
        self.load_electricity_price_timeseries(conn)
        self.define_output_constraint_timeseries(conn)

        # Convert to time discrete model.
        self.discretize_model()

    def parse_parameter(self, parameter):
        """
        - Checks if given parameter is a number.
        - If the parameter is not a number, it is assumed to be the name of a parameter from `building_parameters`
        """
        # TODO: Apply parameter parsing when loading data instead when using data.
        try:
            return float(parameter)
        except ValueError:
            return self.building_parameters[parameter]

    def define_initial_state(self):
        """Define initial value of the state vector for given definition in `building_initial_state_types`."""

        # Zone air temperature.
        self.set_state_initial.loc[
            self.set_state_initial.index.isin(self.building_zones['zone_name'] + '_temperature')
        ] = (
            self.parse_parameter(self.building_scenarios['initial_zone_temperature'])
        )

        # Surface temperature.
        self.set_state_initial.loc[
            self.set_state_initial.index.isin(
                pd.concat([
                    self.building_surfaces_adiabatic['surface_name'] + '_temperature',
                    self.building_surfaces_exterior['surface_name'] + '_temperature',
                    self.building_surfaces_interior['surface_name'] + '_temperature'
                ])
            )
        ] = (
            self.parse_parameter(self.building_scenarios['initial_surface_temperature'])
        )

        # CO2 concentration.
        self.set_state_initial.loc[
            self.set_state_initial.index.isin(self.building_zones['zone_name'] + '_co2_concentration')
        ] = (
            self.parse_parameter(self.building_scenarios['initial_co2_concentration'])
        )

        # Zone air absolute humidity.
        self.set_state_initial.loc[
            self.set_state_initial.index.isin(self.building_zones['zone_name'] + '_absolute_humidity')
        ] = (
            self.parse_parameter(self.building_scenarios['initial_absolute_humidity'])
        )

        # Sensible storage state of charge.
        self.set_state_initial.loc[
            self.set_state_initial.index.str.contains('_sensible_thermal_storage_state_of_charge')
        ] = (
            self.parse_parameter(self.building_scenarios['initial_sensible_thermal_storage_state_of_charge'])
        )

        # Battery storage state of charge.
        self.set_state_initial.loc[
            self.set_state_initial.index.str.contains('_battery_storage_state_of_charge')
        ] = (
            self.parse_parameter(self.building_scenarios['initial_battery_storage_state_of_charge'])
        )

    def calculate_coefficients_zone(self):
        """Calculate zone parameters / heat transfer coefficients for use in, e.g., surface and radiator models."""

        # Instantiate columns for parameters / heat transfer coefficients.
        self.building_zones['zone_surfaces_wall_area'] = None
        self.building_zones['zone_surfaces_window_area'] = None
        self.building_zones['zone_surfaces_wall_emissivity'] = None
        self.building_zones['zone_surfaces_window_emissivity'] = None

        # Calculate zone parameters / heat transfer coefficients.
        for zone_name, zone_data in self.building_zones.iterrows():
            # Collect all surfaces adjacent to the zone.
            zone_surfaces = (
                pd.concat(
                    [
                        self.building_surfaces_exterior.loc[
                            self.building_surfaces_exterior['zone_name'].isin([zone_name]),
                            :
                        ],
                        self.building_surfaces_interior.loc[
                            self.building_surfaces_interior['zone_name'].isin([zone_name]),
                            :
                        ],
                        self.building_surfaces_interior.loc[
                            self.building_surfaces_interior['zone_adjacent_name'].isin([zone_name]),
                            :
                        ],
                        self.building_surfaces_adiabatic.loc[
                            self.building_surfaces_adiabatic['zone_name'].isin([zone_name]),
                            :
                        ]
                    ],
                    sort=False
                )
            )

            # Calculate parameters / heat transfer coefficients.
            self.building_zones.at[zone_name, 'zone_surfaces_wall_area'] = (
                (
                    zone_surfaces['surface_area'].apply(self.parse_parameter)
                    * (1 - zone_surfaces['window_wall_ratio'].apply(self.parse_parameter))
                ).sum()
            )
            self.building_zones.at[zone_name, 'zone_surfaces_window_area'] = (
                (
                    zone_surfaces['surface_area'].apply(self.parse_parameter)
                    * zone_surfaces['window_wall_ratio'].apply(self.parse_parameter)
                ).sum()
            )
            self.building_zones.at[zone_name, 'zone_surfaces_wall_emissivity'] = (
                zone_surfaces['emissivity'].apply(self.parse_parameter).mean()
            )
            # TODO: Ignore surfaces with no windows.
            self.building_zones.at[zone_name, 'zone_surfaces_window_emissivity'] = (
                zone_surfaces['emissivity_window'].apply(self.parse_parameter).mean()
            )

    def calculate_coefficients_surface(self):
        """Calculate heat transfer coefficients for the surface models."""

        # Instantiate columns for heat transfer coefficients.
        self.building_surfaces_exterior['heat_transfer_coefficient_surface_sky'] = None
        self.building_surfaces_exterior['heat_transfer_coefficient_surface_ground'] = None
        self.building_surfaces_exterior['heat_transfer_coefficient_window_sky'] = None
        self.building_surfaces_exterior['heat_transfer_coefficient_window_ground'] = None

        # Calculate heat transfer coefficients.
        for surface_name, surface_data in self.building_surfaces_exterior.iterrows():
            surface_data['heat_transfer_coefficient_surface_sky'] = (
                4.0
                * self.parse_parameter('stefan_boltzmann_constant')
                * self.parse_parameter(surface_data['emissivity'])
                * self.parse_parameter(surface_data['sky_view_factor'])
                * (
                    self.parse_parameter(self.building_scenarios['linearization_exterior_surface_temperature'])
                    / 2.0
                    + self.parse_parameter(self.building_scenarios['linearization_sky_temperature'])
                    / 2.0
                    + 273.15
                ) ** 3
            )
            surface_data['heat_transfer_coefficient_surface_ground'] = (
                4.0
                * self.parse_parameter('stefan_boltzmann_constant')
                * self.parse_parameter(surface_data['emissivity'])
                * (1.0 - self.parse_parameter(surface_data['sky_view_factor']))
                * (
                    self.parse_parameter(self.building_scenarios['linearization_exterior_surface_temperature'])
                    / 2.0
                    + self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature'])
                    / 2.0
                    + 273.15
                ) ** 3
            )
            if pd.notnull(surface_data['window_type']):
                surface_data['heat_transfer_coefficient_window_sky'] = (
                    4.0
                    * self.parse_parameter('stefan_boltzmann_constant')
                    * self.parse_parameter(surface_data['emissivity_window'])
                    * self.parse_parameter(surface_data['sky_view_factor'])
                    * (
                        self.parse_parameter(self.building_scenarios['linearization_exterior_surface_temperature'])
                        / 2.0
                        + self.parse_parameter(self.building_scenarios['linearization_sky_temperature'])
                        / 2.0
                        + 273.15
                    ) ** 3
                )
                surface_data['heat_transfer_coefficient_window_ground'] = (
                    4.0
                    * self.parse_parameter('stefan_boltzmann_constant')
                    * self.parse_parameter(surface_data['emissivity_window'])
                    * (1.0 - self.parse_parameter(surface_data['sky_view_factor']))
                    * (
                        self.parse_parameter(self.building_scenarios['linearization_exterior_surface_temperature'])
                        / 2.0
                        + self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature'])
                        / 2.0
                        + 273.15
                    ) ** 3
                )

    def calculate_coefficients_radiator(self):
        """Calculate heat transfer coefficients for the radiator model."""

        if pd.notnull(self.building_zones['hvac_radiator_type']).any():
            # Instantiate columns for heat transfer coefficients.
            self.building_zones['heat_capacitance_hull'] = None
            self.building_zones['thermal_resistance_radiator_hull_conduction'] = None
            self.building_zones['thermal_resistance_radiator_front_zone'] = None
            self.building_zones['thermal_resistance_radiator_front_surfaces'] = None
            self.building_zones['thermal_resistance_radiator_front_zone_surfaces'] = None
            self.building_zones['thermal_resistance_radiator_rear_zone'] = None
            self.building_zones['thermal_resistance_radiator_rear_surfaces'] = None
            self.building_zones['thermal_resistance_radiator_rear_zone_surfaces'] = None

            # Instantiate additional columns for multi-panel radiators.
            if (self.building_zones['radiator_panel_number'] == '2').any():
                self.building_zones['thermal_resistance_radiator_panel_1_rear_zone'] = None
                self.building_zones['thermal_resistance_radiator_panel_2_front_zone'] = None

            # Calculate heat transfer coefficients.
            for zone_name, zone_data in self.building_zones.iterrows():
                if pd.notnull(zone_data['hvac_radiator_type']):
                    # Calculate geometric parameters and heat capacity.
                    thickness_water_layer = (
                        self.parse_parameter(zone_data['radiator_water_volume'])
                        / self.parse_parameter(zone_data['radiator_panel_area'])
                    )
                    thickness_hull_layer = (
                        # Thickness for hull on one side of the panel.
                        0.5 * (
                            self.parse_parameter(zone_data['radiator_panel_thickness'])
                            - thickness_water_layer
                        )
                    )
                    radiator_hull_volume = (
                        # Volume for hull on one side of the panel.
                        thickness_hull_layer
                        * self.parse_parameter(zone_data['radiator_panel_area'])
                    )
                    self.building_zones.at[zone_name, 'heat_capacitance_hull'] = (
                        radiator_hull_volume
                        * self.parse_parameter(zone_data['radiator_hull_heat_capacity'])
                    )
                    self.building_zones.at[zone_name, 'heat_capacitance_water'] = (
                        self.parse_parameter(zone_data['radiator_water_volume'])
                        * self.parse_parameter('water_specific_heat')
                    )

                    # Calculate fundamental thermal resistances.
                    thermal_resistance_conduction = (
                        thickness_hull_layer
                        / (
                            self.parse_parameter(zone_data['radiator_hull_conductivity'])
                            * self.parse_parameter(zone_data['radiator_panel_area'])
                        )
                    )
                    thermal_resistance_convection = (
                        1.0
                        / (
                            self.parse_parameter(zone_data['radiator_convection_coefficient'])
                            * self.parse_parameter(zone_data['radiator_panel_area'])
                        )
                    )
                    temperature_radiator_surfaces_mean = (
                        0.5
                        * (
                            0.5
                            * (
                                self.parse_parameter(zone_data['radiator_supply_temperature_nominal'])
                                + self.parse_parameter(zone_data['radiator_return_temperature_nominal'])
                            )
                            + self.parse_parameter(self.building_scenarios['linearization_surface_temperature'])
                        )
                    )
                    thermal_resistance_radiation_front = (
                        (
                            (1.0 / self.parse_parameter(zone_data['radiator_panel_area']))
                            + (
                                (1.0 - self.parse_parameter(zone_data['radiator_emissivity']))
                                / (
                                    self.parse_parameter(zone_data['radiator_panel_area'])
                                    * self.parse_parameter(zone_data['radiator_emissivity'])
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
                                4.0 * self.parse_parameter('stefan_boltzmann_constant')
                                * (temperature_radiator_surfaces_mean ** 3.0)
                            )
                        )
                    )
                    thermal_resistance_radiation_rear = (
                        (
                            (1.0 / self.parse_parameter(zone_data['radiator_panel_area']))
                            + (
                                (1.0 - self.parse_parameter(zone_data['radiator_emissivity']))
                                / (
                                    self.parse_parameter(zone_data['radiator_panel_area'])
                                    * self.parse_parameter(zone_data['radiator_emissivity'])
                                )
                            )
                            + (
                                # TODO: Use total zone surface area and emissivity?
                                (1.0 - zone_data['zone_surfaces_wall_emissivity'])
                                / (
                                    self.parse_parameter(zone_data['radiator_panel_area'])
                                    * zone_data['zone_surfaces_wall_emissivity']
                                )
                            )
                        )
                        / (
                            (
                                4.0 * self.parse_parameter('stefan_boltzmann_constant')
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
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_hull_conduction'] = (
                        thermal_resistance_conduction
                    )
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_front_zone'] = (
                        thermal_resistance_star_sum_front / thermal_resistance_radiation_front
                    )
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_front_surfaces'] = (
                        thermal_resistance_star_sum_front / thermal_resistance_convection
                    )
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_front_zone_surfaces'] = (
                        thermal_resistance_star_sum_front / (0.5 * thermal_resistance_conduction)
                    )
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_rear_zone'] = (
                        thermal_resistance_star_sum_rear / thermal_resistance_radiation_rear
                    )
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_rear_surfaces'] = (
                        thermal_resistance_star_sum_rear / thermal_resistance_convection
                    )
                    self.building_zones.at[zone_name, 'thermal_resistance_radiator_rear_zone_surfaces'] = (
                        thermal_resistance_star_sum_rear / (0.5 * thermal_resistance_conduction)
                    )

                    if (self.building_zones['radiator_panel_number'] == '2').any():
                        thermal_resistance_convection_fin = (
                            1.0
                            / (
                                thermal_resistance_convection
                                * self.parse_parameter(zone_data['radiator_fin_effectiveness'])
                            )
                        )

                        self.building_zones.at[zone_name, 'thermal_resistance_radiator_panel_1_rear_zone'] = (
                            0.5 * thermal_resistance_conduction
                            + thermal_resistance_convection
                        )
                        self.building_zones.at[zone_name, 'thermal_resistance_radiator_panel_2_front_zone'] = (
                            0.5 * thermal_resistance_conduction
                            + thermal_resistance_convection_fin
                        )

    def define_heat_transfer_surfaces_exterior(self):
        """Thermal model: Exterior surfaces"""
        # TODO: Rename irradiation_gain_coefficient
        # TODO: Rename thermal_resistance_surface
        # TODO: Exterior window transmission factor
        for surface_name, surface_data in self.building_surfaces_exterior.iterrows():
            if self.parse_parameter(surface_data['heat_capacity']) != 0.0:  # Surfaces with non-zero heat capacity
                # Conductive heat transfer from the exterior towards the core of surface
                self.disturbance_matrix.at[
                    surface_name + '_temperature',
                    'irradiation_' + surface_data['direction_name']
                ] = (
                        self.disturbance_matrix.at[
                            surface_name + '_temperature',
                            'irradiation_' + surface_data['direction_name']
                        ]
                ) + (
                        self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )
                self.disturbance_matrix.at[
                    surface_name + '_temperature',
                    'ambient_air_temperature'
                ] = (
                        self.disturbance_matrix.at[
                            surface_name + '_temperature',
                            'ambient_air_temperature'
                        ]
                ) + (
                        (
                                self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                        )
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )
                self.disturbance_matrix.at[
                    surface_name + '_temperature',
                    'sky_temperature'
                ] = (
                        self.disturbance_matrix.at[
                            surface_name + '_temperature',
                            'sky_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )
                self.state_matrix.at[
                    surface_name + '_temperature',
                    surface_name + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_name + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )

                # Conductive heat transfer from the interior towards the core of surface
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_matrix.at[
                        surface_name + '_temperature',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_matrix.at[
                                surface_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / self.building_zones.loc[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    + (
                                            self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    )
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                            ) ** (- 1)
                            / self.heat_capacity_vector[surface_name]
                    )
                self.state_matrix.at[
                    surface_name + '_temperature',
                    surface_name + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_name + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )
                self.state_matrix.at[
                    surface_name + '_temperature',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )

                # Convective heat transfer from the surface towards zone
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_matrix.at[
                                surface_data['zone_name'] + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / self.building_zones.loc[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (1.0 - (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                            ) ** (- 1))
                            / self.heat_capacity_vector[surface_data['zone_name']]
                    )
                self.state_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    surface_name + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            surface_name + '_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.state_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
            else:  # Surfaces with neglected heat capacity
                # Complete convective heat transfer from surface to zone
                self.disturbance_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    'irradiation_' + surface_data['direction_name']
                ] = (
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + surface_data['direction_name']
                        ]
                ) + (
                        self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.disturbance_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    'ambient_air_temperature'
                ] = (
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'ambient_air_temperature'
                        ]
                ) + (
                        (
                                self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                        )
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.disturbance_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    'sky_temperature'
                ] = (
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'sky_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.state_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                + 1.0
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + 1.0
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_matrix.at[
                                surface_data['zone_name'] + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / self.building_zones.loc[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (1.0 - (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                            ) ** (- 1))
                            / self.heat_capacity_vector[surface_data['zone_name']]
                    )

            # Windows for each exterior surface - Modelled as surfaces with neglected heat capacity
            if self.parse_parameter(surface_data['window_wall_ratio']) != 0.0:
                # Complete convective heat transfer from surface to zone
                self.disturbance_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    'irradiation_' + surface_data['direction_name']
                ] = (
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'irradiation_' + surface_data['direction_name']
                        ]
                ) + (
                        self.parse_parameter(surface_data['irradiation_gain_coefficient_window'])
                        * self.parse_parameter(surface_data['surface_area'])
                        * self.parse_parameter(surface_data['window_wall_ratio'])
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.disturbance_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    'ambient_air_temperature'
                ] = (
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'ambient_air_temperature'
                        ]
                ) + (
                        (
                                self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                        )
                        * self.parse_parameter(surface_data['surface_area'])
                        * self.parse_parameter(surface_data['window_wall_ratio'])
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.disturbance_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    'sky_temperature'
                ] = (
                        self.disturbance_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            'sky_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                        * self.parse_parameter(surface_data['surface_area'])
                        * self.parse_parameter(surface_data['window_wall_ratio'])
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.state_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * self.parse_parameter(surface_data['window_wall_ratio'])
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                + 1.0
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + 1.0
                                / (self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1))
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_matrix.at[
                                surface_data['zone_name'] + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / self.building_zones.loc[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient_window'])
                            * self.parse_parameter(surface_data['surface_area'])
                            * self.parse_parameter(surface_data['window_wall_ratio'])
                            * (1.0 - (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_window_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1))
                            ) ** (- 1))
                            / self.heat_capacity_vector[surface_data['zone_name']]
                    )

    def define_heat_transfer_surfaces_interior(self):
        """Thermal model: Interior surfaces"""
        for surface_name, surface_data in self.building_surfaces_interior.iterrows():
            for zone_name in [surface_data['zone_name'], surface_data['zone_adjacent_name']]:
                if self.parse_parameter(surface_data['heat_capacity']) != 0.0:  # Surfaces with non-zero heat capacity
                    # Conductive heat transfer from the interior towards the core of surface
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                        self.building_surfaces_exterior['zone_name'] == zone_name
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            surface_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] = (
                                self.disturbance_matrix.at[
                                    surface_name + '_temperature',
                                    'irradiation_' + zone_exterior_surface_data['direction_name']
                                ]
                        ) + (
                                (
                                        self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                        * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                        / self.building_zones.loc[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                                * self.parse_parameter(surface_data['surface_area'])
                                * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                                * (
                                        1.0
                                        + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        / (
                                                2.0
                                                * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                        )
                                ) ** (- 1)
                                / self.heat_capacity_vector[surface_name]
                        )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                surface_name + '_temperature',
                                surface_name + '_temperature'
                            ]
                    ) + (
                            - 1.0
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                            ) ** (- 1)
                            / self.heat_capacity_vector[surface_name]
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        zone_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                surface_name + '_temperature',
                                zone_name + '_temperature'
                            ]
                    ) + (
                            self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                            ) ** (- 1)
                            / self.heat_capacity_vector[surface_name]
                    )

                    # Convective heat transfer from the surface towards zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                        self.building_surfaces_exterior['zone_name'] == zone_name
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            zone_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] = (
                                self.disturbance_matrix.at[
                                    zone_name + '_temperature',
                                    'irradiation_' + zone_exterior_surface_data['direction_name']
                                ]
                        ) + (
                                (
                                        self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                        * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                        / self.building_zones.loc[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                                * self.parse_parameter(surface_data['surface_area'])
                                * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                                * (1.0 - (
                                    1.0
                                    + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                                ) ** (- 1))
                                / self.heat_capacity_vector[zone_name]
                        )
                    self.state_matrix.at[
                        zone_name + '_temperature',
                        surface_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                zone_name + '_temperature',
                                surface_name + '_temperature'
                            ]
                    ) + (
                            self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                            ) ** (- 1)
                            / self.heat_capacity_vector[zone_name]
                    )
                    self.state_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                zone_name + '_temperature',
                                zone_name + '_temperature'
                            ]
                    ) + (
                            - 1.0
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                            ) ** (- 1)
                            / self.heat_capacity_vector[zone_name]
                    )
                else:  # Surfaces with neglected heat capacity
                    # Get adjacent / opposite zone_name
                    if zone_name == surface_data['zone_name']:
                        zone_adjacent_name = surface_data['zone_adjacent_name']
                    else:
                        zone_adjacent_name = surface_data['zone_name']

                    # Total adjacent zone surface area for later calculating share of interior (indirect) irradiation
                    zone_adjacent_surface_area = sum(
                        self.parse_parameter(zone_surface_data['surface_area'])
                        * (1 - self.parse_parameter(zone_surface_data['window_wall_ratio']))
                        for zone_surface_name, zone_surface_data in pd.concat(
                            [
                                self.building_surfaces_exterior[:][
                                    self.building_surfaces_exterior['zone_name'] == zone_adjacent_name
                                    ],
                                self.building_surfaces_interior[:][
                                    self.building_surfaces_interior['zone_name'] == zone_adjacent_name
                                    ],
                                self.building_surfaces_interior[:][
                                    self.building_surfaces_interior['zone_adjacent_name'] == zone_adjacent_name
                                    ],
                                self.building_surfaces_adiabatic[:][
                                    self.building_surfaces_adiabatic['zone_name'] == zone_adjacent_name
                                    ]
                            ],
                            sort=False
                        ).iterrows()  # For all surfaces adjacent to the zone
                    )

                    # Complete convective heat transfer from adjacent zone to zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                        self.building_surfaces_exterior['zone_name'] == zone_adjacent_name
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            zone_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] = (
                                self.disturbance_matrix.at[
                                    zone_name + '_temperature',
                                    'irradiation_' + zone_exterior_surface_data['direction_name']
                                ]
                        ) + (
                                (
                                        self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                        * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                        / zone_adjacent_surface_area
                                )  # Considers the share at the respective surface
                                * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                                * self.parse_parameter(surface_data['surface_area'])
                                * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                                * (
                                        1.0
                                        + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        / self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                ) ** (- 1)
                                / self.heat_capacity_vector[zone_name]
                        )
                    self.state_matrix.at[
                        zone_name + '_temperature',
                        zone_adjacent_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                zone_name + '_temperature',
                                zone_adjacent_name + '_temperature'
                            ]
                    ) + (
                            self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                            ) ** (- 1)
                            / self.heat_capacity_vector[zone_name]
                    )
                    self.state_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                zone_name + '_temperature',
                                zone_name + '_temperature'
                            ]
                    ) + (
                            - 1.0
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                            ) ** (- 1)
                            / self.heat_capacity_vector[zone_name]
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                        self.building_surfaces_exterior['zone_name'] == zone_name
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            zone_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] = (
                                self.disturbance_matrix.at[
                                    zone_name + '_temperature',
                                    'irradiation_' + zone_exterior_surface_data['direction_name']
                                ]
                        ) + (
                                (
                                        self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                        * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                        / self.building_zones.loc[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                                * self.parse_parameter(surface_data['surface_area'])
                                * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                                * (1.0 - (
                                    1.0
                                    + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    / self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                ) ** (- 1))
                                / self.heat_capacity_vector[zone_name]
                        )

                # Windows for each interior surface - Modelled as surfaces with neglected heat capacity
                if self.parse_parameter(surface_data['window_wall_ratio']) != 0.0:
                    # Get adjacent / opposite zone_name
                    if zone_name == surface_data['zone_name']:
                        zone_adjacent_name = surface_data['zone_adjacent_name']
                    else:
                        zone_adjacent_name = surface_data['zone_name']

                    # Total adjacent zone surface area for later calculating share of interior (indirect) irradiation
                    zone_adjacent_surface_area = sum(
                        self.parse_parameter(zone_surface_data['surface_area'])
                        * (1 - self.parse_parameter(zone_surface_data['window_wall_ratio']))
                        for zone_surface_name, zone_surface_data in pd.concat(
                            [
                                self.building_surfaces_exterior[:][
                                    self.building_surfaces_exterior['zone_name'] == zone_adjacent_name
                                    ],
                                self.building_surfaces_interior[:][
                                    self.building_surfaces_interior['zone_name'] == zone_adjacent_name
                                    ],
                                self.building_surfaces_interior[:][
                                    self.building_surfaces_interior['zone_adjacent_name'] == zone_adjacent_name
                                    ],
                                self.building_surfaces_adiabatic[:][
                                    self.building_surfaces_adiabatic['zone_name'] == zone_adjacent_name
                                    ]
                            ],
                            sort=False
                        ).iterrows()  # For all surfaces adjacent to the zone
                    )

                    # Complete convective heat transfer from adjacent zone to zone
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                        self.building_surfaces_exterior['zone_name'] == zone_adjacent_name
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            zone_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] = (
                                self.disturbance_matrix.at[
                                    zone_name + '_temperature',
                                    'irradiation_' + zone_exterior_surface_data['direction_name']
                                ]
                        ) + (
                                (
                                        self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                        * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                        / zone_adjacent_surface_area
                                )  # Considers the share at the respective surface
                                * self.parse_parameter(surface_data['irradiation_gain_coefficient_window'])
                                * self.parse_parameter(surface_data['surface_area'])
                                * self.parse_parameter(surface_data['window_wall_ratio'])
                                * (
                                        1.0
                                        + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                        / self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1)
                                ) ** (- 1)
                                / self.heat_capacity_vector[zone_name]
                        )
                    self.state_matrix.at[
                        zone_name + '_temperature',
                        zone_adjacent_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                zone_name + '_temperature',
                                zone_adjacent_name + '_temperature'
                            ]
                    ) + (
                            self.parse_parameter(surface_data['surface_area'])
                            * self.parse_parameter(surface_data['window_wall_ratio'])
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1)
                            ) ** (- 1)
                            / self.heat_capacity_vector[zone_name]
                    )
                    self.state_matrix.at[
                        zone_name + '_temperature',
                        zone_name + '_temperature'
                    ] = (
                            self.state_matrix.at[
                                zone_name + '_temperature',
                                zone_name + '_temperature'
                            ]
                    ) + (
                            - 1.0
                            * self.parse_parameter(surface_data['surface_area'])
                            * self.parse_parameter(surface_data['window_wall_ratio'])
                            * (
                                    1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + 1.0
                                    / self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1)
                            ) ** (- 1)
                            / self.heat_capacity_vector[zone_name]
                    )
                    for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                        self.building_surfaces_exterior['zone_name'] == zone_name
                    ].iterrows():
                        # Interior irradiation through all exterior surfaces adjacent to the zone
                        self.disturbance_matrix.at[
                            zone_name + '_temperature',
                            'irradiation_' + zone_exterior_surface_data['direction_name']
                        ] = (
                                self.disturbance_matrix.at[
                                    zone_name + '_temperature',
                                    'irradiation_' + zone_exterior_surface_data['direction_name']
                                ]
                        ) + (
                                (
                                        self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                        * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                        / self.building_zones.loc[zone_name, 'zone_surfaces_wall_area']
                                )  # Considers the share at the respective surface
                                * self.parse_parameter(surface_data['irradiation_gain_coefficient_window'])
                                * self.parse_parameter(surface_data['surface_area'])
                                * self.parse_parameter(surface_data['window_wall_ratio'])
                                * (1.0 - (
                                    1.0
                                    + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    + self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    / self.parse_parameter(surface_data['thermal_resistance_window']) ** (- 1)
                                ) ** (- 1))
                                / self.heat_capacity_vector[zone_name]
                        )

    def define_heat_transfer_surfaces_adiabatic(self):
        """Thermal model: Adiabatic surfaces"""
        for surface_name, surface_data in self.building_surfaces_adiabatic.iterrows():
            if self.parse_parameter(surface_data['heat_capacity']) != 0.0:  # Surfaces with non-zero heat capacity
                # Conductive heat transfer from the interior towards the core of surface
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_matrix.at[
                        surface_name + '_temperature',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_matrix.at[
                                surface_name + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / self.building_zones.loc[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (
                                    1.0
                                    + (
                                            self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                    )
                                    / (
                                            2.0
                                            * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                    )
                            ) ** (- 1)
                            / self.heat_capacity_vector[surface_name]
                    )
                self.state_matrix.at[
                    surface_name + '_temperature',
                    surface_name + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_name + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )
                self.state_matrix.at[
                    surface_name + '_temperature',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_name + '_temperature',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_name]
                )

                # Convective heat transfer from the surface towards zone
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_matrix.at[
                        surface_data['zone_name'] + '_temperature',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_matrix.at[
                                surface_data['zone_name'] + '_temperature',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / self.building_zones.loc[surface_data['zone_name'], 'zone_surfaces_wall_area']
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * self.parse_parameter(surface_data['surface_area'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (1.0 - (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                            ) ** (- 1))
                            / self.heat_capacity_vector[surface_data['zone_name']]
                    )
                self.state_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    surface_name + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            surface_name + '_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
                self.state_matrix.at[
                    surface_data['zone_name'] + '_temperature',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_matrix.at[
                            surface_data['zone_name'] + '_temperature',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                        / self.heat_capacity_vector[surface_data['zone_name']]
                )
            else:  # Surfaces with neglected heat capacity
                warnings.warn("Adiabatic surfaces with zero heat capacity have no effect: " + surface_name)

    def define_heat_transfer_infiltration(self):
        for index, row in self.building_zones.iterrows():
            self.state_matrix.at[
                index + '_temperature',
                index + '_temperature'
            ] = (
                    self.state_matrix.at[
                        index + '_temperature',
                        index + '_temperature'
                    ]
                    - self.parse_parameter(row['infiltration_rate'])
                    * self.parse_parameter('heat_capacity_air')
                    * self.parse_parameter(row['zone_area'])
                    * self.parse_parameter(row['zone_height'])
                    / self.heat_capacity_vector[index]
            )
            self.disturbance_matrix.at[
                index + '_temperature',
                'ambient_air_temperature'
            ] = (
                    self.disturbance_matrix.at[
                        index + '_temperature',
                        'ambient_air_temperature'
                    ]
                    + self.parse_parameter(row['infiltration_rate'])
                    * self.parse_parameter('heat_capacity_air')
                    * self.parse_parameter(row['zone_area'])
                    * self.parse_parameter(row['zone_height'])
                    / self.heat_capacity_vector[index]
            )

    def define_heat_transfer_window_air_flow(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['window_type']):
                self.control_matrix.at[
                    index + '_temperature',  # the "index" is the zone_name. Defined in line 87
                    index + '_window_air_flow'
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_window_air_flow'
                        ]
                        + self.parse_parameter('heat_capacity_air')
                        * (
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature'])
                                - self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_temperature_cool']
                                )
                        )
                        / self.heat_capacity_vector[index]
                )

    def define_heat_transfer_internal_gains(self):
        for index, row in self.building_zones.iterrows():
            self.disturbance_matrix.at[
                index + '_temperature',
                row['internal_gain_type'] + '_occupancy'
            ] = (
                    self.disturbance_matrix.at[
                        index + '_temperature',
                        row['internal_gain_type'] + '_occupancy'
                    ]
                    + self.parse_parameter(row['internal_gain_occupancy_factor'])
                    * self.parse_parameter(row['zone_area'])
                    / self.heat_capacity_vector[index]
            )
            self.disturbance_matrix.at[
                index + '_temperature',
                row['internal_gain_type'] + '_appliances'
            ] = (
                    self.disturbance_matrix.at[
                        index + '_temperature',
                        row['internal_gain_type'] + '_appliances'
                    ]
                    + self.parse_parameter(row['internal_gain_appliances_factor'])
                    * self.parse_parameter(row['zone_area'])
                    / self.heat_capacity_vector[index]
            )

    def define_heat_transfer_hvac_generic(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['hvac_generic_type']):
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_generic_heat_thermal_power'
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_generic_heat_thermal_power'
                        ]
                        + 1
                        / self.heat_capacity_vector[index]
                )
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_generic_cool_thermal_power'
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_generic_cool_thermal_power'
                        ]
                        - 1
                        / self.heat_capacity_vector[index]
                )

    def define_heat_transfer_hvac_radiator(self):
        """Define state equations describing the heat transfer occurring due to radiators."""

        if pd.notnull(self.building_zones['hvac_radiator_type']).any():
            zones_index = (
                pd.notnull(self.building_zones['hvac_radiator_type'])
            )
            zones_2_panels_index = (
                zones_index
                & (self.building_zones['radiator_panel_number'] == '2')
            )

            # Thermal power input to water.
            self.control_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power'
            ] += (
                1.0
                / self.building_zones.loc[zones_index, 'heat_capacitance_water']
            ).values

            # Heat transfer between radiator hull front and water.
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
            ] += (
                - 1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
            ] += (
                1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
            ] += (
                1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_water']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
            ] += (
                - 1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_water']
            ).values

            if zones_2_panels_index.any():
                # Heat transfer between radiator panel 1 hull rear and water.
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                ).values

                # Heat transfer between radiator panel 2 hull front and water.
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                ] += (
                    1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_water_mean_temperature'
                ] += (
                    - 1.0
                    / (0.5 * self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_hull_conduction'])
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_water']
                ).values

            # Heat transfer between radiator hull rear and water.
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
            ] += (
                - 1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
            ] += (
                1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
            ] += (
                1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_water']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_water_mean_temperature'
            ] += (
                - 1.0
                / (0.5 * self.building_zones.loc[zones_index, 'thermal_resistance_radiator_hull_conduction'])
                / self.building_zones.loc[zones_index, 'heat_capacitance_water']
            ).values

            # Heat transfer between radiator hull front and zone air.
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
            ] += (
                - 1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature'
            ] += (
                1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
            ] += (
                1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                / self.heat_capacity_vector[self.building_zones.loc[zones_index, 'zone_name']]
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature'
            ] += (
                - 1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_front_zone']
                / self.heat_capacity_vector[self.building_zones.loc[zones_index, 'zone_name']]
            ).values

            if zones_2_panels_index.any():
                # Heat transfer between radiator panel 1 hull rear and zone air.
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                ] += (
                    - 1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                ] += (
                    1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_1_hull_rear_temperature'
                ] += (
                    1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                    / self.heat_capacity_vector[self.building_zones.loc[zones_2_panels_index, 'zone_name']]
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                ] += (
                    - 1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_1_rear_zone']
                    / self.heat_capacity_vector[self.building_zones.loc[zones_2_panels_index, 'zone_name']]
                ).values

                # Heat transfer between radiator panel 2 hull front and zone air.
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                ] += (
                    - 1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                ] += (
                    1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                    / self.building_zones.loc[zones_2_panels_index, 'heat_capacitance_hull']
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_radiator_panel_2_hull_front_temperature'
                ] += (
                    1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                    / self.heat_capacity_vector[self.building_zones.loc[zones_2_panels_index, 'zone_name']]
                ).values
                self.state_matrix.loc[
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature',
                    self.building_zones.loc[zones_2_panels_index, 'zone_name'] + '_temperature'
                ] += (
                    - 1.0
                    / self.building_zones.loc[zones_2_panels_index, 'thermal_resistance_radiator_panel_2_front_zone']
                    / self.heat_capacity_vector[self.building_zones.loc[zones_2_panels_index, 'zone_name']]
                ).values

            # Heat transfer between radiator hull rear and zone air.
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
            ] += (
                - 1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature'
            ] += (
                1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
            ] += (
                1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                / self.heat_capacity_vector[self.building_zones.loc[zones_index, 'zone_name']]
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_temperature'
            ] += (
                - 1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_rear_zone']
                / self.heat_capacity_vector[self.building_zones.loc[zones_index, 'zone_name']]
            ).values

            # Heat transfer between radiator hull front / rear and zone surfaces.
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_front_temperature'
            ] += (
                - 1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_front_surfaces']
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            self.state_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_hull_rear_temperature'
            ] += (
                - 1.0
                / self.building_zones.loc[zones_index, 'thermal_resistance_radiator_rear_surfaces']
                / self.building_zones.loc[zones_index, 'heat_capacitance_hull']
            ).values
            for zone_name, zone_data in self.building_zones.loc[zones_index, :].iterrows():
                for surface_name, surface_data in (
                    pd.concat(
                        [
                            self.building_surfaces_exterior.loc[
                                self.building_surfaces_exterior['zone_name'].isin([zone_name]),
                                :
                            ],
                            self.building_surfaces_interior.loc[
                                self.building_surfaces_interior['zone_name'].isin([zone_name]),
                                :
                            ],
                            self.building_surfaces_interior.loc[
                                self.building_surfaces_interior['zone_adjacent_name'].isin([zone_name]),
                                :
                            ],
                            self.building_surfaces_adiabatic.loc[
                                self.building_surfaces_adiabatic['zone_name'].isin([zone_name]),
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
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        / zone_data['zone_surfaces_wall_area']
                        / zone_data['heat_capacitance_hull']
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        zone_name + '_radiator_hull_front_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_front_surfaces']
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        / zone_data['zone_surfaces_wall_area']
                        / self.heat_capacity_vector[surface_name]
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_front_surfaces']
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        / zone_data['zone_surfaces_wall_area']
                        / self.heat_capacity_vector[surface_name]
                    )

                    # Back.
                    self.state_matrix.at[
                        zone_name + '_radiator_hull_rear_temperature',
                        surface_name + '_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_rear_surfaces']
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        / zone_data['zone_surfaces_wall_area']
                        / zone_data['heat_capacitance_hull']
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        zone_name + '_radiator_hull_rear_temperature'
                    ] += (
                        1.0
                        / zone_data['thermal_resistance_radiator_rear_surfaces']
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        / zone_data['zone_surfaces_wall_area']
                        / self.heat_capacity_vector[surface_name]
                    )
                    self.state_matrix.at[
                        surface_name + '_temperature',
                        surface_name + '_temperature'
                    ] += (
                        - 1.0
                        / zone_data['thermal_resistance_radiator_rear_surfaces']
                        * self.parse_parameter(surface_data['surface_area'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        / zone_data['zone_surfaces_wall_area']
                        / self.heat_capacity_vector[surface_name]
                    )

    def define_heat_transfer_hvac_ahu(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['hvac_ahu_type']):
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_ahu_heat_air_flow'
                ] += (
                        self.parse_parameter('heat_capacity_air')
                        * (
                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_temperature_heat']
                                )
                        )
                        / self.heat_capacity_vector[index]
                )
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_ahu_cool_air_flow'
                ] += (
                        self.parse_parameter('heat_capacity_air')
                        * (
                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_temperature_cool']
                                )  # (20 - 25) < 0: same as TU
                        )
                        / self.heat_capacity_vector[index]
                )

    def define_heat_transfer_hvac_tu(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['hvac_tu_type']):
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_tu_heat_air_flow'
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_tu_heat_air_flow'
                        ]
                        + self.parse_parameter('heat_capacity_air')
                        * (
                                self.parse_parameter(row['tu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_temperature_heat']
                                )
                        )
                        / self.heat_capacity_vector[index]
                )
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_tu_cool_air_flow'
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_tu_cool_air_flow'
                        ]
                        + self.parse_parameter('heat_capacity_air')
                        * (
                                self.parse_parameter(row['tu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_temperature_cool']
                                )  # = (20 - 25) < 0 ==> make the temperature decrease in the room
                        )
                        / self.heat_capacity_vector[index]
                )

    def define_co2_transfer_hvac_ahu(self):
        if pd.notnull(self.building_scenarios['co2_model_type'][0]):
            for index, row in self.building_zones.iterrows():
                if pd.notnull(row['hvac_ahu_type']) | pd.notnull(row['window_type']):
                    self.state_matrix.at[
                        index + '_co2_concentration',
                        index + '_co2_concentration'
                    ] = (
                            self.state_matrix.at[
                                index + '_co2_concentration',
                                index + '_co2_concentration'
                            ]
                            - (
                                    self.parse_parameter(
                                        self.building_scenarios['linearization_ventilation_rate_per_square_meter']
                                    )
                                    / self.parse_parameter(row['zone_height'])
                            )
                    )
                    if pd.notnull(row['hvac_ahu_type']):
                        self.control_matrix.at[
                            index + '_co2_concentration',
                            index + '_ahu_heat_air_flow'
                        ] = (
                                self.control_matrix.at[
                                    index + '_co2_concentration',
                                    index + '_ahu_heat_air_flow'
                                ]
                                - (
                                        self.parse_parameter(self.building_scenarios['linearization_co2_concentration'])
                                        / self.parse_parameter(row['zone_height'])
                                        / self.parse_parameter(row['zone_area']))
                        )
                        self.control_matrix.at[
                            index + '_co2_concentration',
                            index + '_ahu_cool_air_flow'
                        ] = (
                                self.control_matrix.at[
                                    index + '_co2_concentration',
                                    index + '_ahu_cool_air_flow'
                                ]
                                - (
                                        self.parse_parameter(self.building_scenarios['linearization_co2_concentration'])
                                        / self.parse_parameter(row['zone_height'])
                                        / self.parse_parameter(row['zone_area']))
                        )
                    if pd.notnull(row['window_type']):
                        self.control_matrix.at[
                            index + '_co2_concentration',
                            index + '_window_air_flow'
                        ] = (
                                self.control_matrix.at[
                                    index + '_co2_concentration',
                                    index + '_window_air_flow'
                                ]
                                - (
                                        self.parse_parameter(self.building_scenarios['linearization_co2_concentration'])
                                        / self.parse_parameter(row['zone_height'])
                                        / self.parse_parameter(row['zone_area']))
                        )
                    # self.disturbance_matrix.at[index + '_co2_concentration', 'constant'] = (
                    #         self.disturbance_matrix.at[index + '_co2_concentration', 'constant']
                    #         - (
                    #                 self.parse_parameter(self.building_scenarios['linearization_co2_concentration']
                    #                 )
                    #                 * self.parse_parameter(row['infiltration_rate']))
                    # )  # TODO: Revise infiltration
                    self.disturbance_matrix.at[
                        index + '_co2_concentration',
                        row['internal_gain_type'] + '_occupancy'
                    ] = (
                            self.disturbance_matrix.at[
                                index + '_co2_concentration',
                                row['internal_gain_type'] + '_occupancy'
                            ]
                            + (
                                    self.parse_parameter('co2_generation_rate_per_person')
                                    / self.parse_parameter(row['zone_height'])
                                    / self.parse_parameter(row['zone_area'])
                            )
                    )
                    # division by zone_area since the occupancy here is in p
                    # if iterative and robust BM, no division by zone_area since the occupancy there is in p/m2
                    self.disturbance_matrix.at[
                        index + '_co2_concentration',
                        'constant'
                    ] = (
                            self.disturbance_matrix.at[
                                index + '_co2_concentration',
                                'constant'
                            ]
                            + (
                                    self.parse_parameter(
                                        self.building_scenarios['linearization_ventilation_rate_per_square_meter']
                                    )
                                    * self.parse_parameter(self.building_scenarios['linearization_co2_concentration'])
                                    / self.parse_parameter(row['zone_height'])
                            )
                    )

    def define_humidity_transfer_hvac_ahu(self):
        if pd.notnull(self.building_scenarios['humidity_model_type'][0]):
            for index, row in self.building_zones.iterrows():
                if pd.notnull(row['hvac_ahu_type']):
                    self.state_matrix.at[
                        index + '_absolute_humidity',
                        index + '_absolute_humidity'
                    ] = (
                            self.state_matrix.at[
                                index + '_absolute_humidity',
                                index + '_absolute_humidity'
                            ]
                            - (
                                    self.parse_parameter(
                                        self.building_scenarios['linearization_ventilation_rate_per_square_meter']
                                    )
                                    / self.parse_parameter(row['zone_height'])
                            )
                    )
                    self.control_matrix.at[index + '_absolute_humidity', index + '_ahu_heat_air_flow'] = (
                            self.control_matrix.at[index + '_absolute_humidity', index + '_ahu_heat_air_flow']
                            - (
                                    (
                                            self.parse_parameter(
                                                self.building_scenarios['linearization_zone_air_humidity_ratio']
                                            )
                                            - humid_air_properties(
                                                'W',
                                                'T',
                                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                                + 273.15,
                                                'R',
                                                self.parse_parameter(row['ahu_supply_air_relative_humidity_setpoint']),
                                                'P',
                                                101325
                                            )
                                    )
                                    / self.parse_parameter(row['zone_height'])
                                    / self.parse_parameter(row['zone_area'])
                            )
                    )
                    self.control_matrix.at[
                        index + '_absolute_humidity',
                        index + '_ahu_cool_air_flow'
                    ] = (
                            self.control_matrix.at[
                                index + '_absolute_humidity',
                                index + '_ahu_cool_air_flow'
                            ]
                            - (
                                    (
                                            self.parse_parameter(
                                                self.building_scenarios['linearization_zone_air_humidity_ratio']
                                            )
                                            - humid_air_properties(
                                                'W',
                                                'T',
                                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                                + 273.15,
                                                'R',
                                                self.parse_parameter(row['ahu_supply_air_relative_humidity_setpoint']),
                                                'P',
                                                101325
                                            )
                                    )
                                    / self.parse_parameter(row['zone_height'])
                                    / self.parse_parameter(row['zone_area'])
                            )
                    )
                    if pd.notnull(row['window_type']):
                        self.control_matrix.at[
                            index + '_absolute_humidity',
                            index + '_window_air_flow'
                        ] = (
                                self.control_matrix.at[
                                    index + '_absolute_humidity',
                                    index + '_window_air_flow'
                                ]
                                - (
                                        (
                                                self.parse_parameter(
                                                    self.building_scenarios['linearization_zone_air_humidity_ratio']
                                                )
                                                - self.parse_parameter(
                                                    self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                                )
                                        )
                                        / self.parse_parameter(row['zone_height'])
                                        / self.parse_parameter(row['zone_area'])
                                )
                        )
                    # self.disturbance_matrix.at[index + '_absolute_humidity', 'constant'] = (
                    #         self.disturbance_matrix.at[index + '_absolute_humidity', 'constant']
                    #         - (
                    #                 (self.parse_parameter(
                    #                     self.building_scenarios['linearization_zone_air_humidity_ratio'])
                    #                  - self.parse_parameter(
                    #                     self.building_scenarios['linearization_ambient_air_humidity_ratio']))
                    #                 * self.parse_parameter(row['infiltration_rate'])
                    #         )
                    # )  # TODO: Revise infiltration
                    self.disturbance_matrix.at[
                        index + '_absolute_humidity',
                        row['internal_gain_type'] + '_occupancy'
                    ] = (
                            self.disturbance_matrix.at[
                                index + '_absolute_humidity',
                                row['internal_gain_type'] + '_occupancy'
                            ]
                            + (
                                    self.parse_parameter('moisture_generation_rate_per_person')
                                    / self.parse_parameter(row['zone_height'])
                                    / self.parse_parameter(row['zone_area'])
                                    / self.parse_parameter('density_air')
                            )
                    )
                    self.disturbance_matrix.at[
                        index + '_absolute_humidity',
                        'constant'
                    ] = (
                            self.disturbance_matrix.at[
                                index + '_absolute_humidity',
                                'constant'
                            ]
                            + (
                                    self.parse_parameter(
                                        self.building_scenarios['linearization_ventilation_rate_per_square_meter']
                                    )
                                    * self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_humidity_ratio']
                                    )
                                    / self.parse_parameter(row['zone_height'])
                            )
                    )

    def define_storage_state_of_charge(self):
        # Sensible thermal storage.
        if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
            # Storage charge.
            self.control_matrix.at[
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_sensible_storage_charge_cool_thermal_power',
            ] = (
                self.control_matrix.at[
                    self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                    self.building_scenarios['building_name'][0] + '_sensible_storage_charge_cool_thermal_power',
                ]
            ) + (
                1.0
                / (
                    self.parse_parameter('water_specific_heat')
                    * self.parse_parameter(self.building_scenarios['storage_sensible_temperature_delta'])
                )
                * self.parse_parameter(self.building_scenarios['storage_round_trip_efficiency'])
            )

            # Storage losses.
            # - Thermal losses are considered negligible, but a very small loss is added to keep the state matrix
            #   non-singular and hence invertible.
            # - TODO: For detailed losses depending on the storage size see `cobmo/README_storage.md`
            self.state_matrix.at[
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge'
            ] = (
                self.state_matrix.at[
                    self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                    self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge'
                ]
            ) - 1e-17

            for zone_name, zone_data in self.building_zones.iterrows():
                # TODO: Differentiate heating / cooling and define heating discharge.

                if pd.notnull(zone_data['hvac_ahu_type']):
                    # Storage discharge to AHU for cooling.
                    self.control_matrix.at[
                        self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                        zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power',
                    ] = (
                        self.control_matrix.at[
                            self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                            zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power'
                        ]
                    ) + (
                        - 1.0
                        / (
                            self.parse_parameter('water_specific_heat')
                            * self.parse_parameter(self.building_scenarios['storage_sensible_temperature_delta'])
                        )
                    )

                if pd.notnull(zone_data['hvac_tu_type']):
                    # Storage discharge to TU for cooling.
                    self.control_matrix.at[
                        self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                        zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power',
                    ] = (
                        self.control_matrix.at[
                            self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power',
                        ]
                    ) + (
                        - 1.0
                        / (
                            self.parse_parameter('water_specific_heat')
                            * self.parse_parameter(self.building_scenarios['storage_sensible_temperature_delta'])
                        )
                    )

        # Battery storage.
        if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
            # Storage charge.
            self.control_matrix.at[
                self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_battery_storage_charge_electric_power'
            ] = self.control_matrix.at[
                self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_battery_storage_charge_electric_power'
            ] + (
                self.parse_parameter(self.building_scenarios['storage_round_trip_efficiency'])
            )  # TODO: Make the battery loss dependent on the outdoor temperature.

            # Storage losses.
            # - There are no battery storage losses, but a very small loss is added to keep the state matrix
            #   non-singular and hence invertible.
            self.state_matrix.at[
                self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge'
            ] = (
                self.state_matrix.at[
                    self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                    self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge'
                ]
            ) - 1E-17

            for zone_name, zone_data in self.building_zones.iterrows():
                # TODO: Differentiate heating / cooling and define heating discharge.

                if pd.notnull(zone_data['hvac_ahu_type']):
                    # Storage discharge to AHU for cooling.
                    self.control_matrix.at[
                        self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                        zone_name + '_battery_storage_to_zone_ahu_cool_electric_power',
                    ] = (
                        self.control_matrix.at[
                            self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_ahu_cool_electric_power',
                        ]
                    ) - 1.0

                    # Storage discharge to AHU for heating.
                    self.control_matrix.at[
                        self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                        zone_name + '_battery_storage_to_zone_ahu_heat_electric_power',
                    ] = (
                        self.control_matrix.at[
                            self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_ahu_heat_electric_power',
                        ]
                    ) - 1.0

                if pd.notnull(zone_data['hvac_tu_type']):
                    # Storage discharge to TU for cooling.
                    self.control_matrix.at[
                        self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                        zone_name + '_battery_storage_to_zone_tu_cool_electric_power',
                    ] = (
                        self.control_matrix.at[
                            self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_tu_cool_electric_power',
                        ]
                    ) - 1.0

                    # Storage discharge to TU for heating.
                    self.control_matrix.at[
                        self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                        zone_name + '_battery_storage_to_zone_tu_heat_electric_power',
                    ] = (
                        self.control_matrix.at[
                            self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                            zone_name + '_battery_storage_to_zone_tu_heat_electric_power',
                        ]
                    ) - 1.0

    def define_output_zone_temperature(self):
        for index, row in self.building_zones.iterrows():
            self.state_output_matrix.at[
                index + '_temperature',
                index + '_temperature'
            ] = 1

    def define_output_zone_co2_concentration(self):
        if pd.notnull(self.building_scenarios['co2_model_type'][0]):
            for index, row in self.building_zones.iterrows():
                if pd.notnull(row['hvac_ahu_type']) | pd.notnull(row['window_type']):
                    self.state_output_matrix.at[
                        index + '_co2_concentration',
                        index + '_co2_concentration'
                    ] = 1

    def define_output_zone_humidity(self):
        if pd.notnull(self.building_scenarios['humidity_model_type'][0]):
            for index, row in self.building_zones.iterrows():
                if pd.notnull(row['hvac_ahu_type']):
                    self.state_output_matrix.at[
                        index + '_absolute_humidity',
                        index + '_absolute_humidity'
                    ] = 1

    def define_output_hvac_generic_power(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['hvac_generic_type']):
                self.control_output_matrix.at[
                    index + '_generic_heat_electric_power',
                    index + '_generic_heat_thermal_power'
                ] = (
                        self.control_output_matrix.at[
                            index + '_generic_heat_electric_power',
                            index + '_generic_heat_thermal_power'
                        ]
                        + 1
                        / self.parse_parameter(row['generic_heating_efficiency'])
                )
                self.control_output_matrix.at[
                    index + '_generic_cool_electric_power',
                    index + '_generic_cool_thermal_power'
                ] = (
                        self.control_output_matrix.at[
                            index + '_generic_cool_electric_power',
                            index + '_generic_cool_thermal_power'
                        ]
                        + 1
                        / self.parse_parameter(row['generic_cooling_efficiency'])
                )

    def define_output_hvac_radiator_power(self):
        """Define output equations for the thermal and electric power demand due to radiators."""

        if pd.notnull(self.building_zones['hvac_radiator_type']).any():
            zones_index = (
                pd.notnull(self.building_zones['hvac_radiator_type'])
            )

            # Thermal power.
            self.control_output_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power'
            ] += (
                1.0
            )

            # Electric power.
            # TODO: Define heating COP for radiators.
            self.control_output_matrix.loc[
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_electric_power',
                self.building_zones.loc[zones_index, 'zone_name'] + '_radiator_thermal_power'
            ] += (
                1.0
            )

    def define_output_hvac_ahu_electric_power(self):
        for zone_name, zone_data in self.building_zones.iterrows():
            if pd.notnull(zone_data['hvac_ahu_type']):
                # Calculate parameters.
                ahu_supply_air_absolute_humidity_setpoint = (
                    cobmo.utils.calculate_absolute_humidity_humid_air(
                        self.parse_parameter(zone_data['ahu_supply_air_temperature_setpoint']),
                        self.parse_parameter(zone_data['ahu_supply_air_relative_humidity_setpoint'])
                    )
                )
                # TODO: Define zone linearization temperature and humidity in database.
                linearization_zone_air_temperature = (
                    0.5
                    * (
                        self.parse_parameter(self.building_scenarios['linearization_zone_air_temperature_heat'])
                        + self.parse_parameter(self.building_scenarios['linearization_zone_air_temperature_cool'])
                    )
                )
                linearization_zone_air_absolute_humidity = (
                    0.5
                    * (
                        self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                        + ahu_supply_air_absolute_humidity_setpoint
                    )
                )
                delta_enthalpy_ahu_recovery = (
                    cobmo.utils.calculate_enthalpy_humid_air(
                        linearization_zone_air_temperature,
                        linearization_zone_air_absolute_humidity
                    )
                    - cobmo.utils.calculate_enthalpy_humid_air(
                        self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature']),
                        linearization_zone_air_absolute_humidity
                    )
                )

                # Calculate enthalpies.
                # TODO: Remove unnecessary HVAC types.
                if (
                    zone_data['ahu_cooling_type'] == 'default'
                    and zone_data['ahu_heating_type'] == 'default'
                    and zone_data['ahu_dehumidification_type'] == 'default'
                    and zone_data['ahu_return_air_heat_recovery_type'] == 'default'
                ):
                    if (
                        self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                        <= ahu_supply_air_absolute_humidity_setpoint
                    ):
                        delta_enthalpy_ahu_cooling = min(
                            0.0,
                            cobmo.utils.calculate_enthalpy_humid_air(
                                self.parse_parameter(zone_data['ahu_supply_air_temperature_setpoint']),
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature']),
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                            )
                        )
                        delta_enthalpy_ahu_heating = max(
                            0.0,
                            cobmo.utils.calculate_enthalpy_humid_air(
                                self.parse_parameter(zone_data['ahu_supply_air_temperature_setpoint']),
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature']),
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                            )
                        )
                        delta_enthalpy_ahu_recovery_cooling = max(
                            delta_enthalpy_ahu_cooling,
                            min(
                                0.0,
                                self.parse_parameter(zone_data['ahu_return_air_heat_recovery_efficiency'])
                                * delta_enthalpy_ahu_recovery
                            )
                        )
                        delta_enthalpy_ahu_recovery_heating = min(
                            delta_enthalpy_ahu_heating,
                            max(
                                0.0,
                                self.parse_parameter(zone_data['ahu_return_air_heat_recovery_efficiency'])
                                * delta_enthalpy_ahu_recovery
                            )
                        )
                    else:
                        delta_enthalpy_ahu_cooling = (
                            cobmo.utils.calculate_dew_point_enthalpy_humid_air(
                                self.parse_parameter(zone_data['ahu_supply_air_temperature_setpoint']),
                                self.parse_parameter(zone_data['ahu_supply_air_relative_humidity_setpoint'])
                            )
                            - cobmo.utils.calculate_enthalpy_humid_air(
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature']),
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                            )
                        )
                        delta_enthalpy_ahu_heating = (
                            cobmo.utils.calculate_enthalpy_humid_air(
                                self.parse_parameter(zone_data['ahu_supply_air_temperature_setpoint']),
                                ahu_supply_air_absolute_humidity_setpoint
                            )
                            - cobmo.utils.calculate_dew_point_enthalpy_humid_air(
                                self.parse_parameter(zone_data['ahu_supply_air_temperature_setpoint']),
                                self.parse_parameter(zone_data['ahu_supply_air_relative_humidity_setpoint'])
                            )
                        )
                        delta_enthalpy_ahu_recovery_cooling = max(
                            delta_enthalpy_ahu_cooling,
                            min(
                                0.0,
                                self.parse_parameter(zone_data['ahu_return_air_heat_recovery_efficiency'])
                                * delta_enthalpy_ahu_recovery
                            )
                        )
                        delta_enthalpy_ahu_recovery_heating = 0.0

                # TODO: Revise code to remove `abs(delta_enthalpy...)` and invert `ahu_cooling_efficiency`.

                # AHU heating.
                # TODO: Split heat outputs into heating / cooling and add thermal power outputs.
                self.control_output_matrix.at[
                    zone_name + '_ahu_heat_electric_power',
                    zone_name + '_ahu_heat_air_flow'
                ] += (
                    self.parse_parameter('density_air')
                    * (
                        (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                        / self.parse_parameter(zone_data['ahu_cooling_efficiency'])
                        + (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                        / self.parse_parameter(zone_data['ahu_heating_efficiency'])
                        + self.parse_parameter(zone_data['ahu_fan_efficiency'])
                    )
                )

                # Sensible thermal storage heating discharge.
                # TODO: Add consideration for sensible storage heating / cooling.
                # if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                #     self.control_output_matrix.at[
                #         zone_name + '_ahu_heat_electric_power',
                #         zone_name + '_sensible_storage_to_zone_ahu_heat_thermal_power'
                #     ] += (
                #         - 1.0 / self.parse_parameter(zone_data['ahu_heating_efficiency'])
                #     )

                # Battery storage heating discharge.
                if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
                    self.control_output_matrix.at[
                        zone_name + '_ahu_heat_electric_power',
                        zone_name + '_battery_storage_to_zone_ahu_heat_electric_power'
                    ] += (
                        - 1.0
                    )

                # AHU cooling.
                self.control_output_matrix.at[
                    zone_name + '_ahu_cool_electric_power_cooling',
                    zone_name + '_ahu_cool_air_flow'
                ] += (
                    self.parse_parameter('density_air')
                    * (
                        (
                            abs(delta_enthalpy_ahu_cooling)
                            - abs(delta_enthalpy_ahu_recovery_cooling)
                        )
                        / self.parse_parameter(zone_data['ahu_cooling_efficiency'])
                    )
                )
                self.control_output_matrix.at[
                    zone_name + '_ahu_cool_thermal_power_cooling',
                    zone_name + '_ahu_cool_air_flow'
                ] += (
                    self.parse_parameter('density_air')
                    * (
                        abs(delta_enthalpy_ahu_cooling)
                        - abs(delta_enthalpy_ahu_recovery_cooling)
                    )
                )
                self.control_output_matrix.at[
                    zone_name + '_ahu_cool_electric_power_heating',
                    zone_name + '_ahu_cool_air_flow'
                ] += (
                    self.parse_parameter('density_air')
                    * (
                        (
                            abs(delta_enthalpy_ahu_heating)
                            - abs(delta_enthalpy_ahu_recovery_heating)
                        )
                        / self.parse_parameter(zone_data['ahu_heating_efficiency'])
                        + self.parse_parameter(zone_data['ahu_fan_efficiency'])  # TODO: Split off fan power.
                    )
                )
                self.control_output_matrix.at[
                    zone_name + '_ahu_cool_thermal_power_heating',
                    zone_name + '_ahu_cool_air_flow'
                ] += (
                    self.parse_parameter('density_air')
                    * (
                        abs(delta_enthalpy_ahu_heating)
                        - abs(delta_enthalpy_ahu_recovery_heating)
                    )
                )

                # TODO: Add thermal power for storage.
                # Sensible thermal storage cooling discharge.
                if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_electric_power_cooling',
                        zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power'
                    ] += (
                        - 1.0 / self.parse_parameter(zone_data['ahu_cooling_efficiency'])
                    )
                    # TODO: Add consideration for sensible storage heating / cooling.

                # Battery storage cooling discharge.
                if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_electric_power_cooling',
                        zone_name + '_battery_storage_to_zone_ahu_cool_electric_power'
                    ] += (
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
                    self.control_output_matrix.at[
                        zone_name + '_ahu_cool_electric_power_heating',
                        zone_name + '_battery_storage_to_zone_ahu_cool_electric_power'
                    ] += (
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

    def define_output_hvac_tu_electric_power(self):
        for zone_name, zone_data in self.building_zones.iterrows():
            if pd.notnull(zone_data['hvac_tu_type']):
                # Calculate enthalpies.
                if (
                    zone_data['tu_cooling_type'] == 'default'
                    and zone_data['tu_heating_type'] == 'default'
                ):
                    if zone_data['tu_air_intake_type'] == 'zone':
                        delta_enthalpy_tu_cooling = self.parse_parameter('heat_capacity_air') * (
                            self.parse_parameter(self.building_scenarios['linearization_zone_air_temperature_cool'])
                            - self.parse_parameter(zone_data['tu_supply_air_temperature_setpoint'])
                        )
                        delta_enthalpy_tu_heating = self.parse_parameter('heat_capacity_air') * (
                            self.parse_parameter(self.building_scenarios['linearization_zone_air_temperature_heat'])
                            - self.parse_parameter(zone_data['tu_supply_air_temperature_setpoint'])
                        )
                    elif zone_data['tu_air_intake_type'] == 'ahu':
                        delta_enthalpy_tu_cooling = self.parse_parameter('heat_capacity_air') * (
                            self.parse_parameter(self.building_scenarios['ahu_supply_air_temperature_setpoint'])
                            - self.parse_parameter(zone_data['tu_supply_air_temperature_setpoint'])
                        )
                        delta_enthalpy_tu_heating = self.parse_parameter('heat_capacity_air') * (
                            self.parse_parameter(self.building_scenarios['ahu_supply_air_temperature_setpoint'])
                            - self.parse_parameter(zone_data['tu_supply_air_temperature_setpoint'])
                        )

                # TU heating.
                self.control_output_matrix.at[
                    zone_name + '_tu_heat_electric_power',
                    zone_name + '_tu_heat_air_flow'
                ] = (
                    self.control_output_matrix.at[
                        zone_name + '_tu_heat_electric_power',
                        zone_name + '_tu_heat_air_flow'
                    ]
                    + self.parse_parameter('density_air')
                    * (
                        abs(delta_enthalpy_tu_heating)
                        / self.parse_parameter(zone_data['tu_heating_efficiency'])
                        + self.parse_parameter(zone_data['tu_fan_efficiency'])
                    )
                )

                # Sensible thermal storage heating discharge.
                # TODO: Add consideration for sensible storage heating / cooling.
                # if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                #     self.control_output_matrix.at[
                #         zone_name + '_tu_heat_electric_power',
                #         zone_name + '_sensible_storage_to_zone_tu_heat_thermal_power'
                #     ] = (
                #         self.control_output_matrix.at[
                #             zone_name + '_tu_heat_electric_power',
                #             zone_name + '_sensible_storage_to_zone_tu_heat_thermal_power'
                #         ]
                #         - 1.0 / self.parse_parameter(zone_data['tu_heating_efficiency'])
                #     )

                # Battery storage heating discharge.
                if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
                    self.control_output_matrix.at[
                        zone_name + '_tu_heat_electric_power',
                        zone_name + '_battery_storage_to_zone_tu_heat_electric_power'
                    ] = (
                        self.control_output_matrix.at[
                            zone_name + '_tu_heat_electric_power',
                            zone_name + '_battery_storage_to_zone_tu_heat_electric_power'
                        ]
                        - 1.0
                    )

                # TU cooling.
                self.control_output_matrix.at[
                    zone_name + '_tu_cool_electric_power',
                    zone_name + '_tu_cool_air_flow'
                ] = (
                    self.control_output_matrix.at[
                        zone_name + '_tu_cool_electric_power',
                        zone_name + '_tu_cool_air_flow']
                    + self.parse_parameter('density_air')
                    * (
                        abs(delta_enthalpy_tu_cooling)
                        / self.parse_parameter(zone_data['tu_cooling_efficiency'])
                        + self.parse_parameter(zone_data['tu_fan_efficiency'])
                    )
                )

                # Sensible thermal storage cooling discharge.
                if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                    self.control_output_matrix.at[
                        zone_name + '_tu_cool_electric_power',
                        zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power'
                    ] = (
                        self.control_output_matrix.at[
                            zone_name + '_tu_cool_electric_power',
                            zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power'
                        ]
                        - 1.0 / self.parse_parameter(zone_data['tu_cooling_efficiency'])
                    )

                # Battery storage cooling discharge.
                if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
                    self.control_output_matrix.at[
                        zone_name + '_tu_cool_electric_power',
                        zone_name + '_battery_storage_to_zone_tu_cool_electric_power'
                    ] = (
                        self.control_output_matrix.at[
                            zone_name + '_tu_cool_electric_power',
                            zone_name + '_battery_storage_to_zone_tu_cool_electric_power'
                        ]
                        - 1.0
                    )

    def define_output_fresh_air_flow(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['hvac_ahu_type']):
                self.control_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    index + '_ahu_heat_air_flow'
                ] = 1
                self.control_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    index + '_ahu_cool_air_flow'
                ] = 1
            if pd.notnull(row['window_type']):
                self.control_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    index + '_window_air_flow'
                ] = 1
            if pd.notnull(row['window_type']) | pd.notnull(row['hvac_ahu_type']):
                self.disturbance_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    'constant'
                ] = (
                        self.disturbance_output_matrix.at[
                            index + '_total_fresh_air_flow',
                            'constant'
                        ]
                        + self.parse_parameter(row['infiltration_rate'])
                        * self.parse_parameter(row['zone_area'])
                        * self.parse_parameter(row['zone_height'])
                )  # TODO: Revise infiltration

    def define_output_ahu_fresh_air_flow(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['hvac_ahu_type']):
                self.control_output_matrix.at[
                    index + '_ahu_fresh_air_flow',
                    index + '_ahu_heat_air_flow'
                ] = 1
                self.control_output_matrix.at[
                    index + '_ahu_fresh_air_flow',
                    index + '_ahu_cool_air_flow'
                ] = 1

    def define_output_window_fresh_air_flow(self):
        for index, row in self.building_zones.iterrows():
            if pd.notnull(row['window_type']):
                self.control_output_matrix.at[
                    index + '_window_fresh_air_flow',
                    index + '_window_air_flow'
                ] = 1

    def define_output_storage_state_of_charge(self):
        # Sensible thermal storage.
        if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
            self.state_output_matrix.at[
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge'
            ] = 1.0

        # Battery storage.
        if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
            self.state_output_matrix.at[
                self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge',
                self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge'
            ] = 1.0

    def define_output_storage_charge(self):
        # TODO: Remove redundant charge ouputs.

        # Sensible thermal storage charge thermal power.
        if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
            # Heating.
            # TODO: Add consideration for sensible storage heating / cooling.
            # self.control_output_matrix.at[
            #     self.building_scenarios['building_name'] + '_sensible_storage_charge_heat_thermal_power',
            #     self.building_scenarios['building_name'] + '_sensible_storage_charge_heat_thermal_power'
            # ] = 1.0

            # Cooling.
            self.control_output_matrix.at[
                self.building_scenarios['building_name'] + '_sensible_storage_charge_cool_thermal_power',
                self.building_scenarios['building_name'] + '_sensible_storage_charge_cool_thermal_power'
            ] = 1.0

        # Sensible thermal storage charge electric power.
        if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
            # Heating.
            # TODO: Add consideration for sensible storage heating / cooling.
            # self.control_output_matrix.at[
            #     self.building_scenarios['building_name'][0] + '_sensible_storage_charge_heat_electric_power',
            #     self.building_scenarios['building_name'][0] + '_sensible_storage_charge_heat_thermal_power'
            # ] = (
            #     self.control_output_matrix.at[
            #         self.building_scenarios['building_name'][0] + '_sensible_storage_charge_heat_electric_power',
            #         self.building_scenarios['building_name'][0] + '_sensible_storage_charge_heat_thermal_power'
            #     ]
            #     + (
            #         1.0
            #         / self.parse_parameter('hvac_ahu_heating_efficiency')
            #         * 1.0000001  # TODO: Very small loss to avoid simultaneous charge and discharge still needed?
            #     )
            # )

            # Cooling.
            self.control_output_matrix.at[
                self.building_scenarios['building_name'][0] + '_sensible_storage_charge_cool_electric_power',
                self.building_scenarios['building_name'][0] + '_sensible_storage_charge_cool_thermal_power'
            ] = (
                self.control_output_matrix.at[
                    self.building_scenarios['building_name'][0] + '_sensible_storage_charge_cool_electric_power',
                    self.building_scenarios['building_name'][0] + '_sensible_storage_charge_cool_thermal_power'
                ]
                + (
                    1.0
                    / self.parse_parameter('hvac_ahu_cooling_efficiency')  # TODO: This might be a problem for CONCEPT.
                    * 1.0000001  # TODO: Very small loss to avoid simultaneous charge and discharge still needed?
                )
            )

        # Battery storage charge.
        if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
            self.control_output_matrix.at[
                self.building_scenarios['building_name'] + '_battery_storage_charge_electric_power',
                self.building_scenarios['building_name'] + '_battery_storage_charge_electric_power'
            ] = 1.0

    def define_output_storage_discharge(self):
        for zone_name, zone_data in self.building_zones.iterrows():
            # Sensible thermal storage.
            if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                # TODO: Add consideration for sensible storage heating / cooling.
                # if pd.notnull(zone_data['hvac_ahu_type']):
                #     # Storage discharge to AHU for heating.
                #     self.control_output_matrix.at[
                #         zone_name + '_sensible_storage_to_zone_ahu_heat_thermal_power',
                #         zone_name + '_sensible_storage_to_zone_ahu_heat_thermal_power'
                #     ] = 1.0
                #
                # if pd.notnull(zone_data['hvac_tu_type']):
                #     # Storage discharge to TU for heating.
                #     self.control_output_matrix.at[
                #         zone_name + '_sensible_storage_to_zone_tu_heat_thermal_power',
                #         zone_name + '_sensible_storage_to_zone_tu_heat_thermal_power'
                #     ] = 1.0

                if pd.notnull(zone_data['hvac_ahu_type']):
                    # Storage discharge to AHU for cooling.
                    self.control_output_matrix.at[
                        zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power',
                        zone_name + '_sensible_storage_to_zone_ahu_cool_thermal_power'
                    ] = 1.0

                    # Storage discharge to AHU for heating.
                    self.control_output_matrix.at[
                        zone_name + '_sensible_storage_to_zone_ahu_heat_thermal_power',
                        zone_name + '_sensible_storage_to_zone_ahu_heat_thermal_power'
                    ] = 1.0

                if pd.notnull(zone_data['hvac_tu_type']):
                    # Storage discharge to TU for cooling.
                    self.control_output_matrix.at[
                        zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power',
                        zone_name + '_sensible_storage_to_zone_tu_cool_thermal_power'
                    ] = 1.0

                    # Storage discharge to TU for heating.
                    self.control_output_matrix.at[
                        zone_name + '_sensible_storage_to_zone_tu_heat_thermal_power',
                        zone_name + '_sensible_storage_to_zone_tu_heat_thermal_power'
                    ] = 1.0

            # Battery storage.
            if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
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

    def define_output_surfaces_exterior_irradiation_gain_exterior(self):
        for surface_name, surface_data in self.building_surfaces_exterior.iterrows():
            if self.parse_parameter(surface_data['heat_capacity']) != 0.0:  # Surfaces with non-zero heat capacity
                self.disturbance_output_matrix.at[
                    surface_name + '_irradiation_gain_exterior',
                    'irradiation_' + surface_data['direction_name']
                ] = (
                        self.disturbance_output_matrix.at[
                            surface_name + '_irradiation_gain_exterior',
                            'irradiation_' + surface_data['direction_name']
                        ]
                ) + (
                        self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                )
            else:  # Surfaces with neglected heat capacity
                self.disturbance_output_matrix.at[
                    surface_data['surface_name'] + '_irradiation_gain_exterior',
                    'irradiation_' + surface_data['direction_name']
                ] = (
                        self.disturbance_output_matrix.at[
                            surface_data['surface_name'] + '_irradiation_gain_exterior',
                            'irradiation_' + surface_data['direction_name']
                        ]
                ) + (
                        self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                )

    def define_output_surfaces_exterior_convection_interior(self):
        for surface_name, surface_data in self.building_surfaces_exterior.iterrows():
            # Total zone surface area for later calculating share of interior (indirect) irradiation
            zone_surface_area = sum(
                self.parse_parameter(zone_surface_data['surface_area'])
                * (1 - self.parse_parameter(zone_surface_data['window_wall_ratio']))
                for zone_surface_name, zone_surface_data in pd.concat(
                    [
                        self.building_surfaces_exterior[:][
                            self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                            ],
                        self.building_surfaces_interior[:][
                            self.building_surfaces_interior['zone_name'] == surface_data['zone_name']
                            ],
                        self.building_surfaces_interior[:][
                            self.building_surfaces_interior['zone_adjacent_name'] == surface_data['zone_name']
                            ],
                        self.building_surfaces_adiabatic[:][
                            self.building_surfaces_adiabatic['zone_name'] == surface_data['zone_name']
                            ]
                    ],
                    sort=False
                ).iterrows()  # For all surfaces adjacent to the zone
            )

            if self.parse_parameter(surface_data['heat_capacity']) != 0.0:  # Surfaces with non-zero heat capacity
                # Convective heat transfer from the surface towards zone
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_output_matrix.at[
                                surface_data['surface_name'] + '_convection_interior',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / zone_surface_area
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (1.0 - (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                            ) ** (- 1))
                    )
                self.state_output_matrix.at[
                    surface_data['surface_name'] + '_convection_interior',
                    surface_name + '_temperature'
                ] = (
                        self.state_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            surface_name + '_temperature'
                        ]
                ) + (
                        (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                )
                self.state_output_matrix.at[
                    surface_data['surface_name'] + '_convection_interior',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                )
                                + 1.0
                                / (
                                        2.0
                                        * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                                )
                        ) ** (- 1)
                )
            else:  # Surfaces with neglected heat capacity
                # Complete convective heat transfer from surface to zone
                self.disturbance_output_matrix.at[
                    surface_data['surface_name'] + '_convection_interior',
                    'irradiation_' + surface_data['direction_name']
                ] = (
                        self.disturbance_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            'irradiation_' + surface_data['direction_name']
                        ]
                ) + (
                        self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                )
                self.disturbance_output_matrix.at[
                    surface_data['surface_name'] + '_convection_interior',
                    'ambient_air_temperature'
                ] = (
                        self.disturbance_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            'ambient_air_temperature'
                        ]
                ) + (
                        (
                                self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                        )
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                )
                self.disturbance_output_matrix.at[
                    surface_data['surface_name'] + '_convection_interior',
                    'sky_temperature'
                ] = (
                        self.disturbance_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            'sky_temperature'
                        ]
                ) + (
                        self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                )
                self.state_output_matrix.at[
                    surface_data['surface_name'] + '_convection_interior',
                    surface_data['zone_name'] + '_temperature'
                ] = (
                        self.state_output_matrix.at[
                            surface_data['surface_name'] + '_convection_interior',
                            surface_data['zone_name'] + '_temperature'
                        ]
                ) + (
                        - 1.0
                        * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                        * (
                                1.0
                                / (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                + 1.0
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + 1.0
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        ) ** (- 1)
                )
                for zone_exterior_surface_name, zone_exterior_surface_data in self.building_surfaces_exterior[:][
                    self.building_surfaces_exterior['zone_name'] == surface_data['zone_name']
                ].iterrows():
                    # Interior irradiation through all exterior surfaces adjacent to the zone
                    self.disturbance_output_matrix.at[
                        surface_data['surface_name'] + '_convection_interior',
                        'irradiation_' + zone_exterior_surface_data['direction_name']
                    ] = (
                            self.disturbance_output_matrix.at[
                                surface_data['surface_name'] + '_convection_interior',
                                'irradiation_' + zone_exterior_surface_data['direction_name']
                            ]
                    ) + (
                            (
                                    self.parse_parameter(zone_exterior_surface_data['surface_area'])
                                    * self.parse_parameter(zone_exterior_surface_data['window_wall_ratio'])
                                    / zone_surface_area
                            )  # Considers the share at the respective surface
                            * self.parse_parameter(surface_data['irradiation_gain_coefficient'])
                            * (1 - self.parse_parameter(surface_data['window_wall_ratio']))
                            * (1.0 - (
                                1.0
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                                + (
                                        self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                                        + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                                )
                                / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                            ) ** (- 1))
                    )

    def load_disturbance_timeseries(self, conn):
        # Load weather timeseries
        weather_timeseries = pd.read_sql(
            """
            SELECT * FROM weather_timeseries 
            WHERE weather_type='{}'
            AND time BETWEEN '{}' AND '{}'
            """.format(
                self.building_scenarios['weather_type'][0],
                self.building_scenarios['time_start'][0],
                self.building_scenarios['time_end'][0]
            ),
            conn
        )
        weather_timeseries.index = pd.to_datetime(weather_timeseries['time'])

        # Load internal gain timeseries
        building_internal_gain_timeseries = pd.read_sql(
            """
            SELECT * FROM building_internal_gain_timeseries 
            WHERE internal_gain_type IN ({})
            AND time BETWEEN '{}' AND '{}'
            """.format(
                ", ".join([
                    "'{}'".format(data_set_name) for data_set_name in self.building_zones['internal_gain_type'].unique()
                ]),
                self.building_scenarios['time_start'][0],
                self.building_scenarios['time_end'][0]
            ),
            conn
        )

        # Pivot internal gain timeseries, so there is one `_occupancy` & one `_appliances` for each `internal_gain_type`
        building_internal_gain_occupancy_timeseries = building_internal_gain_timeseries.pivot(
            index='time',
            columns='internal_gain_type',
            values='internal_gain_occupancy'
        )
        building_internal_gain_occupancy_timeseries.columns = (
                building_internal_gain_occupancy_timeseries.columns + '_occupancy'
        )
        building_internal_gain_appliances_timeseries = building_internal_gain_timeseries.pivot(
            index='time',
            columns='internal_gain_type',
            values='internal_gain_appliances'
        )
        building_internal_gain_appliances_timeseries.columns = (
                building_internal_gain_appliances_timeseries.columns + '_appliances'
        )
        building_internal_gain_timeseries = pd.concat(
            [
                building_internal_gain_occupancy_timeseries,
                building_internal_gain_appliances_timeseries
            ],
            axis='columns'
        )
        building_internal_gain_timeseries.index = pd.to_datetime(building_internal_gain_timeseries.index)

        # Reindex, interpolate and construct full disturbance timeseries
        # TODO: Initialize disturbance_timeseries in _init_
        self.disturbance_timeseries = pd.concat(
            [
                weather_timeseries[[
                    'ambient_air_temperature',
                    'sky_temperature',
                    'irradiation_horizontal',
                    'irradiation_east',
                    'irradiation_south',
                    'irradiation_west',
                    'irradiation_north'
                    # 'storage_ambient_air_temperature'  # TODO: Cleanup `storage_ambient_air_temperature`.
                ]].reindex(
                    self.set_timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
                ).ffill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
                ),
                building_internal_gain_timeseries.reindex(
                    self.set_timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
                ).ffill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
                ),
                (
                    pd.DataFrame(
                        1.0,
                        self.set_timesteps,
                        ['constant']
                    ) if self.define_constant else pd.DataFrame([])  # Append constant only when needed.
                )
            ],
            axis='columns',
        ).rename_axis('disturbance_name', axis='columns')

    def load_electricity_price_timeseries(self, conn):
        if pd.isnull(self.building_scenarios['price_type'][0]):
            # If no price_type defined, generate a flat price signal.
            self.electricity_price_timeseries = (
                pd.DataFrame(
                    [[None, 1.0]],
                    columns=['price_type', 'price'],
                    index=self.set_timesteps
                )
            )
        else:
            self.electricity_price_timeseries = (
                pd.read_sql(
                    """
                    SELECT * FROM electricity_price_timeseries 
                    WHERE price_type=(
                        SELECT price_type from building_scenarios 
                        WHERE scenario_name='{}'
                    )
                    """.format(self.scenario_name),
                    conn,
                    index_col='time',
                    parse_dates=['time']
                )
            )

            # Reindex / interpolate to match set_timesteps.
            self.electricity_price_timeseries.reindex(
                self.set_timesteps
            ).interpolate(
                'quadratic'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            )

    def define_output_constraint_timeseries(self, conn):
        """
        - Generate minimum/maximum constraint timeseries based on `building_zone_constraint_profiles`.
        """
        # TODO: Make construction / interpolation simpler and more efficient.

        # Initialise constraint timeseries as +/- infinity.
        self.output_constraint_timeseries_maximum = pd.DataFrame(
            1.0 * np.infty,
            self.set_timesteps,
            self.set_outputs
        )
        self.output_constraint_timeseries_minimum = -self.output_constraint_timeseries_maximum
        # Minimum bound for "power" outputs.
        self.output_constraint_timeseries_minimum.loc[
            :,
            [column for column in self.output_constraint_timeseries_minimum.columns if '_power' in column]
        ] = 0.0

        # Minimum bound for "flow" outputs.
        self.output_constraint_timeseries_minimum.loc[
            :,
            [column for column in self.output_constraint_timeseries_minimum.columns if '_flow' in column]
        ] = 0.0

        # Minimum bound for storage charge and discharge outputs.
        self.output_constraint_timeseries_minimum.loc[
            :,
            [column for column in self.output_constraint_timeseries_minimum.columns if '_storage_charge' in column]
        ] = 0.0
        self.output_constraint_timeseries_minimum.loc[
            :,
            [column for column in self.output_constraint_timeseries_minimum.columns if '_storage_to_zone' in column]
        ] = 0.0

        # If a heating/cooling session is defined, the cooling/heating air flow is forced to 0.
        # Comment: The cooling or heating coil may still be working, because of the dehumidification,
        # however it would not appear explicitly in the output.
        if self.building_scenarios['heating_cooling_session'][0] == 'heating':
            self.output_constraint_timeseries_maximum.loc[
                :, [column for column in self.output_constraint_timeseries_minimum.columns if '_cool' in column]
            ] = 0
        if self.building_scenarios['heating_cooling_session'][0] == 'cooling':
            self.output_constraint_timeseries_maximum.loc[
                :, [column for column in self.output_constraint_timeseries_minimum.columns if '_heat' in column]
            ] = 0

        for zone_name, zone_data in self.building_zones.iterrows():
            # Load zone_constraint_profile for each zone.
            building_zone_constraint_profile = pd.read_sql(
                """
                SELECT * FROM building_zone_constraint_profiles 
                WHERE zone_constraint_profile='{}'
                """.format(zone_data['zone_constraint_profile']),
                conn
            )

            # Create index function for `from_weekday` (mapping from `timestep.weekday()` to `from_weekday`).
            constraint_profile_index_day = scipy.interpolate.interp1d(
                building_zone_constraint_profile['from_weekday'],
                building_zone_constraint_profile['from_weekday'],
                kind='zero',
                fill_value='extrapolate'
            )
            for timestep in self.set_timesteps:
                # Create index function for `from_time` (mapping `timestep.timestamp` to `from_time`).
                constraint_profile_index_time = scipy.interpolate.interp1d(
                    pd.to_datetime(
                        str(timestep.date())
                        + ' '
                        + building_zone_constraint_profile['from_time'][
                            building_zone_constraint_profile['from_weekday']
                            == constraint_profile_index_day(timestep.weekday())
                        ]
                    ).view('int64'),
                    building_zone_constraint_profile.index[
                        building_zone_constraint_profile['from_weekday']
                        == constraint_profile_index_day(timestep.weekday())
                        ].values,
                    kind='zero',
                    fill_value='extrapolate'
                )

                # Select constraint values.
                self.output_constraint_timeseries_minimum.at[
                    timestep,
                    zone_name + '_temperature'
                ] = self.parse_parameter(
                    building_zone_constraint_profile['minimum_air_temperature'][
                        int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                    ]
                )
                self.output_constraint_timeseries_maximum.at[
                    timestep,
                    zone_name + '_temperature'
                ] = self.parse_parameter(
                    building_zone_constraint_profile['maximum_air_temperature'][
                        int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                    ]
                )

                if pd.notnull(zone_data['hvac_ahu_type']) | pd.notnull(zone_data['window_type']):
                    if pd.notnull(self.building_scenarios['demand_controlled_ventilation_type'][0]):
                        if pd.notnull(self.building_scenarios['co2_model_type'][0]):
                            self.output_constraint_timeseries_maximum.at[
                                timestep,
                                zone_name + '_co2_concentration'
                            ] = (
                                self.parse_parameter(
                                    building_zone_constraint_profile['maximum_co2_concentration'][
                                        int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                    ]
                                )
                            )
                            self.output_constraint_timeseries_minimum.at[
                                timestep,
                                zone_name + '_total_fresh_air_flow'
                            ] = (
                                self.parse_parameter(
                                    building_zone_constraint_profile['minimum_fresh_air_flow_per_area'][
                                        int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                    ]
                                )
                                * self.parse_parameter(zone_data['zone_area'])
                            )
                        else:
                            self.output_constraint_timeseries_minimum.at[
                                timestep, zone_name + '_total_fresh_air_flow'
                            ] = (
                                    self.parse_parameter(
                                        building_zone_constraint_profile['minimum_fresh_air_flow_per_person'][
                                            int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                        ]
                                    )
                                    * (
                                        self.disturbance_timeseries[
                                            self.building_zones["internal_gain_type"].loc[zone_name] + "_occupancy"
                                        ].loc[timestep] * self.parse_parameter(zone_data['zone_area'])
                                        + self.parse_parameter(
                                            building_zone_constraint_profile['minimum_fresh_air_flow_per_area'][
                                                int(constraint_profile_index_time(
                                                    timestep.to_datetime64().astype('int64')
                                                ))
                                            ]
                                        )
                                        * self.parse_parameter(zone_data['zone_area'])
                                    )
                            )
                    else:
                        if pd.notnull(zone_data['hvac_ahu_type']):
                            self.output_constraint_timeseries_minimum.at[
                                timestep,
                                zone_name + '_total_fresh_air_flow'
                            ] = (
                                self.parse_parameter(
                                    building_zone_constraint_profile['minimum_fresh_air_flow_per_area_no_dcv'][
                                        int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                    ]
                                )
                                * self.parse_parameter(zone_data['zone_area'])
                            )
                        elif pd.notnull(zone_data['window_type']):
                            self.output_constraint_timeseries_minimum.at[
                                timestep,
                                zone_name + '_window_fresh_air_flow'
                            ] = (
                                self.parse_parameter(
                                    building_zone_constraint_profile['minimum_fresh_air_flow_per_area_no_dcv'][
                                        int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                    ]
                                )
                                * self.parse_parameter(zone_data['zone_area'])
                            )
                # If a ventilation system is enabled, if DCV, then CO2 or constraint on total fresh air flow.
                # If no DCV, then constant constraint on AHU or on windows if no AHU

                if pd.notnull(self.building_scenarios['humidity_model_type'][0]):
                    if pd.notnull(zone_data['hvac_ahu_type']):
                        self.output_constraint_timeseries_minimum.at[
                            timestep,
                            zone_name + '_absolute_humidity'
                        ] = humid_air_properties(
                            'W',
                            'R',
                            self.parse_parameter(
                                building_zone_constraint_profile['minimum_relative_humidity'][
                                    int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                ]
                            )
                            / 100,
                            'T',
                            self.parse_parameter(
                                self.building_scenarios['linearization_zone_air_temperature_cool']
                            )
                            + 273.15,
                            'P',
                            101325
                        )
                        self.output_constraint_timeseries_maximum.at[
                            timestep,
                            zone_name + '_absolute_humidity'
                        ] = humid_air_properties(
                            'W',
                            'R',
                            self.parse_parameter(
                                building_zone_constraint_profile['maximum_relative_humidity'][
                                    int(constraint_profile_index_time(timestep.to_datetime64().astype('int64')))
                                ]
                            )
                            / 100,
                            'T',
                            self.parse_parameter(
                                self.building_scenarios['linearization_zone_air_temperature_cool']
                            )
                            + 273.15,
                            'P',
                            101325
                        )

                # Storage state of charge constraints.
                # Sensible thermal storage.
                # TODO: Validate storage size units.
                if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                    self.output_constraint_timeseries_maximum.at[
                        timestep,
                        self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge'
                    ] = (
                        self.parse_parameter(self.building_scenarios['storage_size'])
                        * self.parse_parameter('water_density')  # Convert volume to mass.
                    )
                if self.building_scenarios['building_storage_type'][0] == 'default_sensible_thermal_storage':
                    self.output_constraint_timeseries_minimum.at[
                        timestep,
                        self.building_scenarios['building_name'][0] + '_sensible_thermal_storage_state_of_charge'
                    ] = 0.0

                # Battery storage.
                if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
                    self.output_constraint_timeseries_maximum.at[
                        timestep,
                        self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge'
                    ] = self.parse_parameter(self.building_scenarios['storage_size'])
                if self.building_scenarios['building_storage_type'][0] == 'default_battery_storage':
                    self.output_constraint_timeseries_minimum.at[
                        timestep,
                        self.building_scenarios['building_name'][0] + '_battery_storage_state_of_charge'
                    ] = (
                        0.0
                        # TODO: Revise implementation of depth of discharge.
                        # + self.parse_parameter(self.building_scenarios['storage_size'])
                        # * (1.0 - self.parse_parameter(self.building_scenarios['storage_battery_depth_of_discharge']))
                    )

    def discretize_model(self):
        """
        - Discretization assuming zero order hold
        - Source: https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        """

        state_matrix_discrete = scipy.linalg.expm(
            self.state_matrix.values
            * pd.to_timedelta(self.building_scenarios['time_step'][0]).seconds
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

    def define_augmented_model(self):
        """Define augmented state space model matrices.

        - Based on Oldewurtel 2011: https://doi.org/10.3929/ethz-a-007157625%0A
        - Construct augmented state space matrices based on 'normal' state space matrices.
        - Augmented state space matrices are stored in the build model (not returned).
        - TODO: Validate matrices.
        - TODO: Check alternatives for np.block(), which is slow for a large number of time steps.
        """
        # Create local numpy.array copies of state space matrices for speed
        state_matrix = self.state_matrix.values
        control_matrix = self.control_matrix.values
        disturbance_matrix = self.disturbance_matrix.values
        state_output_matrix = self.state_output_matrix.values
        control_output_matrix = self.control_output_matrix.values
        disturbance_output_matrix = self.disturbance_output_matrix.values

        # Construct first column, i.e., rows, of augmented matrices.
        state_matrix_augmented_rows = [np.eye(state_matrix.shape[0])]
        control_matrix_augmented_rows = [np.zeros_like(control_matrix)]
        disturbance_matrix_augmented_rows = [np.zeros_like(disturbance_matrix)]
        state_output_matrix_augmented_rows = [state_output_matrix]
        control_output_matrix_augmented_rows = [control_output_matrix]
        disturbance_output_matrix_augmented_rows = [disturbance_output_matrix]
        for timestep in range(1, len(self.set_timesteps)):
            state_matrix_augmented_rows.append(
                state_matrix ** timestep
            )
            control_matrix_augmented_rows.append(
                (state_matrix ** (timestep - 1)).dot(control_matrix)
            )
            disturbance_matrix_augmented_rows.append(
                (state_matrix ** (timestep - 1)).dot(disturbance_matrix)
            )
            state_output_matrix_augmented_rows.append(
                state_output_matrix.dot(state_matrix_augmented_rows[-1])
            )
            control_output_matrix_augmented_rows.append(
                state_output_matrix.dot(control_matrix_augmented_rows[-1])
            )
            disturbance_output_matrix_augmented_rows.append(
                state_output_matrix.dot(disturbance_matrix_augmented_rows[-1])
            )

        # Construct remaining columns of augmented matrices (except state and state_output).
        control_matrix_augmented_cols = [control_matrix_augmented_rows]
        disturbance_matrix_augmented_cols = [disturbance_matrix_augmented_rows]
        control_output_matrix_augmented_cols = [control_output_matrix_augmented_rows]
        disturbance_output_matrix_augmented_cols = [disturbance_output_matrix_augmented_rows]
        for timestep in range(1, len(self.set_timesteps)):
            control_matrix_augmented_cols.append(
                [np.zeros(control_matrix.shape)] * timestep
                + control_matrix_augmented_rows[:-timestep]
            )
            disturbance_matrix_augmented_cols.append(
                [np.zeros(disturbance_matrix.shape)] * timestep
                + disturbance_matrix_augmented_rows[:-timestep]
            )
            control_output_matrix_augmented_cols.append(
                [np.zeros(control_output_matrix.shape)] * timestep
                + control_output_matrix_augmented_rows[:-timestep]
            )
            disturbance_output_matrix_augmented_cols.append(
                [np.zeros(disturbance_output_matrix.shape)] * timestep
                + disturbance_output_matrix_augmented_rows[:-timestep]
            )

        # Construct full augmented matrices.
        self.state_matrix_augmented = np.zeros(
            (state_matrix.shape[0] * len(self.set_timesteps), state_matrix.shape[0])
        )
        self.control_matrix_augmented = np.zeros(
            (control_matrix.shape[0] * len(self.set_timesteps),) * 2
        )
        self.disturbance_matrix_augmented = np.zeros(
            (disturbance_matrix.shape[0] * len(self.set_timesteps),) * 2
        )
        self.state_output_matrix_augmented = np.zeros(
            (state_output_matrix.shape[0] * len(self.set_timesteps), state_output_matrix.shape[0])
        )
        self.control_output_matrix_augmented = np.zeros(
            (control_output_matrix.shape[0] * len(self.set_timesteps),) * 2
        )
        self.disturbance_output_matrix_augmented = np.zeros(
            (disturbance_output_matrix.shape[0] * len(self.set_timesteps),) * 2
        )
        self.state_matrix_augmented = np.vstack(state_matrix_augmented_rows)
        self.control_matrix_augmented = np.block(control_matrix_augmented_cols).transpose()
        self.disturbance_matrix_augmented = np.block(disturbance_matrix_augmented_cols).transpose()
        self.state_output_matrix_augmented = np.vstack(state_output_matrix_augmented_rows)
        self.control_output_matrix_augmented = np.block(control_output_matrix_augmented_cols).transpose()
        self.disturbance_output_matrix_augmented = np.block(disturbance_output_matrix_augmented_cols).transpose()

    def simulate(
            self,
            state_initial,
            control_timeseries,
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
        state_timeseries = pd.DataFrame(
            np.nan,
            self.set_timesteps,
            self.set_states
        )
        state_timeseries.iloc[0, :] = state_initial
        output_timeseries = pd.DataFrame(
            np.nan,
            self.set_timesteps,
            self.set_outputs
        )

        # Iterative simulation of state space equations
        for timestep in range(len(self.set_timesteps) - 1):
            state_timeseries.iloc[timestep + 1, :] = (
                    np.dot(self.state_matrix.values, state_timeseries.iloc[timestep, :].values)
                    + np.dot(self.control_matrix.values, control_timeseries.iloc[timestep, :].values)
                    + np.dot(self.disturbance_matrix.values, disturbance_timeseries.iloc[timestep, :].values)
            )
        for timestep in range(1, len(self.set_timesteps)):
            # TODO: Check `timestep - 1` (This was added to match with EnergyPlus outputs)
            output_timeseries.iloc[timestep - 1, :] = (
                    np.dot(self.state_output_matrix.values, state_timeseries.iloc[timestep, :].values)
                    + np.dot(self.control_output_matrix.values, control_timeseries.iloc[timestep, :].values)
                    + np.dot(self.disturbance_output_matrix.values, disturbance_timeseries.iloc[timestep, :].values)
            )

        return (
            state_timeseries,
            output_timeseries
        )

