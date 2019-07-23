"""
Building model class definition
"""

import warnings
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.interpolate
# Using CoolProp for calculating humid air properties: http://www.coolprop.org/fluid_properties/HumidAir.html
from CoolProp.HumidAirProp import HAPropsSI as humid_air_properties


class Building(object):
    """
    Building scenario object to store all information
    """

    def __init__(self, conn, scenario_name):
        # Obtain scenario name.
        self.scenario_name = scenario_name

        # Load building information from database
        self.building_scenarios = pd.read_sql(
            """
            select * from building_scenarios 
            join buildings using (building_name) 
            join building_linearization_types using (linearization_type) 
            where scenario_name='{}'
            """.format(self.scenario_name),
            conn
        )
        self.building_parameters = pd.read_sql(
            """
            select * from building_parameter_sets 
            where parameter_set in ('constants', '{}')
            """.format(self.building_scenarios['parameter_set'][0]),
            conn
        )
        self.building_parameters = pd.Series(
            self.building_parameters['parameter_value'].values,
            self.building_parameters['parameter_name'].values
        )  # Convert to series for shorter indexing
        self.building_surfaces_adiabatic = pd.read_sql(
            """
            select * from building_surfaces_adiabatic 
            join building_surface_types using (surface_type) 
            left join building_window_types using (window_type) 
            join building_zones using (zone_name, building_name) 
            where building_name='{}'
            """.format(self.building_scenarios['building_name'][0]),
            conn
        )
        self.building_surfaces_adiabatic.index = self.building_surfaces_adiabatic['surface_name']
        self.building_surfaces_exterior = pd.read_sql(
            """
            select * from building_surfaces_exterior 
            join building_surface_types using (surface_type) 
            left join building_window_types using (window_type) 
            join building_zones using (zone_name, building_name) 
            where building_name='{}'
            """.format(self.building_scenarios['building_name'][0]),
            conn
        )
        self.building_surfaces_exterior.index = self.building_surfaces_exterior['surface_name']
        self.building_surfaces_interior = pd.read_sql(
            """
            select * from building_surfaces_interior 
            join building_surface_types using (surface_type) 
            left join building_window_types using (window_type) 
            join building_zones using (zone_name, building_name) 
            where building_name='{}'
            """.format(self.building_scenarios['building_name'][0]),
            conn
        )
        self.building_surfaces_interior.index = self.building_surfaces_interior['surface_name']
        self.building_zones = pd.read_sql(
            """
            select * from building_zones 
            join building_zone_types using (zone_type) 
            join building_internal_gain_types using (internal_gain_type) 
            left join building_blind_types using (blind_type) 
            left join building_hvac_generic_types using (hvac_generic_type) 
            left join building_hvac_ahu_types using (hvac_ahu_type) 
            left join building_hvac_tu_types using (hvac_tu_type) 
            where building_name='{}'
            """.format(self.building_scenarios['building_name'][0]),
            conn
        )
        self.building_zones.index = self.building_zones['zone_name']

        # Add constant timeseries in disturbance vector, if any CO2 model or HVAC or window
        self.define_constant = (
                (self.building_scenarios['co2_model_type'][0] != '')
                | (self.building_zones['hvac_ahu_type'] != '').any()
                | (self.building_zones['window_type'] != '').any()
        )

        # Define sets
        self.set_states = pd.Index(
            pd.concat([
                self.building_zones['zone_name'] + '_temperature',
                self.building_surfaces_adiabatic['surface_name'][
                    self.building_surfaces_adiabatic['heat_capacity'] != '0'
                    ] + '_temperature',
                self.building_surfaces_exterior['surface_name'][
                    self.building_surfaces_exterior['heat_capacity'] != '0'
                    ] + '_temperature',
                self.building_surfaces_interior['surface_name'][
                    self.building_surfaces_interior['heat_capacity'] != '0'
                    ] + '_temperature',
                self.building_zones['zone_name'][
                    ((self.building_zones['hvac_ahu_type'] != '') | (self.building_zones['window_type'] != ''))
                    & (self.building_scenarios['co2_model_type'][0] != '')
                    ] + '_co2_concentration',
                self.building_zones['zone_name'][
                    (self.building_zones['hvac_ahu_type'] != '')
                    & (self.building_scenarios['humidity_model_type'][0] != '')
                    ] + '_absolute_humidity'
            ]),
            name='state_name'
        )
        self.set_controls = pd.Index(
            pd.concat([
                self.building_zones['zone_name'][
                    self.building_zones['hvac_generic_type'] != ''
                    ] + '_generic_heat_thermal_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_generic_type'] != ''
                    ] + '_generic_cool_thermal_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_ahu_type'] != ''
                    ] + '_ahu_heat_air_flow',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_ahu_type'] != ''
                    ] + '_ahu_cool_air_flow',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_tu_type'] != ''
                    ] + '_tu_heat_air_flow',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_tu_type'] != ''
                    ] + '_tu_cool_air_flow',
                self.building_zones['zone_name'][
                    (self.building_zones['window_type'] != '')
                ] + '_window_air_flow'
            ]),
            name='control_name'
        )
        self.set_disturbances = pd.Index(
            pd.concat([
                pd.Series([
                    'ambient_air_temperature',
                    'sky_temperature',
                    'irradiation_horizontal',
                    'irradiation_east',
                    'irradiation_south',
                    'irradiation_west',
                    'irradiation_north'
                ]),
                pd.Series(self.building_zones['internal_gain_type'].unique() + '_occupancy'),
                pd.Series(self.building_zones['internal_gain_type'].unique() + '_appliances'),
                (pd.Series(['constant']) if self.define_constant else pd.Series([]))
            ]),
            name='disturbance_name'
        )
        self.set_outputs = pd.Index(
            pd.concat([
                self.building_zones['zone_name'] + '_temperature',
                self.building_zones['zone_name'][
                    ((self.building_zones['hvac_ahu_type'] != '') | (self.building_zones['window_type'] != ''))
                    & (self.building_scenarios['co2_model_type'][0] != '')
                    ] + '_co2_concentration',
                self.building_zones['zone_name'][
                    (self.building_zones['hvac_ahu_type'] != '')
                    & (self.building_scenarios['humidity_model_type'][0] != '')
                    ] + '_absolute_humidity',
                self.building_zones['zone_name'][
                    (self.building_zones['hvac_ahu_type'] != '')
                    | (self.building_zones['window_type'] != '')
                    ] + '_total_fresh_air_flow',
                self.building_zones['zone_name'][
                    (self.building_zones['hvac_ahu_type'] != '')
                ] + '_ahu_fresh_air_flow',
                self.building_zones['zone_name'][
                    (self.building_zones['window_type'] != '')
                ] + '_window_fresh_air_flow',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_generic_type'] != ''
                    ] + '_generic_heat_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_generic_type'] != ''
                    ] + '_generic_cool_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_ahu_type'] != ''
                    ] + '_ahu_heat_electric_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_ahu_type'] != ''
                    ] + '_ahu_cool_electric_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_tu_type'] != ''
                    ] + '_tu_heat_electric_power',
                self.building_zones['zone_name'][
                    self.building_zones['hvac_tu_type'] != ''
                    ] + '_tu_cool_electric_power',
                # Validation outputs.
                self.building_surfaces_exterior['surface_name'] + '_irradiation_gain_exterior',
                self.building_surfaces_exterior['surface_name'] + '_convection_interior'
            ]),
            name='output_name'
        )
        self.set_timesteps = pd.Index(
            pd.date_range(
                start=pd.to_datetime(self.building_scenarios['time_start'][0]),
                end=pd.to_datetime(self.building_scenarios['time_end'][0]),
                freq=pd.to_timedelta(self.building_scenarios['time_step'][0])
            ),
            name='time'
        )

        # Define model matrices
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

        # Define heat capacity vector
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

        # Definition of parameters / coefficients
        self.define_heat_transfer_coefficients()

        # Define heat fluxes and co2 transfers
        self.define_heat_transfer_surfaces_exterior()
        self.define_heat_transfer_surfaces_interior()
        self.define_heat_transfer_surfaces_adiabatic()
        self.define_heat_transfer_infiltration()
        self.define_heat_transfer_internal_gains()
        self.define_heat_transfer_hvac_generic()
        self.define_heat_transfer_hvac_ahu()
        self.define_heat_transfer_hvac_tu()
        self.define_co2_transfer_hvac_ahu()
        self.define_heat_transfer_window_air_flow()
        self.define_humidity_transfer_hvac_ahu()

        # Define outputs
        self.define_output_zone_temperature()
        self.define_output_zone_co2_concentration()
        self.define_output_zone_humidity()
        self.define_output_hvac_generic_power()
        self.define_output_hvac_ahu_electric_power()
        self.define_output_hvac_tu_electric_power()
        self.define_output_fresh_air_flow()
        self.define_output_window_fresh_air_flow()
        self.define_output_ahu_fresh_air_flow()

        # Define validation outputs.
        self.define_output_surfaces_exterior_irradiation_gain_exterior()
        self.define_output_surfaces_exterior_convection_interior()

        # Define timeseries
        self.load_disturbance_timeseries(conn)
        # self.define_output_constraint_timeseries(conn)

        # Convert to time discrete model
        self.discretize_model()

    def parse_parameter(self, parameter):
        """
        - Checks if given parameter is a number.
        - If the parameter is not a number, it is assumed to be the name of a parameter from `building_parameters`
        """
        try:
            return float(parameter)
        except ValueError:
            return self.building_parameters[parameter]

    def define_heat_transfer_coefficients(self):
        # Create empty rows in dataframe
        self.building_surfaces_exterior['heat_transfer_coefficient_surface_sky'] = ''
        self.building_surfaces_exterior['heat_transfer_coefficient_surface_ground'] = ''
        self.building_surfaces_exterior['heat_transfer_coefficient_window_sky'] = ''
        self.building_surfaces_exterior['heat_transfer_coefficient_window_ground'] = ''

        # Calculate heat transfer coefficients
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

    def define_heat_transfer_surfaces_exterior(self):
        """Thermal model: Exterior surfaces"""
        # TODO: Rename irradiation_gain_coefficient
        # TODO: Rename thermal_resistance_surface
        # TODO: Exterior window transmission factor
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
                                    / zone_surface_area
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
                                    / zone_surface_area
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
                                    / zone_surface_area
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
                                    / zone_surface_area
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
                # Total zone surface area for later calculating share of interior (indirect) irradiation
                zone_surface_area = sum(
                    self.parse_parameter(zone_surface_data['surface_area'])
                    * (1 - self.parse_parameter(zone_surface_data['window_wall_ratio']))
                    for zone_surface_name, zone_surface_data in pd.concat(
                        [
                            self.building_surfaces_exterior[:][
                                self.building_surfaces_exterior['zone_name'] == zone_name
                                ],
                            self.building_surfaces_interior[:][
                                self.building_surfaces_interior['zone_name'] == zone_name
                                ],
                            self.building_surfaces_interior[:][
                                self.building_surfaces_interior['zone_adjacent_name'] == zone_name
                                ],
                            self.building_surfaces_adiabatic[:][
                                self.building_surfaces_adiabatic['zone_name'] == zone_name
                                ]
                        ],
                        sort=False
                    ).iterrows()  # For all surfaces adjacent to the zone
                )

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
                                        / zone_surface_area
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
                                        / zone_surface_area
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
                                        / zone_surface_area
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
                                        / zone_surface_area
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
                                    / zone_surface_area
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
                                    / zone_surface_area
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
            if row['window_type'] != '':
                self.control_matrix.at[
                    index + '_temperature',
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
            if row['hvac_generic_type'] != '':
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

    def define_heat_transfer_hvac_ahu(self):
        for index, row in self.building_zones.iterrows():
            if row['hvac_ahu_type'] != '':
                self.control_matrix.at[
                    index + '_temperature',
                    index + '_ahu_heat_air_flow'
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_ahu_heat_air_flow'
                        ]
                        + self.parse_parameter('heat_capacity_air')
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
                ] = (
                        self.control_matrix.at[
                            index + '_temperature',
                            index + '_ahu_cool_air_flow'
                        ]
                        + self.parse_parameter('heat_capacity_air')
                        * (
                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(
                                        self.building_scenarios['linearization_zone_air_temperature_cool']
                                )
                        )
                        / self.heat_capacity_vector[index]
                )

    def define_heat_transfer_hvac_tu(self):
        for index, row in self.building_zones.iterrows():
            if row['hvac_tu_type'] != '':
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
                                )
                        )
                        / self.heat_capacity_vector[index]
                )

    def define_co2_transfer_hvac_ahu(self):
        if self.building_scenarios['co2_model_type'][0] != '':
            for index, row in self.building_zones.iterrows():
                if (row['hvac_ahu_type'] != '') | (row['window_type'] != ''):
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
                    if row['hvac_ahu_type'] != '':
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
                    if row['window_type'] != '':
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
        if self.building_scenarios['humidity_model_type'][0] != '':
            for index, row in self.building_zones.iterrows():
                if row['hvac_ahu_type'] != '':
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
                    if row['window_type'] != '':
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

    def define_output_zone_temperature(self):
        for index, row in self.building_zones.iterrows():
            self.state_output_matrix.at[
                index + '_temperature',
                index + '_temperature'
            ] = 1

    def define_output_zone_co2_concentration(self):
        if self.building_scenarios['co2_model_type'][0] != '':
            for index, row in self.building_zones.iterrows():
                if (row['hvac_ahu_type'] != '') | (row['window_type'] != ''):
                    self.state_output_matrix.at[
                        index + '_co2_concentration',
                        index + '_co2_concentration'
                    ] = 1

    def define_output_zone_humidity(self):
        if self.building_scenarios['humidity_model_type'][0] != '':
            for index, row in self.building_zones.iterrows():
                if row['hvac_ahu_type'] != '':
                    self.state_output_matrix.at[
                        index + '_absolute_humidity',
                        index + '_absolute_humidity'
                    ] = 1

    def define_output_hvac_generic_power(self):
        for index, row in self.building_zones.iterrows():
            if row['hvac_generic_type'] != '':
                self.control_output_matrix.at[
                    index + '_generic_heat_power',
                    index + '_generic_heat_thermal_power'
                ] = (
                        self.control_output_matrix.at[
                            index + '_generic_heat_power',
                            index + '_generic_heat_thermal_power'
                        ]
                        + 1
                        / self.parse_parameter(row['generic_heating_efficiency'])
                )
                self.control_output_matrix.at[
                    index + '_generic_cool_power',
                    index + '_generic_cool_thermal_power'
                ] = (
                        self.control_output_matrix.at[
                            index + '_generic_cool_power',
                            index + '_generic_cool_thermal_power'
                        ]
                        + 1
                        / self.parse_parameter(row['generic_cooling_efficiency'])
                )

    def define_output_hvac_ahu_electric_power(self):
        for index, row in self.building_zones.iterrows():
            if row['hvac_ahu_type'] != '':
                # Calculate enthalpies
                # TODO: Remove unnecessary HVAC types
                if (
                        row['ahu_cooling_type'] == 'default'
                        and row['ahu_heating_type'] == 'default'
                        and row['ahu_dehumidification_type'] == 'default'
                        and row['ahu_return_air_heat_recovery_type'] == 'default'
                ):
                    if (
                            self.parse_parameter(self.building_scenarios['linearization_ambient_air_humidity_ratio'])
                            <= humid_air_properties(
                                'W',
                                'R',
                                self.parse_parameter(row['ahu_supply_air_relative_humidity_setpoint'])
                                / 100,
                                'T',
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature'])
                                + 273.15,
                                'P',
                                101325
                            )
                    ):
                        delta_enthalpy_ahu_cooling = min(
                            0,
                            humid_air_properties(
                                'H',
                                'T',
                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                + 273.15,
                                'W',
                                self.parse_parameter(
                                    self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                ),
                                'P',
                                101325
                            )
                            - humid_air_properties(
                                'H',
                                'T',
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature'])
                                + 273.15,
                                'W',
                                self.parse_parameter(
                                    self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                ),
                                'P',
                                101325
                            )
                        )
                        delta_enthalpy_ahu_heating = max(
                            0,
                            humid_air_properties(
                                'H',
                                'T',
                                self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                + 273.15,
                                'W',
                                self.parse_parameter(
                                    self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                ),
                                'P',
                                101325
                            )
                            - humid_air_properties(
                                'H',
                                'T',
                                self.parse_parameter(self.building_scenarios['linearization_ambient_air_temperature'])
                                + 273.15,
                                'W',
                                self.parse_parameter(
                                    self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                ),
                                'P',
                                101325
                            )
                        )
                        delta_enthalpy_cooling_recovery = min(
                            0,
                            self.parse_parameter(row['ahu_return_air_heat_recovery_efficiency'])
                            * (
                                    humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_temperature_heat']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                                    - humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_temperature']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                            )
                        )
                        delta_enthalpy_heating_recovery = max(
                            0,
                            self.parse_parameter(row['ahu_return_air_heat_recovery_efficiency'])
                            * (
                                    humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_temperature_heat']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                                    - humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_temperature']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                            )
                        )
                    else:
                        delta_enthalpy_ahu_cooling = (
                                humid_air_properties(
                                    'H',
                                    'R',
                                    1,
                                    'W',
                                    humid_air_properties(
                                        'W',
                                        'T',
                                        self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                        + 273.15,
                                        'R',
                                        self.parse_parameter(
                                            row['ahu_supply_air_relative_humidity_setpoint']
                                        )
                                        / 100,
                                        'P',
                                        101325
                                    ),
                                    'P',
                                    101325
                                )
                                - humid_air_properties(
                                    'H',
                                    'T',
                                    self.parse_parameter(
                                        self.building_scenarios['linearization_ambient_air_temperature']
                                    )
                                    + 273.15,
                                    'W',
                                    self.parse_parameter(
                                        self.building_scenarios['linearization_ambient_air_humidity_ratio']
                                    ),
                                    'P',
                                    101325
                                )
                        )
                        delta_enthalpy_ahu_heating = (
                                humid_air_properties(
                                    'H',
                                    'T',
                                    self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                    + 273.15,
                                    'R',
                                    self.parse_parameter(row['ahu_supply_air_relative_humidity_setpoint'])
                                    / 100,
                                    'P',
                                    101325
                                )
                                - humid_air_properties(
                                    'H',
                                    'R',
                                    1,
                                    'W',
                                    humid_air_properties(
                                        'W',
                                        'T',
                                        self.parse_parameter(row['ahu_supply_air_temperature_setpoint'])
                                        + 273.15,
                                        'R',
                                        self.parse_parameter(row['ahu_supply_air_relative_humidity_setpoint'])
                                        / 100,
                                        'P',
                                        101325
                                    ),
                                    'P',
                                    101325
                                )
                        )
                        delta_enthalpy_cooling_recovery = min(
                            0,
                            self.parse_parameter(row['ahu_return_air_heat_recovery_efficiency'])
                            * (
                                    humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_temperature_heat']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                                    - humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_temperature']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                            )
                        )
                        delta_enthalpy_heating_recovery = max(
                            0,
                            self.parse_parameter(row['ahu_return_air_heat_recovery_efficiency'])
                            * (
                                    humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_temperature_heat']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                                    - humid_air_properties(
                                        'H',
                                        'T',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_ambient_air_temperature']
                                        )
                                        + 273.15,
                                        'W',
                                        self.parse_parameter(
                                            self.building_scenarios['linearization_zone_air_humidity_ratio']
                                        ),
                                        'P',
                                        101325
                                    )
                            )
                        )

                # Matrix entries
                self.control_output_matrix.at[
                    index + '_ahu_heat_electric_power',
                    index + '_ahu_heat_air_flow'
                ] = (
                        self.control_output_matrix.at[
                            index + '_ahu_heat_electric_power',
                            index + '_ahu_heat_air_flow'
                        ]
                        + self.parse_parameter('density_air')
                        * (
                                (
                                        abs(delta_enthalpy_ahu_cooling)
                                        - abs(delta_enthalpy_cooling_recovery)
                                )
                                / self.parse_parameter(row['ahu_cooling_efficiency'])
                                + (
                                            abs(delta_enthalpy_ahu_heating)
                                            - abs(delta_enthalpy_heating_recovery)
                                )
                                / self.parse_parameter(row['ahu_heating_efficiency'])
                                + self.parse_parameter(row['ahu_fan_efficiency'])
                        )
                )
                self.control_output_matrix.at[
                    index + '_ahu_cool_electric_power',
                    index + '_ahu_cool_air_flow'
                ] = (
                        self.control_output_matrix.at[
                            index + '_ahu_cool_electric_power',
                            index + '_ahu_cool_air_flow'
                        ]
                        + self.parse_parameter('density_air')
                        * (
                                (
                                        abs(delta_enthalpy_ahu_cooling)
                                        - abs(delta_enthalpy_cooling_recovery)
                                )
                                / self.parse_parameter(row['ahu_cooling_efficiency'])
                                + (
                                        abs(delta_enthalpy_ahu_heating)
                                        - abs(delta_enthalpy_heating_recovery)
                                )
                                / self.parse_parameter(row['ahu_heating_efficiency'])
                                + self.parse_parameter(row['ahu_fan_efficiency'])
                        )
                )

    def define_output_hvac_tu_electric_power(self):
        for index, row in self.building_zones.iterrows():
            if row['hvac_tu_type'] != '':
                # Calculate enthalpies
                if (
                        row['tu_cooling_type'] == 'default'
                        and row['tu_heating_type'] == 'default'
                ):
                    if row['tu_air_intake_type'] == 'zone':
                        delta_enthalpy_tu_cooling = self.parse_parameter('heat_capacity_air') * (
                                self.parse_parameter(self.building_scenarios['linearization_zone_air_temperature_cool'])
                                - self.parse_parameter(row['tu_supply_air_temperature_setpoint'])
                        )
                        delta_enthalpy_tu_heating = self.parse_parameter('heat_capacity_air') * (
                                self.parse_parameter(self.building_scenarios['linearization_zone_air_temperature_heat'])
                                - self.parse_parameter(row['tu_supply_air_temperature_setpoint'])
                        )
                    elif row['tu_air_intake_type'] == 'ahu':
                        delta_enthalpy_tu_cooling = self.parse_parameter('heat_capacity_air') * (
                                self.parse_parameter(self.building_scenarios['ahu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(row['tu_supply_air_temperature_setpoint'])
                        )
                        delta_enthalpy_tu_heating = self.parse_parameter('heat_capacity_air') * (
                                self.parse_parameter(self.building_scenarios['ahu_supply_air_temperature_setpoint'])
                                - self.parse_parameter(row['tu_supply_air_temperature_setpoint'])
                        )

                # Matrix entries
                self.control_output_matrix.at[
                    index + '_tu_heat_electric_power',
                    index + '_tu_heat_air_flow'
                ] = (
                        self.control_output_matrix.at[
                            index + '_tu_heat_electric_power',
                            index + '_tu_heat_air_flow'
                        ]
                        + self.parse_parameter('density_air')
                        * (
                                abs(delta_enthalpy_tu_heating)
                                / self.parse_parameter(row['tu_heating_efficiency'])
                                + self.parse_parameter(row['tu_fan_efficiency'])
                        )
                )
                self.control_output_matrix.at[
                    index + '_tu_cool_electric_power',
                    index + '_tu_cool_air_flow'
                ] = (
                        self.control_output_matrix.at[
                            index + '_tu_cool_electric_power',
                            index + '_tu_cool_air_flow']
                        + self.parse_parameter('density_air')
                        * (
                                abs(delta_enthalpy_tu_cooling)
                                / self.parse_parameter(row['tu_cooling_efficiency'])
                                + self.parse_parameter(row['tu_fan_efficiency'])
                        )
                )

    def define_output_fresh_air_flow(self):
        for index, row in self.building_zones.iterrows():
            if row['hvac_ahu_type'] != '':
                self.control_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    index + '_ahu_heat_air_flow'
                ] = 1
                self.control_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    index + '_ahu_cool_air_flow'
                ] = 1
            if row['window_type'] != '':
                self.control_output_matrix.at[
                    index + '_total_fresh_air_flow',
                    index + '_window_air_flow'
                ] = 1
            if (row['window_type'] != '') | (row['hvac_ahu_type'] != ''):
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
            if row['hvac_ahu_type'] != '':
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
            if row['window_type'] != '':
                self.control_output_matrix.at[
                    index + '_window_fresh_air_flow',
                    index + '_window_air_flow'
                ] = 1

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
                        # * (
                        #         1.0
                        #         + (
                        #                 self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                        #                 + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                        #                 + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                        #         )
                        #         / (
                        #                 2.0
                        #                 * self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1)
                        #         )
                        # ) ** (- 1)
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
                        # * (
                        #         1.0
                        #         + (
                        #                 self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                        #                 + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                        #                 + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                        #         )
                        #         / self.parse_parameter('heat_transfer_coefficient_interior_convection')
                        #         + (
                        #                 self.parse_parameter('heat_transfer_coefficient_exterior_convection')
                        #                 + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_ground'])
                        #                 + self.parse_parameter(surface_data['heat_transfer_coefficient_surface_sky'])
                        #         )
                        #         / (self.parse_parameter(surface_data['thermal_resistance_surface']) ** (- 1))
                        # ) ** (- 1)
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
            select * from weather_timeseries 
            where weather_type='{}'
            and time between '{}' and '{}'
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
            select * from building_internal_gain_timeseries 
            where internal_gain_type in ({})
            and time between '{}' and '{}'
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
                ]].reindex(
                    self.set_timesteps
                ).interpolate(
                    'quadratic'
                ),
                building_internal_gain_timeseries.reindex(
                    self.set_timesteps
                ).interpolate(
                    'quadratic'
                ),
                (
                    pd.DataFrame(
                        1.0,
                        self.set_timesteps,
                        ['constant']
                    ) if self.define_constant else pd.DataFrame([])  # Append constant only when needed
                )
            ],
            axis='columns',
        ).rename_axis('disturbance_name', axis='columns')

    def define_output_constraint_timeseries(self, conn):
        """
        - Generate minimum/maximum constraint timeseries based on `building_zone_constraint_profiles`
        - TODO: Make construction / interpolation simpler and more efficient
        """
        # Initialise constraint timeseries as +/- infinity
        self.output_constraint_timeseries_maximum = pd.DataFrame(
            1.0 * np.infty,
            self.set_timesteps,
            self.set_outputs
        )
        self.output_constraint_timeseries_minimum = -self.output_constraint_timeseries_maximum

        # Outputs that are some kind of power can only be positive (greater than zero)
        self.output_constraint_timeseries_minimum.loc[
            :,
            [column for column in self.output_constraint_timeseries_minimum.columns if '_power' in column]
        ] = 0

        # Outputs that are some kind of flow can only be positive (greater than zero)
        self.output_constraint_timeseries_minimum.loc[
            :,
            [column for column in self.output_constraint_timeseries_minimum.columns if '_flow' in column]
        ] = 0

        # If a heating/cooling session is defined, the heating/cooling air flow is forced to 0
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

        for index_zone, row_zone in self.building_zones.iterrows():
            # For each zone, select zone_constraint_profile
            building_zone_constraint_profile = pd.read_sql(
                """
                select * from building_zone_constraint_profiles 
                where zone_constraint_profile='{}'
                """.format(row_zone['zone_constraint_profile']),
                conn
            )

            # Create index function for `from_weekday` (mapping from `row_time.weekday()` to `from_weekday`)
            constraint_profile_index_day = scipy.interpolate.interp1d(
                building_zone_constraint_profile['from_weekday'],
                building_zone_constraint_profile['from_weekday'],
                kind='zero',
                fill_value='extrapolate'
            )
            for row_time in self.set_timesteps:
                # Create index function for `from_time` (mapping `row_time.timestamp` to `from_time`)
                constraint_profile_index_time = scipy.interpolate.interp1d(
                    pd.to_datetime(
                        str(row_time.date())
                        + ' '
                        + building_zone_constraint_profile['from_time'][
                            building_zone_constraint_profile['from_weekday']
                            == constraint_profile_index_day(row_time.weekday())
                        ]
                    ).view('int64'),
                    building_zone_constraint_profile.index[
                        building_zone_constraint_profile['from_weekday']
                        == constraint_profile_index_day(row_time.weekday())
                        ].values,
                    kind='zero',
                    fill_value='extrapolate'
                )

                # Select constraint values
                self.output_constraint_timeseries_minimum.at[
                    row_time,
                    index_zone + '_temperature'
                ] = self.parse_parameter(
                    building_zone_constraint_profile['minimum_air_temperature'][
                        int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                    ]
                )
                self.output_constraint_timeseries_maximum.at[
                    row_time,
                    index_zone + '_temperature'
                ] = self.parse_parameter(
                    building_zone_constraint_profile['maximum_air_temperature'][
                        int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                    ]
                )
                if (row_zone['hvac_ahu_type'] != '') | (row_zone['window_type'] != ''):
                    if self.building_scenarios['demand_controlled_ventilation_type'][0] != '':
                        if self.building_scenarios['co2_model_type'][0] != '':
                            self.output_constraint_timeseries_maximum.at[
                                row_time,
                                index_zone + '_co2_concentration'
                            ] = (
                                self.parse_parameter(
                                    building_zone_constraint_profile['maximum_co2_concentration'][
                                        int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                                    ]
                                )
                            )
                            self.output_constraint_timeseries_minimum.at[
                                row_time,
                                index_zone + '_total_fresh_air_flow'
                            ] = (
                                    self.parse_parameter(
                                        building_zone_constraint_profile['minimum_fresh_air_flow_per_area'][
                                            int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                                        ]
                                    )
                                    * self.parse_parameter(row_zone['zone_area'])
                            )
                        else:
                            self.output_constraint_timeseries_minimum.at[
                                row_time, index_zone + '_total_fresh_air_flow'
                            ] = (
                                    self.parse_parameter(
                                        building_zone_constraint_profile['minimum_fresh_air_flow_per_person'][
                                            int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                                        ]
                                    )
                                    * (
                                            self.disturbance_timeseries[
                                                self.building_zones["internal_gain_type"].loc[index_zone] + "_occupancy"
                                            ].loc[row_time] * self.parse_parameter(row_zone['zone_area'])
                                            + self.parse_parameter(
                                                building_zone_constraint_profile['minimum_fresh_air_flow_per_area'][
                                                    int(constraint_profile_index_time(
                                                        row_time.to_datetime64().astype('int64')
                                                    ))
                                                ]
                                            )
                                            * self.parse_parameter(row_zone['zone_area'])
                                    )
                            )
                    else:
                        if row_zone['hvac_ahu_type'] != '':
                            self.output_constraint_timeseries_minimum.at[
                                row_time,
                                index_zone + '_total_fresh_air_flow'
                            ] = (
                                    self.parse_parameter(
                                        building_zone_constraint_profile['minimum_fresh_air_flow_per_area_no_dcv'][
                                            int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                                        ]
                                    )
                                    * self.parse_parameter(row_zone['zone_area'])
                            )
                        elif row_zone['window_type'] != '':
                            self.output_constraint_timeseries_minimum.at[
                                row_time,
                                index_zone + '_window_fresh_air_flow'
                            ] = (
                                    self.parse_parameter(
                                        building_zone_constraint_profile['minimum_fresh_air_flow_per_area_no_dcv'][
                                            int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
                                        ]
                                    )
                                    * self.parse_parameter(row_zone['zone_area'])
                            )
                # If a ventilation system is enabled, if DCV, then CO2 or constraint on total fresh air flow.
                # If no DCV, then constant constraint on AHU or on windows if no AHU

                if self.building_scenarios['humidity_model_type'][0] != '':
                    if row_zone['hvac_ahu_type'] != '':
                        self.output_constraint_timeseries_minimum.at[
                            row_time,
                            index_zone + '_absolute_humidity'
                        ] = humid_air_properties(
                            'W',
                            'R',
                            self.parse_parameter(
                                building_zone_constraint_profile['minimum_relative_humidity'][
                                    int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
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
                            row_time,
                            index_zone + '_absolute_humidity'
                        ] = humid_air_properties(
                            'W',
                            'R',
                            self.parse_parameter(
                                building_zone_constraint_profile['maximum_relative_humidity'][
                                    int(constraint_profile_index_time(row_time.to_datetime64().astype('int64')))
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

        # Transpose to align with result of state space output matrix equation
        self.output_constraint_timeseries_minimum = self.output_constraint_timeseries_minimum.transpose()
        self.output_constraint_timeseries_maximum = self.output_constraint_timeseries_maximum.transpose()

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
