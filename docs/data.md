# Data reference

``` warning::
    This reference is work in progress.
```

## `blind_types`

Window blind characteristics. *Currently not used.*

| Column | Unit | Description |
| --- |:---:| --- |
| `blind_type` | | Unique type identifier. |
| `blind_efficiency` | - | Blind efficiency (absorbed irradiation / incident irradiation). |

## `buildings`

Building definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Unique building identifier.|
| `weather_type` | | Type identifier as defined in `weather_types`. |
| `building_storage_type` | | Type identifier as defined in `storage_types`. |

## `constraint_schedules`

The constraint timeseries is constructed by obtaining the appropriate value for `internal_gain_occupancy` / `internal_gain_appliances` based on the `time_period` in `ddTHH:MM:SS` format. Each value is kept constant at the given value for any daytime greater than or equal to `HH:MM:SS` and any weekday greater than or equal to `dd` until the next defined `ddTHH:MM:SS`. Note that the daily schedule is repeated for any weekday greater than or equal to `dd` until the next defined `dd`. The initial value for each `zone_constraint_profile` must start at `time_period = 00T00:00:00`.

| Column | Unit | Description |
| --- |:---:| --- |
| `constraint_type` | | Constraint type identifier. |
| `time_period` | | Time period in `ddTHH:MM:SS` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM:SS` is the daytime. |
| `minimum_air_temperature` | °C | |
| `maximum_air_temperature` | °C | |
| `minimum_fresh_air_flow_per_area` | m/s | |
| `minimum_fresh_air_flow_per_person` | m³/s/person | |
| `maximum_co2_concentration` | ppm | |
| `minimum_fresh_air_flow_per_area_no_dcv` | m/s | |
| `minimum_relative_humidity` | % | |
| `maximum_relative_humidity` | % | |

## `electricity_price_timeseries`

Electricity price time series definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Price type identifier. |
| `time` | | Timestamp according to ISO 8601. |
| `price` | SGD | *Currently only in SGD* |

## `hvac_ahu_types`

Air handling unit (AHU) set points and characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_ahu_type` | | Unique type identifier. |
| `ahu_supply_air_temperature_setpoint` | °C | Supply air temperature at outlet of AHU towards zone. |
| `ahu_supply_air_relative_humidity_setpoint` | % | Supply air relative humidity at outlet of AHU towards zone. |
| `ahu_fan_efficiency` | J/kg | Fan efficiency (electric power / air mass flow rate). |
| `ahu_cooling_efficiency` | - | Chiller plant efficiency (thermal power / electrical power). |
| `ahu_heating_efficiency` | - | Heating plant efficiency (thermal power / electrical power). |
| `ahu_return_air_heat_recovery_efficiency` | - | Recovery efficiency (recovered power / available power). |

## `hvac_generic_types`

Generic HVAC system characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_generic_type` | | Unique type identifier. |
| `generic_heating_efficiency` | - | Heating efficiency (thermal power / electric power). |
| `generic_cooling_efficiency` | - | Cooling efficiency (thermal power / electric power). |

## `hvac_radiator_types`

| Column | Unit | Description |
| --- |:---:| --- |
| `radiator_supply_temperature_nominal` | °C | |
| `radiator_return_temperature_nominal` | °C | |
| `hvac_radiator_type` | | Unique type identifier. |
| `radiator_panel_number` | - | Number of radiator panels. |
| `radiator_panel_area` | m² | Radiator panel surface area. |
| `radiator_panel_thickness` | m | Radiator panel thickness. |
| `radiator_water_volume` | m³ | Radiator panel water volume. |
| `radiator_convection_coefficient` | W/(m²K) | Natural convection coefficient. |
| `radiator_emissivity` | - | Emissivity factor. |
| `radiator_hull_conductivity` | W/(mK) | Radiator hull material conductivity. |
| `radiator_hull_heat_capacity` | J/m³ | Radiator hull material heat capacity. |
| `radiator_fin_effectiveness` | - | Radiator fin effectiveness. |

## `hvac_tu_types`

Terminal unit (TU) set points and characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_tu_type` | | Unique type identifier. |
| `tu_air_intake_type` | | *Currently not used.* |
| `tu_supply_air_temperature_setpoint` | °C | Supply air temperature at outlet of TU towards zone. |
| `tu_fan_efficiency` | J/kg | Fan efficiency (electric power / air mass flow rate). |
| `tu_cooling_efficiency` | - | Chiller plant efficiency (thermal power / electrical power). |
| `tu_heating_efficiency` | - | Heating plant efficiency (thermal power / electrical power). |

## `initial_state_types`

Initial state settings to be taken for simulation and optimization.

| Column | Unit | Description |
| --- |:---:| --- |
| initial_state_type | | Unique type identifier. |
| initial_zone_temperature | °C | Initial zone air temperature. |
| initial_surface_temperature | °C | Initial surface temperature. |
| initial_co2_concentration | ppm | Initial zone air CO₂ concentration. |
| initial_absolute_humidity | kg/kg | Initial zone air absolute humidity (mass of water / mass of air). |
| initial_sensible_thermal_storage_state_of_charge | m³ | Initial stored volume of useable water layer in the sensible thermal storage. |
| initial_battery_storage_state_of_charge | kWh | Initial stored electric energy in the battery storage.|

## `internal_gain_schedules`

The internal gain timeseries is constructed by obtaining the appropriate value for `internal_gain_occupancy` / `internal_gain_appliances` based on the `time_period` in `ddTHH:MM:SS` format. Each value is kept constant at the given value for any daytime greater than or equal to `HH:MM:SS` and any weekday greater than or equal to `dd` until the next defined `ddTHH:MM:SS`. Note that the daily schedule is repeated for any weekday greater than or equal to `dd` until the next defined `dd`. The initial value for each `zone_constraint_profile` must start at `time_period = 00T00:00:00`.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | Type identifier as defined in `internal_gain_types`. |
| `time_period` | | Time period in `ddTHH:MM:SS` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM:SS` is the daytime. |
| `internal_gain_occupancy` | W/m² | Internal gains related to occupants.|
| `internal_gain_appliances` | W/m² | Internal gains related to appliances. |

## `internal_gain_timeseries`

Time series of internal gains.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | Type identifier as defined in `internal_gain_types`. |
| `time` | | Timestamp according to ISO 8601. |
| `internal_gain_occupancy` | W/m² | Internal gains related to occupants.|
| `internal_gain_appliances` | W/m² | Internal gains related to appliances. |

## `internal_gain_types`

Internal gain type definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | Unique type identifier. |
| `internal_gain_definition_type` | | Definition type, corresponding to definitions in `internal_gain_schedules` or `internal_gain_timeseries`. Choices: `schedule`, `timeseries`. |
| `internal_gain_occupancy_factor` | - | Occupancy gains scaling factor. |
| `internal_gain_appliances_factor` | - | Appliance gains scaling factor. |

## `linearization_types`

Linearization point definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `linearization_type` | | Unique type identifier. |
| `linearization_zone_air_temperature_heat` | °C | |
| `linearization_zone_air_temperature_cool` | °C | |
| `linearization_surface_temperature` | °C | |
| `linearization_exterior_surface_temperature` | °C | |
| `linearization_internal_gain_occupancy` | W/m² | |
| `linearization_internal_gain_appliances` | W/m² | |
| `linearization_ambient_air_temperature` | °C | |
| `linearization_sky_temperature` | °C | |
| `linearization_ambient_air_humidity_ratio` | kg/kg | Absolute humidity (mass of water / mass of air). |
| `linearization_zone_air_humidity_ratio` | kg/kg | Absolute humidity (mass of water / mass of air). |
| `linearization_irradiation` | W/m² | |
| `linearization_co2_concentration` | ppm | |
| `linearization_ventilation_rate_per_square_meter` | m/s | |

## `parameters`

Parameter definitions, which includes constants and user-defined parameters.

In all tables, a `parameter_name` string can be used to define numerical parameters in place of numerical values. During building model setup, those strings will be parsed from the `parameters` table to obtain the corresponding numerical values.

| Column | Unit | Description |
| --- |:---:| --- |
| `parameter_set` | | Parameter set identifier. |
| `parameter_name` | | Parameter name string. |
| `parameter_value` | various | Numerical value. |
| `parameter_unit` | | Comment field for the parameter unit. |
| `parameter_comment` | | Comment field for further explanations. |

## `scenarios`

Building scenario definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier. |
| `building_name` | | Building identifier as defined in `buildings`. |
| `parameter_set` | | Parameter set identifier as defined in `parameters`. |
| `linearization_type` | | Type identifier as defined in `linearization_types`. |
| `demand_controlled_ventilation_type` | | *Currently not used.* |
| `co2_model_type` | | *Currently not used.* |
| `humidity_model_type` | | *Currently not used.* |
| `heating_cooling_session` | | *Currently not used.* |
| `price_type` | | Type identifier as defined in `electricity_price_timeseries`. |
| `time_start` | | Timestamp according to ISO 8601. |
| `time_end` | | Timestamp according to ISO 8601. |
| `time_step` | | Time step length in seconds. |

## `storage_types`

Storage characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_storage_type` | | Unique type identifier. |
| `storage_size` | m³ or kWh | Volume of sensible thermal storage or electric energy capacity of battery storage. |
| `storage_round_trip_efficiency` | - | Round trip efficiency (discharged energy / charged energy). |
| `storage_battery_depth_of_discharge` | - | Maximum utilizable battery storage capacity (utilized capacity / total capacity). |
| `storage_sensible_temperature_delta` | K | Temperature difference between the supply and return water temperature of sensible thermal storage. |
| `storage_lifetime` | years | Storage lifetime. *Only used for storage planning problems.* |
| `storage_planning_energy_installation_cost` | SGD/m³ or SGD/kWh | Investment cost per installed energy capacity. *Only used for storage planning problems.* |
| `storage_planning_power_installation_cost` | SGD/kW | Investment cost per installed power capacity. *Only used for storage planning problems.* |
| `storage_planning_fixed_installation_cost` | SGD | Investment cost per installed storage unit. *Only used for storage planning problems.* |

## `surface_types`

Surface type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `surface_type` | | Unique type identifier. |
| `heat_capacity` | J/(m³K) | Specific heat capacity. |
| `thermal_resistance_surface` | m²K/W | Specific thermal resistance. |
| `absorptivity` | - | Absorptivity factor / irradiation gain coefficient (absorbed irradiation / incident irradiation). |
| `emissivity` | - | Emissivity factor. |
| `window_type` | | Type identifier as defined in `window_types`. |
| `window_wall_ratio` | - | Window surface area / total surface area. |
| `sky_view_factor` | - | Sky view factor. |

## `surfaces_adiabatic`

Adiabatic surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Building identifier as defined in `buildings`. |
| `zone_name` | | Zone identifier as defined in `zones`. |
| `surface_name` | | Unique surface identifier. |
| `surface_type` | | Type identifier as defined in `surface_types`. |
| `surface_area` | m² | Surface area. |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

## `surfaces_exterior`

Exterior surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Building identifier as defined in `buildings`. |
| `zone_name` | | Zone identifier as defined in `zones`. |
| `direction_name` | | Direction name (`horizontal`, `east`, `south`, `west`, `north`). |
| `surface_name` | | Unique surface identifier. |
| `surface_type` | | Type identifier as defined in `surface_types`. |
| `surface_area` | m² | Surface area. |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

## `surfaces_interior`

Interior surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Building identifier as defined in `buildings`. |
| `zone_name` | | Zone on primary side identifier as defined in `zones`. |
| `zone_adjacent_name` | | Zone on secondary side identifier as defined in `zones`. |
| `surface_name` | | Unique surface identifier. |
| `surface_type` | | Type identifier as defined in `surface_types`. |
| `surface_area` | m² | Surface area. |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

## `weather_timeseries`

Weather time series definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `weather_type` | | Type identifier as defined in `weather_types`. |
| `time` | | Timestamp according to ISO 8601. |
| `ambient_air_temperature` | °C | Ambient air dry-bulb temperature. |
| `sky_temperature` | °C | Equivalent sky temperature. |
| `ambient_air_humidity_ratio` | kg/kg | Ambient air absolute humidity (mass of water / mass of air). |
| `irradiation_horizontal` | W/m² | Irradiation onto a horizontal surface. |
| `irradiation_east` | W/m² | Irradiation onto a vertical surface oriented towards East. |
| `irradiation_south` | W/m² | Irradiation onto a vertical surface oriented towards South. |
| `irradiation_west` | W/m² | Irradiation onto a vertical surface oriented towards West. |
| `irradiation_north` | W/m² | Irradiation onto a vertical surface oriented towards North. |

## `weather_types`

Additional weather and site information.

| Column | Unit | Description |
| --- |:---:| --- |
| `weather_type` | | Unique type identifier. |
| `time_zone` | | Time zone indicator according to ISO 8601. |
| `latitude` | | Latitude of the weather station. |
| `longitude` | | Longitude of the weather station. |
| `temperature_difference_sky_ambient` | K | *To be revised.* |

## `window_types`

Window characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `window_type` | | Unique type identifier. |
| `thermal_resistance_window` | m²K/W | Specific thermal resistance. |
| `absorptivity_window` | - | Irradiation gain coefficient (absorbed irradiation / incident irradiation). |
| `emissivity_window` | - | Emissivity factor. |

## `zone_types`

Zone type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `zone_type` | | Unique type identifier.|
| `heat_capacity` | J/(m³K) | Specific heat capacity. |
| `infiltration_rate` | 1/h | Infiltration rate. |
| `internal_gain_type` | | Type identifier as defined in `internal_gain_types`. |
| `window_type` | | Type identifier as defined in `window_types`. |
| `blind_type` | | Type identifier as defined in `blind_types`. *Currently not used.* |
| `hvac_generic_type` | | Type identifier as defined in `hvac_generic_types`. |
| `hvac_radiator_type` | | Type identifier as defined in `hvac_radiator_types`. |
| `hvac_ahu_type` | | Type identifier as defined in `hvac_ahu_types`. |
| `hvac_tu_type` | | Type identifier as defined in `hvac_tu_types`. |
| `constraint_type` | | Constraint type identifier as defined in `constraint_schedules`. |

## `zones`

Zone geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Building identifier as defined in `buildings`. |
| `zone_name` | | Unique zone identifier. |
| `zone_type` | | Type identifier as defined in `zone_types`. |
| `zone_height` | m | Zone height (clear interior height from base to ceiling). |
| `zone_area` | m² | Zone area (interior area excluding surfaces). |
| `zone_comment` | | Explanatory comment, e.g., description of the geometry. |
