# Data reference

``` warning::
    This reference is work in progress.
```

## `building_blind_types`

Window blind characteristics. *Currently not used.*

| Column | Unit | Description |
| --- |:---:| --- |
| `blind_type` | | |
| `blind_efficiency` | | |

## `building_hvac_ahu_types`

Air handling unit (AHU) set points and characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_ahu_type` | | |
| `ahu_cooling_type` | | *Currently not used.* |
| `ahu_heating_type` | | *Currently not used.* |
| `ahu_dehumidification_type` | | *Currently not used.* |
| `ahu_return_air_heat_recovery_type` | | *Currently not used.* |
| `ahu_supply_air_temperature_setpoint` | °C | |
| `ahu_supply_air_relative_humidity_setpoint` | % | |
| `ahu_fan_efficiency` | J/kg | Fan efficiency (electric power / air mass flow rate). |
| `ahu_cooling_efficiency` | - | Chiller plant efficiency (thermal power / electrical power). |
| `ahu_heating_efficiency` | - | Heating plant efficiency (thermal power / electrical power). |
| `ahu_return_air_heat_recovery_efficiency` | - | Recovery efficiency (recovered power / available power). |

## `building_hvac_generic_types`

Generic HVAC system characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_generic_type` | | |
| `generic_heating_efficiency` | - | Heating efficiency (thermal power / electric power). |
| `generic_cooling_efficiency` | - | Cooling efficiency (thermal power / electric power). |

## `building_hvac_tu_types`

Terminal unit (TU) set points and characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_tu_type` | | |
| `tu_cooling_type` | | *Currently not used.* |
| `tu_heating_type` | | *Currently not used.* |
| `tu_air_intake_type` | | *Currently not used.* |
| `tu_supply_air_temperature_setpoint` | °C | |
| `tu_fan_efficiency` | J/kg | Fan efficiency (electric power / air mass flow rate). |
| `tu_cooling_efficiency` | - | Chiller plant efficiency (thermal power / electrical power). |
| `tu_heating_efficiency` | - | Heating plant efficiency (thermal power / electrical power). |

## `building_internal_gain_timeseries`

Time series of internal gains.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | |
| `time` | | Timestamp according to ISO 8601. |
| `internal_gain_occupancy` | W/m² | Internal gains related to occupants.|
| `internal_gain_appliances` | W/m² | Internal gains related to appliances. |

## `building_internal_gain_types`

Internal gain type definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | |
| `internal_gain_occupancy_factor` | - | Auxiliary scaling factor. |
| `internal_gain_appliances_factor` | - | Auxiliary scaling factor. |

## `building_linearization_types`

Linearization point definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `linearization_type` | | |
| `linearization_zone_air_temperature_heat` | °C | |
| `linearization_zone_air_temperature_cool` | °C | |
| `linearization_surface_temperature` | °C | |
| `linearization_exterior_surface_temperature` | °C | |
| `linearization_internal_gain_occupancy` | W/m² | |
| `linearization_internal_gain_appliances` | W/m² | |
| `linearization_ambient_air_temperature` | °C | |
| `linearization_sky_temperature` | °C | |
| `linearization_ambient_air_humidity_ratio` | kg/kg | Mass of water / mass of air. |
| `linearization_zone_air_humidity_ratio` | kg/kg | Mass of water / mass of air. |
| `linearization_irradiation` | W/m² | |
| `linearization_co2_concentration` | ppm | |
| `linearization_ventilation_rate_per_square_meter` | m/s | |

## `building_parameter_sets`

Parameter definitions, which includes constants and user-defined parameters. In all other tables, the `parameter_name` can be used to define numerical parameters in place of numerical values. During building model setup, those parameters will be parsed from the `building_parameter_sets` table.

| Column | Unit | Description |
| --- |:---:| --- |
| `parameter_set` | | |
| `parameter_name` | | Parameter name string. |
| `parameter_value` | various | Numerical value. |
| `parameter_unit` | | Comment field for the parameter unit. |
| `parameter_comment` | | Comment field for further explanations. |

## `building_scenarios`

Building scenario definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | |
| `building_name` | | |
| `parameter_set` | | |
| `linearization_type` | | |
| `demand_controlled_ventilation_type` | | |
| `co2_model_type` | | |
| `humidity_model_type` | | |
| `heating_cooling_session` | | *Currently not used.* |
| `price_type` | | |
| `time_start` | | Timestamp according to ISO 8601. |
| `time_end` | | Timestamp according to ISO 8601. |
| `time_step` | | Time step length in seconds. |

## `building_storage_types`

Storage characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_storage_type` | | |
| `storage_size` | | |
| `storage_round_trip_efficiency` | | |
| `storage_battery_depth_of_discharge` || | |
| `storage_sensible_temperature_delta` | | |
| `storage_lifetime` || | |
| `storage_planning_energy_installation_cost` | | | |
| `storage_planning_power_installation_cost` || | |
| `storage_planning_fixed_installation_cost` || | |

## `building_surface_types`

Surface type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `surface_type` | | |
| `heat_capacity` | J/(m³K) | Specific heat capacity. |
| `thermal_resistance_surface` | m²K/W | Specific thermal resistance. |
| `irradiation_gain_coefficient` | - | Irradiation gain coefficient (absorbed irradiation / incident irradiation). |
| `emissivity` | - | Emissivity factor. |
| `window_type` | | |
| `window_wall_ratio` | - | Window surface area / total surface area. |
| `sky_view_factor` | - | Sky view factor. |

## `building_surfaces_adiabatic`

Adiabatic surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | |
| `zone_name` | | |
| `surface_name` | | |
| `surface_type` | | |
| `surface_area` | m² | |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

## `building_surfaces_exterior`

Exterior surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | |
| `zone_name` | | |
| `direction_name` | | |
| `surface_name` | | |
| `surface_type` | | |
| `surface_area` | m² | |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

## `building_surfaces_interior`

Interior surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | |
| `zone_name` | | |
| `zone_adjacent_name` | | |
| `surface_name` | | |
| `surface_type` | | |
| `surface_area` | m² | |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

## `building_window_types`

Window characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `window_type` | | |
| `thermal_resistance_window` | m²K/W | Specific thermal resistance. |
| `irradiation_gain_coefficient_window` | - | Irradiation gain coefficient (absorbed irradiation / incident irradiation). |
| `emissivity_window` | - | Emissivity factor. |

## `building_zone_constraint_profiles`

Constraint profile definitions. The constraints are constructed by keeping a constant value for given `from_time` until the next values' `from_time`. The profile is repeated for each weekday from the given `from_weekday`.

| Column | Unit | Description |
| --- |:---:| --- |
| `zone_constraint_profile` | | |
| `from_weekday` | | Start weekday number (0 - Monday ... 6 - Sunday). | |
| `from_time` | | Start time in HH:MM:SS format. | |
| `minimum_air_temperature` | °C | |
| `maximum_air_temperature` | °C | |
| `minimum_fresh_air_flow_per_area` | m/s | |
| `minimum_fresh_air_flow_per_person` | m³/s/person | |
| `maximum_co2_concentration` | ppm | |
| `minimum_fresh_air_flow_per_area_no_dcv` | m/s | |
| `minimum_relative_humidity` | % | |
| `maximum_relative_humidity` | % | |


## `building_zone_types`

Zone type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `zone_type` | | |
| `heat_capacity` | J/(m³K) | Specific heat capacity. |
| `base_surface_type` | | |
| `ceiling_surface_type` | | |
| `infiltration_rate` | 1/h | |
| `internal_gain_type` | | |
| `window_type` | | |
| `blind_type` | | *currently not used.* |
| `hvac_generic_type` | | |
| `hvac_ahu_type` | | |
| `hvac_tu_type` | | |
| `zone_constraint_profile` | | |

## `building_zones`

Zone geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | |
| `zone_name` | | |
| `zone_type` | | |
| `zone_height` | m | |
| `zone_area` | m² | |
| `zone_comment` | | Explanatory comment, e.g., description of the geometry. |

## `buildings`

Building definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | |
| `weather_type` | | |
| `building_storage_type` | | |

## `electricity_price_timeseries`

Electricity price time series definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | |
| `time` | | Timestamp according to ISO 8601. |
| `price` | SGD | *Currently only in SGD* |

## `weather_timeseries`

Weather time series definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `weather_type` | | |
| `time` | | Timestamp according to ISO 8601. |
| `ambient_air_temperature` | °C | |
| `sky_temperature` | °C | |
| `ambient_air_humidity_ratio` | kg/kg | Mass of water / mass of air. |
| `irradiation_horizontal` | W/m² | |
| `irradiation_east` | W/m² | |
| `irradiation_south` | W/m² | |
| `irradiation_west` | W/m² | |
| `irradiation_north` | W/m² | |

## `weather_types`

Additional weather and site information.

| Column | Unit | Description |
| --- |:---:| --- |
| `weather_type` | | |
| `time_zone` | | Time zone indicator according to ISO 8601. |
| `latitude` | | |
| `longitude` | | |
| `temperature_difference_sky_ambient` | K | |
