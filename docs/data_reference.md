# Data reference

``` warning::
    This reference is work in progress.
```

CoBMo scenarios are defined through CSV files, where each CSV file represents a table as defined below (the file name is interpreted as the table name). Internally, CoBMo loads all CSV files into a local SQLITE database for more convenient processing. The default location for CoBMo scenario definitions is in the `data` directory in the repository and all CSV files in the `data` directory are automatically loaded into the database. The CSV files may be structured into sub-directories, but all files are eventually combined into the same database. Hence, all type / element identifiers must be unique across all scenario definitions.

## Scenario data

### `scenarios`

Building scenario definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier. |
| `building_name` | | Building identifier as defined in `buildings`. |
| `parameter_set` | | Parameter set identifier as defined in `parameters`. |
| `linearization_type` | | Type identifier as defined in `linearization_types`. |
| `price_type` | | Type identifier as defined in `electricity_price_timeseries`. |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |

### `parameters`

Parameter definitions, which includes constants and user-defined parameters.

In all tables, a `parameter_name` string can be used to define numerical parameters in place of numerical values. During building model setup, those strings will be parsed from the `parameters` table to obtain the corresponding numerical values.

| Column | Unit | Description |
| --- |:---:| --- |
| `parameter_set` | | Parameter set identifier. |
| `parameter_name` | | Parameter name string. |
| `parameter_value` | various | Numerical value. |
| `parameter_unit` | | Comment field for the parameter unit. |
| `parameter_comment` | | Comment field for further explanations. |

### `initial_state_types`

Initial state settings to be taken for simulation and optimization.

| Column | Unit | Description |
| --- |:---:| --- |
| `initial_state_type` | | Unique type identifier. |
| `initial_zone_temperature` | °C | Initial zone air temperature. |
| `initial_surface_temperature` | °C | Initial surface temperature. |
| `initial_co2_concentration` | ppm | Initial zone air CO₂ concentration. |
| `initial_absolute_humidity` | kg/kg | Initial zone air absolute humidity (mass of water / mass of air). |
| `initial_storage_state_of_charge` | % | Initial state of charge of thermal / battery storage. Maximum value: 100% |

### `linearization_types`

Linearization point definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `linearization_type` | | Unique type identifier. |
| `linearization_zone_air_temperature_heat` | °C | |
| `linearization_zone_air_temperature_cool` | °C | |
| `linearization_surface_temperature` | °C | |
| `linearization_exterior_surface_temperature` | °C | |
| `linearization_ambient_air_temperature` | °C | |
| `linearization_sky_temperature` | °C | |
| `linearization_zone_air_absolute_humidity` | kg/kg | Absolute humidity (mass of water / mass of air). |
| `linearization_ambient_air_absolute_humidity` | kg/kg | Absolute humidity (mass of water / mass of air). |
| `linearization_zone_air_co2_concentration` | ppm | |
| `linearization_zone_fresh_air_flow` | l/s/m² | |

### `electricity_price_timeseries`

Electricity price time series definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Price type identifier. |
| `time` | | Timestamp according to ISO 8601. |
| `price` | $/kWh | Price value. *Currently, prices / costs are assumed to be in SGD.* |

## Building data

### `buildings`

Building definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Unique building identifier.|
| `weather_type` | | Type identifier as defined in `weather_types`. |
| `storage_type` | | Type identifier as defined in `storage_types`. |

### `weather_types`

Additional weather and site information.

| Column | Unit | Description |
| --- |:---:| --- |
| `weather_type` | | Unique type identifier. |
| `time_zone` | | Time zone indicator according to ISO 8601. |
| `latitude` | | Latitude of the weather station. |
| `longitude` | | Longitude of the weather station. |
| `temperature_difference_sky_ambient` | K | *To be revised.* |

### `weather_timeseries`

Weather time series definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `weather_type` | | Type identifier as defined in `weather_types`. |
| `time` | | Timestamp according to ISO 8601. |
| `ambient_air_temperature` | °C | Ambient air dry-bulb temperature. |
| `sky_temperature` | °C | Equivalent sky temperature. |
| `ambient_air_absolute_humidity` | kg/kg | Ambient air absolute humidity (mass of water / mass of air). |
| `irradiation_horizontal` | W/m² | Irradiation onto a horizontal surface. |
| `irradiation_east` | W/m² | Irradiation onto a vertical surface oriented towards East. |
| `irradiation_south` | W/m² | Irradiation onto a vertical surface oriented towards South. |
| `irradiation_west` | W/m² | Irradiation onto a vertical surface oriented towards West. |
| `irradiation_north` | W/m² | Irradiation onto a vertical surface oriented towards North. |

### `storage_types`

Storage characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `storage_type` | | Unique type identifier. |
| `storage_commodity_type` | | Storage commodity type. Choices: `battery`, `sensible_cooling`, `sensible_heating` |
| `storage_capacity` | m³/m² or kWh/m² | Volume of sensible thermal storage or electric energy capacity of battery storage. Define in per-square-meter of the total building zone area. |
| `storage_round_trip_efficiency` | - | Round trip efficiency (discharged energy / charged energy). |
| `storage_battery_depth_of_discharge` | - | Maximum utilizable battery storage capacity (utilized capacity / total capacity). *Only needed for commodity type `battery`.* |
| `storage_sensible_temperature_delta` | K | Temperature difference between the supply and return water temperature of sensible thermal storage. |
| `storage_self_discharge_rate` | %/h | Storage self-discharge rate. |

## Zone data

### `zones`

Zone geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Building identifier as defined in `buildings`. |
| `zone_name` | | Unique zone identifier. |
| `zone_type` | | Type identifier as defined in `zone_types`. |
| `zone_height` | m | Zone height (clear interior height from base to ceiling). |
| `zone_area` | m² | Zone area (interior area excluding surfaces). |
| `zone_comment` | | Explanatory comment, e.g., description of the geometry. |

### `zone_types`

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
| `fresh_air_flow_control_type` | | Control mode for the fresh air flow / indoor air quality (IAQ). Choices: empty (IAQ is ensured by providing a fixed amount of fresh air per zone area, as defined by `minimum_fresh_air_flow` in `constraint_schedules`.), `occupancy_based` (IAQ is ensured by providing a fixed amount of fresh air per person and per zone area, as defined by `minimum_fresh_air_flow_occupants`, `minimum_fresh_air_flow_building` in `constraint_schedules`.), `co2_based` (IAQ is ensured by constraining the CO₂ concentration, as defined by `maximum_co2_concentration` in `constraint_schedules`.) |
| `humidity_control_type` | | Control mode for the humidity level. Choices: empty (Humidity is not controlled.), `humidity_based` (Humidity is constrained, as defined by `minimum_relative_humidity` and `maximum_relative_humidity` in `constraint_schedules`.) |

### `internal_gain_types`

Internal gain type definitions.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | Unique type identifier. |
| `internal_gain_definition_type` | | Definition type, corresponding to definitions in `internal_gain_schedules` or `internal_gain_timeseries`. Choices: `schedule` (Defines occupancy / appliance schedule in `internal_gain_schedules`.), `timeseries` (Defines occupancy / appliance timeseries in `internal_gain_timeseries`.) |
| `occupancy_density` | person/m² | Peak occupancy density. |
| `occupancy_heat_gain` | W/person | Sensible heat gain due to occupants. |
| `occupancy_co2_gain` | ml/s/person | CO₂ gain due to occupants. *Only required when using `iaq_based` fresh air flow control in `zone_types`.*  |
| `occupancy_humidity_gain` | g/h/person | Moisture gain due to occupants. *Only required when using `humidity_based` humidity control in `zone_types`.* |
| `appliances_heat_gain` | W/m² | Peak sensible heat gain due to appliances. |

### `internal_gain_schedules`

The internal gain timeseries is constructed by obtaining the appropriate value for `internal_gain_occupancy` / `internal_gain_appliances` based on the `time_period` in `ddTHH:MM` format. Higher resolution time steps are interpolated linearly and lower resolution time steps are aggregated using mean values. The daily schedule is repeated for any weekday greater than or equal to `dd` until the next defined `dd`. The initial value for each `internal_gain_type` must start at `time_period = 01T00:00`.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | Type identifier as defined in `internal_gain_types`. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `internal_gain_occupancy` | - | Occupancy rate between 0 and 1, where 1 represents peak occupancy. |
| `internal_gain_appliances` | - | Appliance activity rate between 0 and 1, where 1 represents peak appliance activity. |

### `internal_gain_timeseries`

Time series of internal gains. Higher resolution time steps are interpolated linearly and lower resolution time steps are aggregated using mean values.

| Column | Unit | Description |
| --- |:---:| --- |
| `internal_gain_type` | | Type identifier as defined in `internal_gain_types`. |
| `time` | | Timestamp according to ISO 8601. |
| `internal_gain_occupancy` | - | Occupancy rate between 0 and 1, where 1 represents peak occupancy. |
| `internal_gain_appliances` | - | Appliance activity rate between 0 and 1, where 1 represents peak appliance activity. |

### `hvac_generic_types`

Generic HVAC system characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_generic_type` | | Unique type identifier. |
| `generic_heating_efficiency` | - | Heating efficiency (thermal power / electric power). |
| `generic_cooling_efficiency` | - | Cooling efficiency (thermal power / electric power). |

### `hvac_ahu_types`

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

### `hvac_tu_types`

Terminal unit (TU) set points and characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `hvac_tu_type` | | Unique type identifier. |
| `tu_air_intake_type` | | *Currently not used.* |
| `tu_supply_air_temperature_setpoint` | °C | Supply air temperature at outlet of TU towards zone. |
| `tu_fan_efficiency` | J/kg | Fan efficiency (electric power / air mass flow rate). |
| `tu_cooling_efficiency` | - | Chiller plant efficiency (thermal power / electrical power). |
| `tu_heating_efficiency` | - | Heating plant efficiency (thermal power / electrical power). |

### `hvac_radiator_types`

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

### `blind_types`

Window blind characteristics. *Currently not used.*

| Column | Unit | Description |
| --- |:---:| --- |
| `blind_type` | | Unique type identifier. |
| `blind_efficiency` | - | Blind efficiency (absorbed irradiation / incident irradiation). |

### `constraint_schedules`

The constraint timeseries is constructed by obtaining the appropriate value for `internal_gain_occupancy` / `internal_gain_appliances` based on the `time_period` in `ddTHH:MM` format. Intermediate time steps are interpolated linearly. The daily schedule is repeated for any weekday greater than or equal to `dd` until the next defined `dd`. The initial value for each `constraint_type` must start at `time_period = 01T00:00`.

| Column | Unit | Description |
| --- |:---:| --- |
| `constraint_type` | | Constraint type identifier. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `minimum_air_temperature` | °C | Maximum indoor air temperature. |
| `maximum_air_temperature` | °C | Maximum indoor air temperature. |
| `minimum_fresh_air_flow` | l/s/m² | Minimum fresh air supply based on floor area (includes ventilation requirements due to occupancy). *Not required when using `occupancy_based` or `iaq_based` fresh air flow control in `zone_types`.* |
| `minimum_fresh_air_flow_building` | l/s/m² | Minimum fresh air supply for removing building emissions. *Only required when using `occupancy_based` fresh air flow control in `zone_types`.*|
| `minimum_fresh_air_flow_occupants` | l/s/person | Minimum fresh air supply for removing occupant emissions. *Only required when using `occupancy_based` fresh air flow control in `zone_types`.* |
| `maximum_co2_concentration` | ppm | Maximum indoor air CO₂ concentration. *Only required when using `iaq_based` fresh air flow control in `zone_types`.* |
| `minimum_relative_humidity` | % | Minimum indoor air relative humidity. *Only required when using `humidity_based` humidity control in `zone_types`.* |
| `maximum_relative_humidity` | % | Maximum indoor air relative humidity. *Only required when using `humidity_based` humidity control in `zone_types`.* |

## Surface data

### `surfaces_exterior`

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

### `surfaces_interior`

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

### `surfaces_adiabatic`

Adiabatic surfaces geometry.

| Column | Unit | Description |
| --- |:---:| --- |
| `building_name` | | Building identifier as defined in `buildings`. |
| `zone_name` | | Zone identifier as defined in `zones`. |
| `surface_name` | | Unique surface identifier. |
| `surface_type` | | Type identifier as defined in `surface_types`. |
| `surface_area` | m² | Surface area. |
| `surface_comment` | | Explanatory comment, e.g., description of the geometry. |

### `surface_types`

Surface type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `surface_type` | | Unique type identifier. |
| `heat_capacity` | J/(m³K) | Specific heat capacity. |
| `heat_transfer_coefficient_surface_conduction` | W/m²K | Conductive internal heat transfer coefficient. |
| `absorptivity` | - | Absorptivity factor / irradiation gain coefficient (absorbed irradiation / incident irradiation). |
| `emissivity` | - | Emissivity factor. |
| `window_type` | | Type identifier as defined in `window_types`. |
| `window_wall_ratio` | - | Window surface area / total surface area. |
| `sky_view_factor` | - | Sky view factor. |

### `window_types`

Window characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `window_type` | | Unique type identifier. |
| `thermal_resistance_window` | m²K/W | Specific thermal resistance. |
| `absorptivity_window` | - | Irradiation gain coefficient (absorbed irradiation / incident irradiation). |
| `emissivity_window` | - | Emissivity factor. |
