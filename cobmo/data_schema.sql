CREATE TABLE IF NOT EXISTS buildings (
    building_name TEXT,
    weather_type TEXT,
    plant_cooling_type TEXT,
    plant_heating_type TEXT,
    storage_type TEXT,
    PRIMARY KEY(building_name)
);
CREATE TABLE IF NOT EXISTS constraint_schedules (
    constraint_type TEXT,
    time_period TEXT,
    minimum_air_temperature TEXT,
    maximum_air_temperature TEXT,
    minimum_fresh_air_flow TEXT,
    minimum_fresh_air_flow_building TEXT,
    minimum_fresh_air_flow_occupants TEXT,
    maximum_co2_concentration TEXT,
    minimum_relative_humidity TEXT,
    maximum_relative_humidity TEXT,
    PRIMARY KEY(constraint_type,time_period)
);
CREATE TABLE IF NOT EXISTS electricity_price_timeseries (
    price_type TEXT,
    time TEXT,
    price REAL,
    PRIMARY KEY(price_type,time)
);
CREATE TABLE IF NOT EXISTS electricity_price_range (
    time_period TEXT,
    price_mean REAL,
    delta_upper REAL,
    delta_lower REAL
);
CREATE TABLE IF NOT EXISTS hvac_ahu_types (
    hvac_ahu_type TEXT,
    ahu_supply_air_temperature_setpoint TEXT,
    ahu_supply_air_relative_humidity_setpoint TEXT,
    ahu_fan_efficiency TEXT,
    ahu_cooling_efficiency TEXT,
    ahu_heating_efficiency TEXT,
    ahu_return_air_heat_recovery_efficiency TEXT,
    PRIMARY KEY(hvac_ahu_type)
);
CREATE TABLE IF NOT EXISTS hvac_generic_types (
    hvac_generic_type TEXT,
    generic_heating_efficiency TEXT,
    generic_cooling_efficiency TEXT,
    PRIMARY KEY(hvac_generic_type)
);
CREATE TABLE IF NOT EXISTS hvac_radiator_types (
    hvac_radiator_type TEXT,
    radiator_supply_temperature_nominal TEXT,
    radiator_return_temperature_nominal TEXT,
    radiator_panel_number TEXT,
    radiator_panel_area TEXT,
    radiator_panel_thickness TEXT,
    radiator_water_volume TEXT,
    radiator_convection_coefficient TEXT,
    radiator_emissivity TEXT,
    radiator_hull_conductivity TEXT,
    radiator_hull_heat_capacity TEXT,
    radiator_fin_effectiveness TEXT,
    radiator_heating_efficiency TEXT,
    PRIMARY KEY(hvac_radiator_type)
);
CREATE TABLE IF NOT EXISTS hvac_tu_types (
    hvac_tu_type TEXT,
    tu_air_intake_type TEXT,
    tu_supply_air_temperature_setpoint TEXT,
    tu_fan_efficiency TEXT,
    tu_cooling_efficiency TEXT,
    tu_heating_efficiency TEXT,
    PRIMARY KEY(hvac_tu_type)
);
CREATE TABLE IF NOT EXISTS hvac_vent_types (
    hvac_vent_type TEXT,
    vent_fan_efficiency TEXT,
    PRIMARY KEY(hvac_vent_type)
);
CREATE TABLE IF NOT EXISTS initial_state_types (
    initial_state_type TEXT,
    initial_zone_temperature TEXT,
    initial_surface_temperature TEXT,
    initial_co2_concentration TEXT,
    initial_absolute_humidity TEXT,
    initial_storage_state_of_charge TEXT,
    PRIMARY KEY(initial_state_type)
);
CREATE TABLE IF NOT EXISTS internal_gain_schedules (
    internal_gain_type TEXT,
    time_period TEXT,
    internal_gain_occupancy REAL DEFAULT 0,
    internal_gain_appliances REAL DEFAULT 0,
    warm_water_demand REAL DEFAULT 0,
    PRIMARY KEY(internal_gain_type,time_period)
);
CREATE TABLE IF NOT EXISTS internal_gain_timeseries (
    internal_gain_type TEXT,
    time TEXT,
    internal_gain_occupancy REAL DEFAULT 0,
    internal_gain_appliances REAL DEFAULT 0,
    warm_water_demand REAL DEFAULT 0,
    PRIMARY KEY(internal_gain_type,time)
);
CREATE TABLE IF NOT EXISTS internal_gain_types (
    internal_gain_type TEXT,
    internal_gain_definition_type TEXT,
    occupancy_density TEXT,
    occupancy_heat_gain TEXT,
    occupancy_co2_gain TEXT,
    occupancy_humidity_gain TEXT,
    appliances_heat_gain TEXT,
    warm_water_demand_thermal_power TEXT,
    PRIMARY KEY(internal_gain_type)
);
CREATE TABLE IF NOT EXISTS linearization_types (
    linearization_type TEXT,
    linearization_zone_air_temperature TEXT,
    linearization_zone_air_temperature_heat TEXT,
    linearization_zone_air_temperature_cool TEXT,
    linearization_surface_temperature TEXT,
    linearization_exterior_surface_temperature TEXT,
    linearization_ambient_air_temperature TEXT,
    linearization_sky_temperature TEXT,
    linearization_zone_air_absolute_humidity TEXT,
    linearization_ambient_air_absolute_humidity TEXT,
    linearization_zone_air_co2_concentration TEXT,
    linearization_zone_fresh_air_flow TEXT,
    PRIMARY KEY(linearization_type)
);
CREATE TABLE IF NOT EXISTS parameters (
    parameter_set TEXT,
    parameter_name TEXT,
    parameter_value REAL,
    parameter_unit TEXT,
    parameter_comment TEXT,
    PRIMARY KEY(parameter_set,parameter_name)
);
CREATE TABLE IF NOT EXISTS plant_cooling_types (
    plant_cooling_type TEXT,
    plant_cooling_efficiency TEXT,
    PRIMARY KEY(plant_cooling_type)
);
CREATE TABLE IF NOT EXISTS plant_heating_types (
    plant_heating_type TEXT,
    plant_heating_efficiency TEXT,
    PRIMARY KEY(plant_heating_type)
);
CREATE TABLE IF NOT EXISTS scenarios (
    scenario_name TEXT,
    building_name TEXT,
    parameter_set TEXT,
    linearization_type TEXT,
    initial_state_type TEXT,
    price_type TEXT,
    timestep_start TEXT,
    timestep_end TEXT,
    timestep_interval TEXT,
    PRIMARY KEY(scenario_name)
);
CREATE TABLE IF NOT EXISTS storage_types (
    storage_type TEXT,
    storage_commodity_type TEXT,
    storage_capacity TEXT,
    storage_round_trip_efficiency TEXT,
    storage_battery_depth_of_discharge TEXT,
    storage_sensible_temperature_delta TEXT,
    storage_self_discharge_rate TEXT,
    PRIMARY KEY(storage_type)
);
CREATE TABLE IF NOT EXISTS surface_types (
    surface_type TEXT,
    heat_capacity TEXT,
    heat_transfer_coefficient_conduction_surface TEXT,
    absorptivity_surface TEXT,
    emissivity_surface TEXT,
    window_type TEXT,
    window_wall_ratio TEXT,
    sky_view_factor TEXT,
    PRIMARY KEY(surface_type)
);
CREATE TABLE IF NOT EXISTS surfaces_adiabatic (
    building_name TEXT,
    zone_name TEXT,
    surface_name TEXT,
    surface_type TEXT,
    surface_area TEXT,
    surface_comment TEXT,
    PRIMARY KEY(building_name,surface_name)
    -- Note that the primary key constraint does not prevent duplicated `surface_name` in other surfaces tables.
);
CREATE TABLE IF NOT EXISTS surfaces_exterior (
    building_name TEXT,
    zone_name TEXT,
    direction_name TEXT,
    surface_name TEXT,
    surface_type TEXT,
    surface_area TEXT,
    surface_comment TEXT,
    PRIMARY KEY(building_name,surface_name)
    -- Note that the primary key constraint does not prevent duplicated `surface_name` in other surfaces tables.
);
CREATE TABLE IF NOT EXISTS surfaces_interior (
    building_name TEXT,
    zone_name TEXT,
    zone_adjacent_name TEXT,
    surface_name TEXT,
    surface_type TEXT,
    surface_area TEXT,
    surface_comment TEXT,
    PRIMARY KEY(building_name,surface_name)
    -- Note that the primary key constraint does not prevent duplicated `surface_name` in other surfaces tables.
);
CREATE TABLE IF NOT EXISTS window_types (
    window_type TEXT,
    heat_transfer_coefficient_conduction_window TEXT,
    absorptivity_window TEXT,
    emissivity_window TEXT,
    PRIMARY KEY(window_type)
);
CREATE TABLE IF NOT EXISTS weather_timeseries (
    weather_type TEXT,
    time TEXT,
    ambient_air_temperature REAL,
    sky_temperature REAL,
    ambient_air_absolute_humidity REAL,
    irradiation_horizontal REAL,
    irradiation_east REAL,
    irradiation_south REAL,
    irradiation_west REAL,
    irradiation_north REAL,
    PRIMARY KEY(weather_type,time)
);
CREATE TABLE IF NOT EXISTS weather_types (
    weather_type TEXT,
    time_zone TEXT,
    latitude REAL,
    longitude REAL,
    temperature_difference_sky_ambient REAL,
    PRIMARY KEY(weather_type)
);
CREATE TABLE IF NOT EXISTS zone_types (
    zone_type TEXT,
    heat_capacity TEXT,
    infiltration_rate TEXT,
    internal_gain_type TEXT,
    hvac_generic_type TEXT,
    hvac_radiator_type TEXT,
    hvac_ahu_type TEXT,
    hvac_tu_type TEXT,
    hvac_vent_type TEXT,
    constraint_type TEXT,
    fresh_air_flow_control_type TEXT,
    humidity_control_type TEXT,
    PRIMARY KEY(zone_type)
);
CREATE TABLE IF NOT EXISTS zones (
    building_name TEXT,
    zone_name TEXT,
    zone_type TEXT,
    zone_height TEXT,
    zone_area TEXT,
    zone_comment TEXT,
    PRIMARY KEY(building_name,zone_name)
);
