BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "blind_types" (
    "blind_type" TEXT,
    "blind_efficiency" TEXT
);
CREATE TABLE IF NOT EXISTS "buildings" (
    "building_name" TEXT,
    "weather_type" TEXT,
    "building_storage_type" TEXT
);
CREATE TABLE IF NOT EXISTS "constraint_schedules" (
    "constraint_type" TEXT,
    "time_period" TEXT,
    "minimum_air_temperature" TEXT,
    "maximum_air_temperature" TEXT,
    "minimum_fresh_air_flow_per_area" TEXT,
    "minimum_fresh_air_flow_per_person" TEXT,
    "maximum_co2_concentration" TEXT,
    "minimum_fresh_air_flow_per_area_no_dcv" TEXT,
    "minimum_relative_humidity" TEXT,
    "maximum_relative_humidity" TEXT
);
CREATE TABLE IF NOT EXISTS "electricity_price_timeseries" (
    "price_type" TEXT,
    "time" TEXT,
    "price" REAL
);
CREATE TABLE IF NOT EXISTS "electricity_price_range" (
    "time_period" TEXT,
    "price_mean" REAL,
    "delta_upper" REAL,
    "delta_lower" REAL
);
CREATE TABLE IF NOT EXISTS "hvac_ahu_types" (
    "hvac_ahu_type" TEXT,
    "ahu_supply_air_temperature_setpoint" TEXT,
    "ahu_supply_air_relative_humidity_setpoint" TEXT,
    "ahu_fan_efficiency" TEXT,
    "ahu_cooling_efficiency" TEXT,
    "ahu_heating_efficiency" TEXT,
    "ahu_return_air_heat_recovery_efficiency" TEXT
);
CREATE TABLE IF NOT EXISTS "hvac_generic_types" (
    "hvac_generic_type" TEXT,
    "generic_heating_efficiency" TEXT,
    "generic_cooling_efficiency" TEXT
);
CREATE TABLE IF NOT EXISTS "hvac_radiator_types" (
    "hvac_radiator_type" TEXT,
    "radiator_supply_temperature_nominal" TEXT,
    "radiator_return_temperature_nominal" TEXT,
    "radiator_panel_number" TEXT,
    "radiator_panel_area" TEXT,
    "radiator_panel_thickness" TEXT,
    "radiator_water_volume" TEXT,
    "radiator_convection_coefficient" TEXT,
    "radiator_emissivity" TEXT,
    "radiator_hull_conductivity" TEXT,
    "radiator_hull_heat_capacity" TEXT,
    "radiator_fin_effectiveness" TEXT
);
CREATE TABLE IF NOT EXISTS "hvac_tu_types" (
    "hvac_tu_type" TEXT,
    "tu_air_intake_type" TEXT,
    "tu_supply_air_temperature_setpoint" TEXT,
    "tu_fan_efficiency" TEXT,
    "tu_cooling_efficiency" TEXT,
    "tu_heating_efficiency" TEXT
);
CREATE TABLE IF NOT EXISTS "initial_state_types" (
    "initial_state_type" TEXT,
    "initial_zone_temperature" TEXT,
    "initial_surface_temperature" TEXT,
    "initial_co2_concentration" TEXT,
    "initial_absolute_humidity" TEXT,
    "initial_sensible_thermal_storage_state_of_charge" TEXT,
    "initial_battery_storage_state_of_charge" TEXT
);
CREATE TABLE IF NOT EXISTS "internal_gain_schedules" (
    "internal_gain_type" TEXT,
    "time_period" TEXT,
    "internal_gain_occupancy" REAL,
    "internal_gain_appliances" REAL
);
CREATE TABLE IF NOT EXISTS "internal_gain_timeseries" (
    "internal_gain_type" TEXT,
    "time" TEXT,
    "internal_gain_occupancy" REAL,
    "internal_gain_appliances" REAL
);
CREATE TABLE IF NOT EXISTS "internal_gain_types" (
    "internal_gain_type" TEXT,
    "internal_gain_definition_type" TEXT,
    "internal_gain_occupancy_factor" TEXT,
    "internal_gain_appliances_factor" TEXT
);
CREATE TABLE IF NOT EXISTS "linearization_types" (
    "linearization_type" TEXT,
    "linearization_zone_air_temperature_heat" TEXT,
    "linearization_zone_air_temperature_cool" TEXT,
    "linearization_surface_temperature" TEXT,
    "linearization_exterior_surface_temperature" TEXT,
    "linearization_internal_gain_occupancy" TEXT,
    "linearization_internal_gain_appliances" TEXT,
    "linearization_ambient_air_temperature" TEXT,
    "linearization_sky_temperature" TEXT,
    "linearization_ambient_air_humidity_ratio" TEXT,
    "linearization_zone_air_humidity_ratio" TEXT,
    "linearization_irradiation" TEXT,
    "linearization_co2_concentration" TEXT,
    "linearization_ventilation_rate_per_square_meter" TEXT
);
CREATE TABLE IF NOT EXISTS "parameters" (
    "parameter_set" TEXT,
    "parameter_name" TEXT,
    "parameter_value" REAL,
    "parameter_unit" TEXT,
    "parameter_comment" TEXT
);
CREATE TABLE IF NOT EXISTS "scenarios" (
    "scenario_name" TEXT,
    "building_name" TEXT,
    "parameter_set" TEXT,
    "linearization_type" TEXT,
    "initial_state_type" TEXT,
    "demand_controlled_ventilation_type" TEXT,
    "co2_model_type" TEXT,
    "humidity_model_type" TEXT,
    "heating_cooling_session" TEXT,
    "price_type" TEXT,
    "time_start" TEXT,
    "time_end" TEXT,
    "time_step" TEXT
);
CREATE TABLE IF NOT EXISTS "storage_types" (
    "building_storage_type" TEXT,
    "storage_size" REAL,
    "storage_round_trip_efficiency" REAL,
    "storage_battery_depth_of_discharge"  REAL,
    "storage_sensible_temperature_delta" REAL,
    "storage_lifetime"  REAL,
    "storage_planning_energy_installation_cost"   REAL,
    "storage_planning_power_installation_cost"  REAL,
    "storage_planning_fixed_installation_cost"  REAL
);
CREATE TABLE IF NOT EXISTS "surface_types" (
    "surface_type" TEXT,
    "heat_capacity" TEXT,
    "thermal_resistance_surface" TEXT,
    "absorptivity" TEXT,
    "emissivity" TEXT,
    "window_type" TEXT,
    "window_wall_ratio" TEXT,
    "sky_view_factor" TEXT
);
CREATE TABLE IF NOT EXISTS "surfaces_adiabatic" (
    "building_name" TEXT,
    "zone_name" TEXT,
    "surface_name" TEXT,
    "surface_type" TEXT,
    "surface_area" TEXT,
    "surface_comment" TEXT
);
CREATE TABLE IF NOT EXISTS "surfaces_exterior" (
    "building_name" TEXT,
    "zone_name" TEXT,
    "direction_name" TEXT,
    "surface_name" TEXT,
    "surface_type" TEXT,
    "surface_area" TEXT,
    "surface_comment" TEXT
);
CREATE TABLE IF NOT EXISTS "surfaces_interior" (
    "building_name" TEXT,
    "zone_name" TEXT,
    "zone_adjacent_name" TEXT,
    "surface_name" TEXT,
    "surface_type" TEXT,
    "surface_area" TEXT,
    "surface_comment" TEXT
);
CREATE TABLE IF NOT EXISTS "window_types" (
    "window_type" TEXT,
    "thermal_resistance_window" TEXT,
    "absorptivity_window" TEXT,
    "emissivity_window" TEXT
);
CREATE TABLE IF NOT EXISTS "weather_timeseries" (
    "weather_type" TEXT,
    "time" TEXT,
    "ambient_air_temperature" REAL,
    "sky_temperature" REAL,
    "ambient_air_humidity_ratio" REAL,
    "irradiation_horizontal" REAL,
    "irradiation_east" REAL,
    "irradiation_south" REAL,
    "irradiation_west" REAL,
    "irradiation_north" REAL
);
CREATE TABLE IF NOT EXISTS "weather_types" (
    "weather_type" TEXT,
    "time_zone" TEXT,
    "latitude" REAL,
    "longitude" REAL,
    "temperature_difference_sky_ambient" REAL
);
CREATE TABLE IF NOT EXISTS "zone_types" (
    "zone_type" TEXT,
    "heat_capacity" TEXT,
    "infiltration_rate" TEXT,
    "internal_gain_type" TEXT,
    "window_type" TEXT,
    "blind_type" TEXT,
    "hvac_generic_type" TEXT,
    "hvac_radiator_type" TEXT,
    "hvac_ahu_type" TEXT,
    "hvac_tu_type" TEXT,
    "constraint_type" TEXT
);
CREATE TABLE IF NOT EXISTS "zones" (
    "building_name" TEXT,
    "zone_name" TEXT,
    "zone_type" TEXT,
    "zone_height" TEXT,
    "zone_area" TEXT,
    "zone_comment" TEXT
);
COMMIT;
