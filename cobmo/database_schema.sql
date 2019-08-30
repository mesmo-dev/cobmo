BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "battery_storage_types" (
	"battery_technology"	TEXT,
	"round_trip_efficiency"	REAL,
	"depth_of_discharge"	REAL,
	"energy_installation_cost_2016"	REAL,
	"energy_installation_cost_2020"	REAL,
	"energy_installation_cost_2025"	REAL,
	"energy_installation_cost_2030"	REAL,
	"power_installation_cost_2016"	REAL,
	"power_installation_cost_2020"	REAL,
	"power_installation_cost_2025"	REAL,
	"power_installation_cost_2030"	REAL,
	"lifetime_reference_2016"	REAL,
	"lifetime_reference_2020"	REAL,
	"lifetime_reference_2025"	REAL,
	"lifetime_reference_2030"	REAL
);
CREATE TABLE IF NOT EXISTS "building_blind_types" (
	"blind_type"	TEXT,
	"blind_efficiency"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_hvac_ahu_types" (
	"hvac_ahu_type"	TEXT,
	"ahu_cooling_type"	TEXT,
	"ahu_heating_type"	TEXT,
	"ahu_dehumidification_type"	TEXT,
	"ahu_return_air_heat_recovery_type"	TEXT,
	"ahu_supply_air_temperature_setpoint"	TEXT,
	"ahu_supply_air_relative_humidity_setpoint"	TEXT,
	"ahu_fan_efficiency"	TEXT,
	"ahu_cooling_efficiency"	TEXT,
	"ahu_heating_efficiency"	TEXT,
	"ahu_return_air_heat_recovery_efficiency"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_hvac_generic_types" (
	"hvac_generic_type"	TEXT,
	"generic_heating_efficiency"	TEXT,
	"generic_cooling_efficiency"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_hvac_tu_types" (
	"hvac_tu_type"	TEXT,
	"tu_cooling_type"	TEXT,
	"tu_heating_type"	TEXT,
	"tu_air_intake_type"	TEXT,
	"tu_supply_air_temperature_setpoint"	TEXT,
	"tu_fan_efficiency"	TEXT,
	"tu_cooling_efficiency"	TEXT,
	"tu_heating_efficiency"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_internal_gain_timeseries" (
	"internal_gain_type"	TEXT,
	"time"	TEXT,
	"internal_gain_occupancy"	REAL,
	"internal_gain_appliances"	REAL
);
CREATE TABLE IF NOT EXISTS "building_internal_gain_types" (
	"internal_gain_type"	TEXT,
	"internal_gain_occupancy_factor"	TEXT,
	"internal_gain_appliances_factor"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_linearization_types" (
	"linearization_type"	TEXT,
	"linearization_zone_air_temperature_heat"	TEXT,
	"linearization_zone_air_temperature_cool"	TEXT,
	"linearization_surface_temperature"	TEXT,
	"linearization_exterior_surface_temperature"	TEXT,
	"linearization_internal_gain_occupancy"	TEXT,
	"linearization_internal_gain_appliances"	TEXT,
	"linearization_ambient_air_temperature"	TEXT,
	"linearization_sky_temperature"	TEXT,
	"linearization_ambient_air_humidity_ratio"	TEXT,
	"linearization_zone_air_humidity_ratio"	TEXT,
	"linearization_irradiation"	TEXT,
	"linearization_co2_concentration"	TEXT,
	"linearization_ventilation_rate_per_square_meter"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_parameter_sets" (
	"parameter_set"	TEXT,
	"parameter_name"	TEXT,
	"parameter_value"	REAL,
	"parameter_unit"	TEXT,
	"parameter_comment"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_scenarios" (
	"scenario_name"	TEXT,
	"building_name"	TEXT,
	"parameter_set"	TEXT,
	"linearization_type"	TEXT,
	"demand_controlled_ventilation_type"	TEXT,
	"co2_model_type"	TEXT,
	"humidity_model_type"	TEXT,
	"heating_cooling_session"	TEXT,
	"time_start"	TEXT,
	"time_end"	TEXT,
	"time_step"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_storage_types" (
	"building_storage_type"	TEXT,
	"storage_initial_state_of_charge"	REAL,
	"storage_UA_external"	REAL,
	"storage_UA_thermocline"	REAL,
	"storage_cooling_temperature_bottom_layer"	REAL,
	"storage_cooling_temperature_top_layer"	REAL,
	"storage_cooling_ambient_temperature"	REAL,
	"storage_sensible_total_delta_temperature_layers"	REAL,
	"storage_round_trip_efficiency"	REAL,
	"storage_size"	REAL,
	"tank_aspect_ratio"  REAL,
    "heat_transfer_coefficient"  REAL,
    "storage_investment_sgd_per_unit"   REAL,
    "investment_sgd_per_X"  TEXT,
    "storage_power_installation_cost"  REAL,
    "storage_fixed_cost"  REAL,
    "storage_depth_of_discharge"  REAL,
    "storage_lifetime"  REAL,
    "peak_electric_power_building_watt"  REAL,
	"unit_heat_transfer_coefficient"  TEXT,
	"storage_size_unit"	TEXT,
	"comments_storage_type_general"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_surface_types" (
	"surface_type"	TEXT,
	"heat_capacity"	TEXT,
	"thermal_resistance_surface"	TEXT,
	"irradiation_gain_coefficient"	TEXT,
	"emissivity"	TEXT,
	"window_type"	TEXT,
	"window_wall_ratio"	TEXT,
	"sky_view_factor"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_surfaces_adiabatic" (
	"building_name"	TEXT,
	"zone_name"	TEXT,
	"surface_name"	TEXT,
	"surface_type"	TEXT,
	"surface_area"	TEXT,
	"surface_comment"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_surfaces_exterior" (
	"building_name"	TEXT,
	"zone_name"	TEXT,
	"direction_name"	TEXT,
	"surface_name"	TEXT,
	"surface_type"	TEXT,
	"surface_area"	TEXT,
	"surface_comment"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_surfaces_interior" (
	"building_name"	TEXT,
	"zone_name"	TEXT,
	"zone_adjacent_name"	TEXT,
	"surface_name"	TEXT,
	"surface_type"	TEXT,
	"surface_area"	TEXT,
	"surface_comment"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_window_types" (
	"window_type"	TEXT,
	"thermal_resistance_window"	TEXT,
	"irradiation_gain_coefficient_window"	TEXT,
	"emissivity_window"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_zone_constraint_profiles" (
	"zone_constraint_profile"	TEXT,
	"from_weekday"	INTEGER,
	"from_time"	TEXT,
	"minimum_air_temperature"	TEXT,
	"maximum_air_temperature"	TEXT,
	"minimum_fresh_air_flow_per_area"	TEXT,
	"minimum_fresh_air_flow_per_person"	TEXT,
	"maximum_co2_concentration"	TEXT,
	"minimum_fresh_air_flow_per_area_no_dcv"	TEXT,
	"minimum_relative_humidity"	TEXT,
	"maximum_relative_humidity"	TEXT

);
CREATE TABLE IF NOT EXISTS "building_zone_types" (
	"zone_type"	TEXT,
	"heat_capacity"	TEXT,
	"base_surface_type"	TEXT,
	"ceiling_surface_type"	TEXT,
	"infiltration_rate"	TEXT,
	"internal_gain_type"	TEXT,
	"window_type"	TEXT,
	"blind_type"	TEXT,
	"hvac_generic_type"	TEXT,
	"hvac_ahu_type"	TEXT,
	"hvac_tu_type"	TEXT,
	"zone_constraint_profile"	TEXT
);
CREATE TABLE IF NOT EXISTS "building_zones" (
	"building_name"	TEXT,
	"zone_name"	TEXT,
	"zone_type"	TEXT,
	"zone_height"	TEXT,
	"zone_area"	TEXT,
	"zone_comment"	TEXT
);
CREATE TABLE IF NOT EXISTS "buildings" (
	"building_name"	TEXT,
	"weather_type"	TEXT,
	"building_storage_type"	TEXT
);
CREATE TABLE IF NOT EXISTS "weather_timeseries" (
	"weather_type"	TEXT,
	"time"	TEXT,
	"ambient_air_temperature"	REAL,
	"sky_temperature"	REAL,
	"ambient_air_humidity_ratio"	REAL,
	"irradiation_horizontal"	REAL,
	"irradiation_east"	REAL,
	"irradiation_south"	REAL,
	"irradiation_west"	REAL,
	"irradiation_north"	REAL,
	"storage_ambient_air_temperature"	REAL
);
CREATE TABLE IF NOT EXISTS "electricity_price_timeseries" (
	"scenario_name"  TEXT,
	"price_type"	TEXT,
	"time"	TEXT,
	"price"	REAL,
	"price_unit"	TEXT
);
CREATE TABLE IF NOT EXISTS "weather_types" (
	"weather_type"	TEXT,
	"time_zone"	TEXT,
	"latitude"	REAL,
	"longitude"	REAL,
	"temperature_difference_sky_ambient"	REAL
);
COMMIT;
