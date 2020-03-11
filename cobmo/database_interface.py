"""Database interface function definitions."""

import glob
from multimethod import multimethod
import numpy as np
import os
import pandas as pd
import sqlite3

import cobmo.config

logger = cobmo.config.get_logger(__name__)


def recreate_database(
        database_path: str = cobmo.config.database_path,
        database_schema_path: str = os.path.join(cobmo.config.cobmo_path, 'cobmo', 'database_schema.sql'),
        csv_path: str = cobmo.config.data_path
) -> None:
    """Recreate SQLITE database from SQL schema file and CSV files."""

    # Connect SQLITE database (creates file, if none).
    database_connection = sqlite3.connect(database_path)
    cursor = database_connection.cursor()

    # Remove old data, if any.
    cursor.executescript(
        """ 
        PRAGMA writable_schema = 1; 
        DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger'); 
        PRAGMA writable_schema = 0; 
        VACUUM; 
        """
    )

    # Recreate SQLITE database (schema) from SQL file.
    with open(database_schema_path, 'r') as database_schema_file:
        cursor.executescript(database_schema_file.read())
    database_connection.commit()

    # Import CSV files into SQLITE database.
    database_connection.text_factory = str  # Allows utf-8 data to be stored.
    cursor = database_connection.cursor()
    for file in glob.glob(os.path.join(csv_path, '*.csv')):
        # Obtain table name.
        table_name = os.path.splitext(os.path.basename(file))[0]

        # Delete existing table content.
        cursor.execute("DELETE FROM {}".format(table_name))
        database_connection.commit()

        # Write new table content.
        logger.debug(f"Loading {file} into database.")
        table = pd.read_csv(file)
        table.to_sql(
            table_name,
            con=database_connection,
            if_exists='append',
            index=False
        )
    cursor.close()
    database_connection.close()


def connect_database(
        database_path: str = cobmo.config.database_path
) -> sqlite3.Connection:
    """Connect to the database at given `data_path` and return connection handle."""

    # Recreate database, if no database exists.
    if not os.path.isfile(database_path):
        logger.debug(f"Database does not exist and is recreated at: {database_path}")
        recreate_database(
            database_path=database_path
        )

    # Obtain connection.
    database_connection = sqlite3.connect(database_path)
    return database_connection


class BuildingData(object):
    """Building data object."""

    scenarios: pd.Series
    parameters: pd.Series
    surfaces_adiabatic: pd.DataFrame
    surfaces_exterior: pd.DataFrame
    surfaces_interior: pd.DataFrame
    zones: pd.DataFrame
    zone_constraint_profiles_dict: dict
    timestep_start: pd.Timestamp
    timestep_end: pd.Timestamp
    timestep_delta: pd.Timedelta
    set_timesteps: pd.Index
    weather_timeseries: pd.DataFrame
    internal_gain_timeseries: pd.DataFrame
    electricity_price_timeseries: pd.DataFrame

    @ multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ) -> None:

        # Obtain database connection.
        database_connection = connect_database()

        self.__init__(
            scenario_name,
            database_connection,
            **kwargs
        )

    @ multimethod
    def __init__(
            self,
            scenario_name: str,
            database_connection: sqlite3.Connection,
            timestep_start=None,
            timestep_end=None,
            timestep_delta=None
    ) -> None:

        # Store scenario name.
        self.scenario_name = scenario_name

        # Obtain building data.
        self.scenarios = (
            pd.read_sql(
                """
                SELECT * FROM scenarios 
                JOIN buildings USING (building_name) 
                JOIN linearization_types USING (linearization_type) 
                LEFT JOIN initial_state_types USING (initial_state_type) 
                LEFT JOIN storage_types USING (building_storage_type) 
                WHERE scenario_name = ?
                """,
                con=database_connection,
                params=[self.scenario_name]
            ).iloc[0]  # Convert to Series for shorter indexing.
        )
        self.parameters = (
            pd.read_sql(
                """
                SELECT parameter_name, parameter_value FROM parameters 
                WHERE parameter_set IN ('constants', ?)
                """,
                con=database_connection,
                params=[self.scenarios['parameter_set']],
                index_col='parameter_name'
            ).iloc[:, 0]  # Convert to Series for shorter indexing.
        )
        self.surfaces_adiabatic = (
            pd.read_sql(
                """
                SELECT * FROM surfaces_adiabatic 
                JOIN surface_types USING (surface_type) 
                LEFT JOIN window_types USING (window_type) 
                JOIN zones USING (zone_name, building_name) 
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[scenario_name]
            )
        )
        self.surfaces_adiabatic.index = self.surfaces_adiabatic['surface_name']
        self.surfaces_exterior = (
            pd.read_sql(
                """
                SELECT * FROM surfaces_exterior 
                JOIN surface_types USING (surface_type) 
                LEFT JOIN window_types USING (window_type) 
                JOIN zones USING (zone_name, building_name) 
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[self.scenario_name]
            )
        )
        self.surfaces_exterior.index = self.surfaces_exterior['surface_name']
        self.surfaces_interior = (
            pd.read_sql(
                """
                SELECT * FROM surfaces_interior 
                JOIN surface_types USING (surface_type) 
                LEFT JOIN window_types USING (window_type) 
                JOIN zones USING (zone_name, building_name) 
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[self.scenario_name]
            )
        )
        self.surfaces_interior.index = self.surfaces_interior['surface_name']
        self.zones = (
            pd.read_sql(
                """
                SELECT * FROM zones 
                JOIN zone_types USING (zone_type) 
                JOIN internal_gain_types USING (internal_gain_type) 
                LEFT JOIN blind_types USING (blind_type) 
                LEFT JOIN hvac_generic_types USING (hvac_generic_type) 
                LEFT JOIN hvac_radiator_types USING (hvac_radiator_type) 
                LEFT JOIN hvac_ahu_types USING (hvac_ahu_type) 
                LEFT JOIN hvac_tu_types USING (hvac_tu_type) 
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
                """,
                con=database_connection,
                params=[self.scenario_name]
            )
        )
        self.zones.index = self.zones['zone_name']
        self.zone_constraint_profiles_dict = (
            dict.fromkeys(self.zones['zone_constraint_profile'].unique())
        )
        for zone_constraint_profile in self.zone_constraint_profiles_dict:
            self.zone_constraint_profiles_dict[zone_constraint_profile] = (
                pd.read_sql(
                    """
                    SELECT * FROM zone_constraint_profiles 
                    WHERE zone_constraint_profile = ?
                    """,
                    con=database_connection,
                    params=[zone_constraint_profile]
                )
            )

        # Define parameter parsing utility function.
        @np.vectorize
        def parse_parameter(parameter_string):
            """Parse parameter utility function.
            - Convert given parameter string as `np.float`.
            - If the parameter string is the name of a parameter from `parameters`, return that parameter.
            - If the parameter string is `None`, `NaN` is returned.
            """
            try:
                return np.float(parameter_string)
            except ValueError:
                return self.parameters[parameter_string]
            except TypeError:
                return np.nan

        # Parse parameters.
        scenarios_numerical_columns = [
            'linearization_zone_air_temperature_heat',
            'linearization_zone_air_temperature_cool',
            'linearization_surface_temperature',
            'linearization_exterior_surface_temperature',
            'linearization_internal_gain_occupancy',
            'linearization_internal_gain_appliances',
            'linearization_ambient_air_temperature',
            'linearization_sky_temperature',
            'linearization_ambient_air_humidity_ratio',
            'linearization_zone_air_humidity_ratio',
            'linearization_irradiation',
            'linearization_co2_concentration',
            'linearization_ventilation_rate_per_square_meter',
            'initial_zone_temperature',
            'initial_surface_temperature',
            'initial_co2_concentration',
            'initial_absolute_humidity',
            'initial_sensible_thermal_storage_state_of_charge',
            'initial_battery_storage_state_of_charge',
            'storage_size',
            'storage_round_trip_efficiency',
            'storage_battery_depth_of_discharge',
            'storage_sensible_temperature_delta',
            'storage_lifetime',
            'storage_planning_energy_installation_cost',
            'storage_planning_power_installation_cost',
            'storage_planning_fixed_installation_cost'
        ]
        self.scenarios.loc[scenarios_numerical_columns] = (
            self.scenarios.loc[scenarios_numerical_columns].apply(parse_parameter)
        )
        building_surfaces_numerical_columns = [
            'surface_area',
            'heat_capacity',
            'thermal_resistance_surface',
            'absorptivity',
            'emissivity',
            'window_wall_ratio',
            'sky_view_factor',
            'thermal_resistance_window',
            'absorptivity_window',
            'emissivity_window',
            'zone_height',
            'zone_area'
        ]
        self.surfaces_adiabatic.loc[:, building_surfaces_numerical_columns] = (
            self.surfaces_adiabatic.loc[:, building_surfaces_numerical_columns].apply(parse_parameter)
        )
        self.surfaces_exterior.loc[:, building_surfaces_numerical_columns] = (
            self.surfaces_exterior.loc[:, building_surfaces_numerical_columns].apply(parse_parameter)
        )
        self.surfaces_interior.loc[:, building_surfaces_numerical_columns] = (
            self.surfaces_interior.loc[:, building_surfaces_numerical_columns].apply(parse_parameter)
        )
        zones_numerical_columns = [
            'zone_area',
            'zone_height',
            'heat_capacity',
            'infiltration_rate',
            'internal_gain_occupancy_factor',
            'internal_gain_appliances_factor',
            'blind_efficiency',
            'generic_heating_efficiency',
            'generic_cooling_efficiency',
            'radiator_supply_temperature_nominal',
            'radiator_return_temperature_nominal',
            'radiator_panel_number',
            'radiator_panel_area',
            'radiator_panel_thickness',
            'radiator_water_volume',
            'radiator_convection_coefficient',
            'radiator_emissivity',
            'radiator_hull_conductivity',
            'radiator_hull_heat_capacity',
            'radiator_fin_effectiveness',
            'ahu_supply_air_temperature_setpoint',
            'ahu_supply_air_relative_humidity_setpoint',
            'ahu_fan_efficiency',
            'ahu_cooling_efficiency',
            'ahu_heating_efficiency',
            'ahu_return_air_heat_recovery_efficiency',
            'tu_supply_air_temperature_setpoint',
            'tu_fan_efficiency',
            'tu_cooling_efficiency',
            'tu_heating_efficiency'
        ]
        self.zones.loc[:, zones_numerical_columns] = (
            self.zones.loc[:, zones_numerical_columns].apply(parse_parameter)
        )
        zone_constraint_profiles_numerical_columns = [
            'minimum_air_temperature',
            'maximum_air_temperature',
            'minimum_fresh_air_flow_per_area',
            'minimum_fresh_air_flow_per_person',
            'maximum_co2_concentration',
            'minimum_fresh_air_flow_per_area_no_dcv',
            'minimum_relative_humidity',
            'maximum_relative_humidity'
        ]
        for zone_constraint_profile in self.zone_constraint_profiles_dict:
            self.zone_constraint_profiles_dict[zone_constraint_profile].loc[
                :, zone_constraint_profiles_numerical_columns
            ] = (
                self.zone_constraint_profiles_dict[zone_constraint_profile].loc[
                    :, zone_constraint_profiles_numerical_columns
                ].apply(parse_parameter)
            )

        # Obtain timestep data.
        if timestep_start is not None:
            self.timestep_start = pd.Timestamp(timestep_start)
        else:
            self.timestep_start = pd.Timestamp(self.scenarios['time_start'])
        if timestep_end is not None:
            self.timestep_end = pd.Timestamp(timestep_end)
        else:
            self.timestep_end = pd.Timestamp(self.scenarios['time_end'])
        if timestep_delta is not None:
            self.timestep_delta = pd.Timedelta(timestep_delta)
        else:
            self.timestep_delta = pd.Timedelta(self.scenarios['time_step'])
        self.set_timesteps = pd.Index(
            pd.date_range(
                start=self.timestep_start,
                end=self.timestep_end,
                freq=self.timestep_delta
            ),
            name='time'
        )

        # Obtain timeseries data.
        timestep_start_string = self.timestep_start.strftime('%Y-%m-%dT%H:%M:%S')  # Shorthand for SQL commands.
        timestep_end_string = self.timestep_end.strftime('%Y-%m-%dT%H:%M:%S')  # Shorthand for SQL commands.
        self.weather_timeseries = (
            pd.read_sql(
                """
                SELECT * FROM weather_timeseries 
                WHERE weather_type = (
                    SELECT weather_type from scenarios 
                    JOIN buildings USING (building_name)
                    WHERE scenario_name = ?
                )
                AND time between ? AND ?
                """,
                con=database_connection,
                params=[self.scenario_name, timestep_start_string, timestep_end_string],
                parse_dates=['time']
            )
        )
        self.weather_timeseries.index = self.weather_timeseries['time']
        self.internal_gain_timeseries = pd.read_sql(
            """
            SELECT * FROM internal_gain_timeseries 
            WHERE internal_gain_type IN (
                SELECT DISTINCT internal_gain_type FROM zones
                JOIN zone_types USING (zone_type) 
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
            )
            AND time between ? AND ?
            """,
            con=database_connection,
            params=[self.scenario_name, timestep_start_string, timestep_end_string],
            parse_dates=['time']
        )
        self.electricity_price_timeseries = (
            pd.read_sql(
                """
                SELECT * FROM electricity_price_timeseries 
                WHERE price_type = (
                    SELECT price_type from scenarios 
                    WHERE scenario_name = ?
                )
                AND time between ? AND ?
                """,
                con=database_connection,
                params=[self.scenario_name, timestep_start_string, timestep_end_string],
                parse_dates=['time']
            )
        )
        self.electricity_price_timeseries.index = self.electricity_price_timeseries['time']

        # Pivot internal gain timeseries, to get one `_occupancy` / `_appliances` for each `internal_gain_type`.
        building_internal_gain_occupancy_timeseries = self.internal_gain_timeseries.pivot(
            index='time',
            columns='internal_gain_type',
            values='internal_gain_occupancy'
        )
        building_internal_gain_occupancy_timeseries.columns = (
            building_internal_gain_occupancy_timeseries.columns + '_occupancy'
        )
        building_internal_gain_appliances_timeseries = self.internal_gain_timeseries.pivot(
            index='time',
            columns='internal_gain_type',
            values='internal_gain_appliances'
        )
        building_internal_gain_appliances_timeseries.columns = (
            building_internal_gain_appliances_timeseries.columns + '_appliances'
        )
        self.internal_gain_timeseries = pd.concat(
            [
                building_internal_gain_occupancy_timeseries,
                building_internal_gain_appliances_timeseries
            ],
            axis='columns'
        )
        self.internal_gain_timeseries.index = pd.to_datetime(self.internal_gain_timeseries.index)

        # Reindex / interpolate timeseries.
        self.weather_timeseries = (
            self.weather_timeseries.reindex(
                self.set_timesteps
            ).interpolate(
                'quadratic'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            )
        )
        self.internal_gain_timeseries = (
            self.internal_gain_timeseries.reindex(
                self.set_timesteps
            ).interpolate(
                'quadratic'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            )
        )
        self.electricity_price_timeseries = (
            self.electricity_price_timeseries.reindex(
                self.set_timesteps
            ).interpolate(
                'quadratic'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            )
        )
