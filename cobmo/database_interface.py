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
    timestep_start: pd.Timestamp
    timestep_end: pd.Timestamp
    timestep_delta: pd.Timedelta
    timesteps: pd.Index
    weather_timeseries: pd.DataFrame
    electricity_price_timeseries: pd.DataFrame
    internal_gain_timeseries: pd.DataFrame
    constraint_timeseries: pd.DataFrame

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
        self.timesteps = pd.Index(
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

        # Obtain weather timeseries.
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
        self.weather_timeseries = (
            self.weather_timeseries.reindex(
                self.timesteps
            ).interpolate(
                'quadratic'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            )
        )

        # Obtain electricity price timeseries.
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
        self.electricity_price_timeseries = (
            self.electricity_price_timeseries.reindex(
                self.timesteps
            ).interpolate(
                'quadratic'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
            )
        )

        # Obtain internal gain timeseries based on schedules.
        internal_gain_schedule = pd.read_sql(
            """
            SELECT * FROM internal_gain_schedule 
            WHERE internal_gain_type IN (
                SELECT DISTINCT internal_gain_type FROM zones
                JOIN zone_types USING (zone_type) 
                JOIN internal_gain_types USING (internal_gain_type)
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
                AND internal_gain_definition_type = 'schedule'
            )
            """,
            con=database_connection,
            params=[self.scenario_name],
            index_col='time_period'
        )
        if len(internal_gain_schedule) > 0:

            # Parse time period index.
            internal_gain_schedule.index = np.vectorize(pd.Period)(internal_gain_schedule.index)

            # Obtain complete schedule for all weekdays.
            # TODO: Check if '01T00:00:00' is defined for each schedule.
            internal_gain_schedule_complete = []
            for internal_gain_type in internal_gain_schedule['internal_gain_type'].unique():
                for day in range(1, 8):
                    if day in internal_gain_schedule.index.day.unique():
                        internal_gain_schedule_complete.append(
                            internal_gain_schedule.loc[(
                                (internal_gain_schedule.index.day == day)
                                & (internal_gain_schedule['internal_gain_type'] == internal_gain_type)
                            ), :]
                        )
                    else:
                        internal_gain_schedule_previous = internal_gain_schedule_complete[-1].copy()
                        internal_gain_schedule_previous.index += pd.Timedelta('1 day')
                        internal_gain_schedule_complete.append(internal_gain_schedule_previous)
            internal_gain_schedule_complete = pd.concat(internal_gain_schedule_complete)

            # Pivot complete schedule.
            # TODO: Multiply internal gain factors.
            internal_gain_schedule_complete = internal_gain_schedule_complete.pivot(
                columns='internal_gain_type',
                values=['internal_gain_occupancy', 'internal_gain_appliances']
            )
            internal_gain_schedule_complete.columns = (
                internal_gain_schedule_complete.columns.map(lambda x: '_'.join(x[::-1]))
            )

            # Obtain complete schedule for each minute of the week.
            internal_gain_schedule_complete = (
                internal_gain_schedule_complete.reindex(
                    pd.period_range(start='01T00:00', end='07T23:59', freq='T')
                ).fillna(method='ffill')
            )

            # Reindex / fill internal gain schedule for given timesteps.
            internal_gain_schedule_complete.index = (
                pd.MultiIndex.from_arrays([
                    internal_gain_schedule_complete.index.day - 1,
                    internal_gain_schedule_complete.index.hour,
                    internal_gain_schedule_complete.index.minute
                ])
            )
            internal_gain_schedule = (
                pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([
                        self.timesteps.weekday,
                        self.timesteps.hour,
                        self.timesteps.minute
                    ]),
                    columns=internal_gain_schedule_complete.columns
                )
            )
            for column in internal_gain_schedule.columns:
                 internal_gain_schedule[column] = (
                     internal_gain_schedule[column].fillna(internal_gain_schedule_complete[column])
                 )
            internal_gain_schedule.index = self.timesteps

        else:
            internal_gain_schedule = None

        # Obtain internal gain timeseries based on timeseries.
        internal_gain_timeseries = pd.read_sql(
            """
            SELECT * FROM internal_gain_timeseries 
            WHERE internal_gain_type IN (
                SELECT DISTINCT internal_gain_type FROM zones
                JOIN zone_types USING (zone_type) 
                JOIN internal_gain_types USING (internal_gain_type)
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
                AND internal_gain_definition_type = 'timeseries'
            )
            AND time between ? AND ?
            """,
            con=database_connection,
            params=[self.scenario_name, timestep_start_string, timestep_end_string],
            index_col='time',
            parse_dates=['time']
        )
        if len(internal_gain_timeseries) > 0:

            # Pivot timeseries.
            # TODO: Multiply internal gain factors.
            internal_gain_timeseries = internal_gain_timeseries.pivot(
                columns='internal_gain_type',
                values=['internal_gain_occupancy', 'internal_gain_appliances']
            )
            internal_gain_timeseries.columns = (
                internal_gain_timeseries.columns.map(lambda x: '_'.join(x[::-1]))
            )

            # Reindex / interpolate timeseries for given timesteps.
            internal_gain_timeseries = (
                internal_gain_timeseries.reindex(
                    self.timesteps
                ).interpolate(
                    'quadratic'
                ).bfill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
                ).ffill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_delta))
                )
            )

        else:
            internal_gain_timeseries = None

        # Merge schedule-based and timeseries-based internal gain timeseries.
        self.internal_gain_timeseries = (
            pd.concat(
                [
                    internal_gain_schedule,
                    internal_gain_timeseries
                ],
                axis='columns'
            )
        )

        # Obtain constraint timeseries based on schedules.
        constraint_schedule = pd.read_sql(
            """
            SELECT * FROM constraint_schedules 
            WHERE constraint_type IN (
                SELECT DISTINCT constraint_type FROM zones
                JOIN zone_types USING (zone_type) 
                WHERE building_name = (
                    SELECT building_name from scenarios 
                    WHERE scenario_name = ?
                )
            )
            """,
            con=database_connection,
            params=[self.scenario_name],
            index_col='time_period'
        )
        if len(constraint_schedule) > 0:

            # Parse parameters.
            constraint_schedule_numerical_columns = [
                'minimum_air_temperature',
                'maximum_air_temperature',
                'minimum_fresh_air_flow_per_area',
                'minimum_fresh_air_flow_per_person',
                'maximum_co2_concentration',
                'minimum_fresh_air_flow_per_area_no_dcv',
                'minimum_relative_humidity',
                'maximum_relative_humidity'
            ]
            constraint_schedule.loc[
                :, constraint_schedule_numerical_columns
            ] = (
                constraint_schedule.loc[
                    :, constraint_schedule_numerical_columns
                ].apply(parse_parameter)
            )

            # Parse time period index.
            constraint_schedule.index = np.vectorize(pd.Period)(constraint_schedule.index)

            # Obtain complete schedule for all weekdays.
            # TODO: Check if '01T00:00:00' is defined for each schedule.
            constraint_schedule_complete = []
            for constraint_type in constraint_schedule['constraint_type'].unique():
                for day in range(1, 8):
                    if day in constraint_schedule.index.day.unique():
                        constraint_schedule_complete.append(
                            constraint_schedule.loc[(
                                (constraint_schedule.index.day == day)
                                & (constraint_schedule['constraint_type'] == constraint_type)
                            ), :]
                        )
                    else:
                        constraint_schedule_previous = constraint_schedule_complete[-1].copy()
                        constraint_schedule_previous.index += pd.Timedelta('1 day')
                        constraint_schedule_complete.append(constraint_schedule_previous)
            constraint_schedule_complete = pd.concat(constraint_schedule_complete)

            # Pivot complete schedule.
            constraint_schedule_complete = constraint_schedule_complete.pivot(
                columns='constraint_type',
                values=constraint_schedule_complete.columns[constraint_schedule_complete.columns != 'constraint_type']
            )
            constraint_schedule_complete.columns = (
                constraint_schedule_complete.columns.map(lambda x: '_'.join(x[::-1]))
            )

            # Obtain complete schedule for each minute of the week.
            constraint_schedule_complete = (
                constraint_schedule_complete.reindex(
                    pd.period_range(start='01T00:00', end='07T23:59', freq='T')
                ).fillna(method='ffill')
            )

            # Reindex / fill internal gain schedule for given timesteps.
            constraint_schedule_complete.index = (
                pd.MultiIndex.from_arrays([
                    constraint_schedule_complete.index.day - 1,
                    constraint_schedule_complete.index.hour,
                    constraint_schedule_complete.index.minute
                ])
            )
            constraint_schedule = (
                pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([
                        self.timesteps.weekday,
                        self.timesteps.hour,
                        self.timesteps.minute
                    ]),
                    columns=constraint_schedule_complete.columns
                )
            )
            for column in constraint_schedule.columns:
                 constraint_schedule[column] = (
                     constraint_schedule[column].fillna(constraint_schedule_complete[column])
                 )
            constraint_schedule.index = self.timesteps

        else:
            constraint_schedule = None

        self.constraint_timeseries = constraint_schedule
