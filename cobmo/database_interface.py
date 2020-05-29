"""Database interface function definitions."""

import glob
import numpy as np
import os
import pandas as pd
import sqlite3
import typing

import cobmo.config

logger = cobmo.config.get_logger(__name__)


def recreate_database(
        database_path: str = cobmo.config.database_path,
        database_schema_path: str = os.path.join(cobmo.config.cobmo_path, 'cobmo', 'database_schema.sql'),
        data_path: str = cobmo.config.data_path,
        additional_data_paths: typing.List[str] = None
) -> None:
    """Recreate SQLITE database from SQL schema file and CSV files."""

    # Connect SQLITE database. Creates file, if none.
    database_connection = sqlite3.connect(database_path)
    database_connection.text_factory = str  # Allows UTF-8 data to be stored.
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

    # Recreate SQLITE database schema from SQL schema file.
    with open(database_schema_path, 'r') as database_schema_file:
        cursor.executescript(database_schema_file.read())
    database_connection.commit()

    # Import CSV files into SQLITE database.
    csv_paths = ([data_path] + additional_data_paths) if additional_data_paths is not None else [data_path]
    for csv_path in csv_paths:
        for file in glob.glob(os.path.join(csv_path, '**', '*.csv'), recursive=True):

            # Exclude CSV files from supplementary data folders.
            if os.path.join('data', 'supplementary_data') not in file:

                # Obtain table name.
                table_name = os.path.splitext(os.path.basename(file))[0]

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
    """Building data object consisting of building data items for the given scenario. The data items are loaded from
    the database on instantiation. Furthermore, parameters in the data tables are substituted with their numerical
    values and timeseries definitions are interpolated from timeseries tables or parsed from schedules where applicable.

    Syntax
        `BuildingData(scenario_name): Instantiate building data object for given `scenario_name`.

    Parameters:
        scenario_name (str): CoBMo building scenario name, as defined in the data table `scenarios`.

    Keyword Arguments:
        database_connection (sqlite3.Connection): Database connection object. If not provided, a new connection
            is established.
        timestep_start (pd.Timestamp): If provided, will used in place of `timestep_start` in the scenario definition.
        timestep_end (pd.Timestamp): If provided, will used in place of `timestep_end` in the scenario definition.
        timestep_interval (pd.Timedelta): If provided, will used in place of `timestep_interval` in the scenario definition.

    Attributes:
        scenario_name (str): CoBMo building scenario name
        parameters (pd.Series): Parameters table, containing only data related to the given scenario.
        scenarios (pd.Series): Scenarios table, containing only the row related to the given scenario.
        surfaces_adiabatic (pd.DataFrame): Adiabatic surfaces table, containing only data related to the given scenario.
        surfaces_exterior (pd.DataFrame): Exterior surfaces table, containing only data related to the given scenario.
        surfaces_interior (pd.DataFrame): Interior surfaces table, containing only data related to the given scenario.
        zones (pd.DataFrame): Zones table, containing only data related to the given scenario.
        timestep_start (pd.Timestamp): Start timestep.
        timestep_end (pd.Timestamp): End timestep.
        timestep_interval (pd.Timedelta): Time interval between timesteps.
        timesteps (pd.Index): Index set of the timesteps.
        weather_timeseries (pd.DataFrame): Weather timeseries for the given scenario.
        electricity_price_timeseries (pd.DataFrame): Electricity price timeseries for the given scenario.
        electricity_price_distribution_timeseries (pd.DataFrame): Electricity price distribution timeseries for
            the given scenario.
        internal_gain_timeseries (pd.DataFrame): Internal gain timeseries for the given scenario.
        constraint_timeseries (pd.DataFrame): Constraint timeseries for the given scenario.
    """

    scenario_name: str
    parameters: pd.Series
    scenarios: pd.Series
    surfaces_adiabatic: pd.DataFrame
    surfaces_exterior: pd.DataFrame
    surfaces_interior: pd.DataFrame
    zones: pd.DataFrame
    timestep_start: pd.Timestamp
    timestep_end: pd.Timestamp
    timestep_interval: pd.Timedelta
    timesteps: pd.Index
    weather_timeseries: pd.DataFrame
    electricity_price_timeseries: pd.DataFrame
    electricity_price_distribution_timeseries: pd.DataFrame = None  # Defaults to None if not defined.
    internal_gain_timeseries: pd.DataFrame = None  # Defaults to None if not defined.
    constraint_timeseries: pd.DataFrame

    def __init__(
            self,
            scenario_name: str,
            database_connection=connect_database(),
            timestep_start=None,
            timestep_end=None,
            timestep_interval=None
    ) -> None:

        # Store scenario name.
        self.scenario_name = scenario_name

        # Obtain parameters.
        self.parameters = (
            pd.read_sql(
                """
                SELECT parameter_name, parameter_value FROM parameters 
                WHERE parameter_set IN (
                    'constants',
                    (SELECT parameter_set FROM scenarios WHERE scenario_name = ?)
                )
                """,
                con=database_connection,
                params=[scenario_name],
                index_col='parameter_name'
            ).iloc[:, 0]  # Convert to Series for shorter indexing.
        )

        # Obtain scenarios.
        self.scenarios = (
            self.parse_parameters_dataframe(pd.read_sql(
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
            )).iloc[0]  # Convert to Series for shorter indexing.
        )

        # Obtain surface definitions.
        self.surfaces_adiabatic = (
            self.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.surfaces_adiabatic.index = self.surfaces_adiabatic['surface_name']
        self.surfaces_exterior = (
            self.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.surfaces_exterior.index = self.surfaces_exterior['surface_name']
        self.surfaces_interior = (
            self.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        self.surfaces_interior.index = self.surfaces_interior['surface_name']

        # Obtain zone definitions.
        self.zones = (
            self.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM zones 
                JOIN zone_types USING (zone_type) 
                LEFT JOIN internal_gain_types USING (internal_gain_type) 
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
            ))
        )
        self.zones.index = self.zones['zone_name']

        # Obtain timestep data.
        if timestep_start is not None:
            self.timestep_start = pd.Timestamp(timestep_start)
        else:
            self.timestep_start = pd.Timestamp(self.scenarios['timestep_start'])
        if timestep_end is not None:
            self.timestep_end = pd.Timestamp(timestep_end)
        else:
            self.timestep_end = pd.Timestamp(self.scenarios['timestep_end'])
        if timestep_interval is not None:
            self.timestep_interval = pd.Timedelta(timestep_interval)
        else:
            self.timestep_interval = pd.Timedelta(self.scenarios['timestep_interval'])
        self.timesteps = pd.Index(
            pd.date_range(
                start=self.timestep_start,
                end=self.timestep_end,
                freq=self.timestep_interval
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
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
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
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
            )
        )

        # Obtain electricity price distribution timeseries.
        electricity_price_range = pd.read_sql(
            """
            SELECT * FROM electricity_price_range
            """,
            con=database_connection,
            index_col='time_period'
        )
        if len(electricity_price_range) > 0:
            # Parse time period index.
            electricity_price_range.index = np.vectorize(pd.Period)(electricity_price_range.index)
            # Obtain complete schedule for all weekdays.
            electricity_price_range_complete = []
            for day in range(1, 8):
                if day in electricity_price_range.index.day.unique():
                    electricity_price_range_complete.append(
                        electricity_price_range.loc[
                                                    (electricity_price_range.index.day == day)
                                                    , :]
                    )
                else:
                    electricity_price_range_previous = electricity_price_range_complete[-1].copy()
                    electricity_price_range_previous.index += pd.Timedelta('1 day')
                    electricity_price_range_complete.append(electricity_price_range_previous)
            electricity_price_range_complete = pd.concat(electricity_price_range_complete)

            # Obtain complete schedule for each minute of the week.
            electricity_price_range_complete = (
                electricity_price_range_complete.reindex(
                    pd.period_range(start='01T00:00', end='07T23:59', freq='T')
                ).fillna(method='ffill')
            )

            # Reindex / fill internal gain schedule for given timesteps.
            electricity_price_range_complete.index = (
                pd.MultiIndex.from_arrays([
                    electricity_price_range_complete.index.day - 1,
                    electricity_price_range_complete.index.hour,
                    electricity_price_range_complete.index.minute
                ])
            )
            electricity_price_range = (
                pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([
                        self.timesteps.weekday,
                        self.timesteps.hour,
                        self.timesteps.minute
                    ]),
                    columns=electricity_price_range_complete.columns
                )
            )
            for column in electricity_price_range.columns:
                electricity_price_range[column] = (
                    electricity_price_range[column].fillna(electricity_price_range_complete[column])
                )
            electricity_price_range.index = self.timesteps
        else:
            electricity_price_range = None

        self.electricity_price_distribution_timeseries = electricity_price_range

        # Obtain internal gain timeseries based on schedules.
        internal_gain_schedule = pd.read_sql(
            """
            SELECT * FROM internal_gain_schedules 
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
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
                ).ffill(
                    limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
                )
            )

        else:
            internal_gain_timeseries = None

        # Merge schedule-based and timeseries-based internal gain timeseries.
        if (internal_gain_schedule is not None) or (internal_gain_timeseries is not None):
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
        constraint_schedule = (
            self.parse_parameters_dataframe(pd.read_sql(
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
            ))
        )
        if len(constraint_schedule) > 0:

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

    def parse_parameter(
            self,
            parameter_string: str
    ):
        """Parse parameter string to numerical value.
        - Replace strings that match `parameter_name` with `parameter_value`.
        - Other strings are are directly parsed into numbers.
        - If a string doesn't match any match `parameter_name` and cannot be parsed, it is replaced with NaN.
        """

        try:
            return np.float(parameter_string)
        except ValueError:
            return self.parameters[parameter_string]
        except TypeError:
            return np.nan

    def parse_parameters_dataframe(
            self,
            dataframe: pd.DataFrame,
            excluded_columns: list = None
    ):
        """Parse parameters into a dataframe.
        - Applies `parse_parameter` for all string columns.
        - Columns in `excluded_columns` are not parsed. By default this includes `_name`, `_type`, `connection` columns.
        """

        # Define excluded columns. By default, all columns containing the following strings are excluded:
        # `_name`, `_type`, `connection`
        if excluded_columns is None:
            excluded_columns = []
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_name')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_type')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_comment')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('parameter_set')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('time')])

        # Select non-excluded, string columns and apply `parse_parameter`.
        selected_columns = (
            dataframe.columns[
                ~dataframe.columns.isin(excluded_columns)
                & (dataframe.dtypes == object)  # `object` represents string type.
            ]
        )
        dataframe.loc[:, selected_columns] = (
            dataframe.loc[:, selected_columns].applymap(self.parse_parameter)
        )

        return dataframe
