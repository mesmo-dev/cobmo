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
        additional_data_paths: typing.List[str] = cobmo.config.config['paths']['additional_data']
) -> None:
    """Recreate SQLITE database from SQL schema file and CSV files in the data path / additional data paths."""

    # Connect SQLITE database (creates file, if none).
    database_connection = sqlite3.connect(cobmo.config.config['paths']['database'])
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
    with open(os.path.join(cobmo.config.base_path, 'cobmo', 'data_schema.sql'), 'r') as database_schema_file:
        cursor.executescript(database_schema_file.read())
    database_connection.commit()

    # Import CSV files into SQLITE database.
    # - Import only from data path, if no additional data paths are specified.
    data_paths = (
        [cobmo.config.config['paths']['data']] + additional_data_paths
        if additional_data_paths is not None
        else [cobmo.config.config['paths']['data']]
    )
    valid_table_names = (
        pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", database_connection).iloc[:, 0].tolist()
    )
    for data_path in data_paths:
        for csv_file in glob.glob(os.path.join(data_path, '**', '*.csv'), recursive=True):

            # Exclude CSV files from supplementary data folders.
            if os.path.join('data', 'supplementary_data') not in csv_file:

                # Debug message.
                logger.debug(f"Loading {csv_file} into database.")

                # Obtain table name.
                table_name = os.path.splitext(os.path.basename(csv_file))[0]
                # Raise exception, if table doesn't exist.
                try:
                    assert table_name in valid_table_names
                except AssertionError:
                    logger.exception(
                        f"Error loading '{csv_file}' into database, because there is no table named '{table_name}'."
                    )
                    raise

                # Load table and write to database.
                try:
                    table = pd.read_csv(csv_file, dtype=str)
                    table.to_sql(
                        table_name,
                        con=database_connection,
                        if_exists='append',
                        index=False
                    )
                except Exception:
                    logger.error(f"Error loading {csv_file} into database.")
                    raise

    cursor.close()
    database_connection.close()


def connect_database() -> sqlite3.Connection:
    """Connect to the database and return connection handle."""

    # Recreate database, if no database exists.
    if not os.path.isfile(cobmo.config.config['paths']['database']):
        logger.debug(f"Database does not exist and is recreated at: {cobmo.config.config['paths']['database']}")
        recreate_database()

    # Obtain connection handle.
    database_connection = sqlite3.connect(cobmo.config.config['paths']['database'])
    return database_connection


class BuildingData(object):
    """Building data object consisting of building data items for the given scenario. The data items are loaded from
    the database on instantiation. Furthermore, parameters in the data tables are substituted with their numerical
    values and timeseries definitions are interpolated from timeseries tables or parsed from schedules where applicable.

    Syntax
        ``BuildingData(scenario_name)``: Instantiate building data object for given `scenario_name`.

    Parameters:
        scenario_name (str): CoBMo building scenario name, as defined in the data table `scenarios`.

    Keyword Arguments:
        database_connection (sqlite3.Connection): Database connection object. If not provided, a new connection
            is established.
        timestep_start (pd.Timestamp): If provided, will used in place of `timestep_start` from the scenario definition.
        timestep_end (pd.Timestamp): If provided, will used in place of `timestep_end` from the scenario definition.
        timestep_interval (pd.Timedelta): If provided, will used in place of `timestep_interval` from the scenario definition.

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
            ).loc[:, 'parameter_value']
        )

        # Obtain scenario.
        scenarios = (
            self.parse_parameters_dataframe(pd.read_sql(
                """
                SELECT * FROM scenarios 
                JOIN buildings USING (building_name) 
                JOIN linearization_types USING (linearization_type) 
                LEFT JOIN initial_state_types USING (initial_state_type) 
                LEFT JOIN storage_types USING (storage_type) 
                LEFT JOIN plant_heating_types USING (plant_heating_type) 
                LEFT JOIN plant_cooling_types USING (plant_cooling_type) 
                WHERE scenario_name = ?
                """,
                con=database_connection,
                params=[self.scenario_name]
            ))
        )
        # Raise error, if scenario not found.
        try:
            assert len(scenarios) > 0
        except AssertionError:
            logger.exception(f"No scenario found for scenario name '{scenario_name}'.")
            raise
        # Convert to Series for shorter indexing.
        self.scenarios = scenarios.iloc[0]

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
                LEFT JOIN hvac_generic_types USING (hvac_generic_type) 
                LEFT JOIN hvac_radiator_types USING (hvac_radiator_type) 
                LEFT JOIN hvac_ahu_types USING (hvac_ahu_type) 
                LEFT JOIN hvac_tu_types USING (hvac_tu_type) 
                LEFT JOIN hvac_vent_types USING (hvac_vent_type) 
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
                'linear'
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
                'linear'
            ).bfill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
            ).ffill(
                limit=int(pd.to_timedelta('1h') / pd.to_timedelta(self.timestep_interval))
            )
        )

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
            # TODO: Check if '01T00:00' is defined for each schedule.
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
            internal_gain_schedule_complete = internal_gain_schedule_complete.pivot(
                columns='internal_gain_type',
                values=['internal_gain_occupancy', 'internal_gain_appliances', 'warm_water_demand']
            )
            internal_gain_schedule_complete.columns = (
                internal_gain_schedule_complete.columns.map(lambda x: '_'.join(x[::-1]))
            )

            # Obtain complete schedule for each minute of the week.
            internal_gain_schedule_complete = (
                internal_gain_schedule_complete.reindex(
                    pd.period_range(start='01T00:00', end='07T23:59', freq='T')
                ).interpolate(method='linear').fillna(method='ffill')
            )

            # Reindex / fill schedule for given timesteps.
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
            internal_gain_timeseries = internal_gain_timeseries.pivot(
                columns='internal_gain_type',
                values=['internal_gain_occupancy', 'internal_gain_appliances', 'warm_water_demand']
            )
            internal_gain_timeseries.columns = (
                internal_gain_timeseries.columns.map(lambda x: '_'.join(x[::-1]))
            )

            # Reindex / interpolate timeseries for given timesteps.
            internal_gain_timeseries = (
                internal_gain_timeseries.reindex(
                    self.timesteps
                ).interpolate(
                    'linear'
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
            # TODO: Check if '01T00:00' is defined for each schedule.
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
                    pd.period_range(start='01T00:00', end='08T00:00', freq='T')
                ).astype(float).interpolate(method='linear').fillna(method='ffill')
            )

            # Reindex / fill schedule for given timesteps.
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

    def parse_parameters_column(
            self,
            column: np.ndarray
    ):
        """Parse parameters into one column of a dataframe.

        - Replace strings that match `parameter_name` with `parameter_value`.
        - Other strings are are directly parsed into numbers.
        - If a string doesn't match any match `parameter_name` and cannot be parsed, it is replaced with NaN.
        - Expects `column` to be passed as `np.ndarray` rather than directly as `pd.Series` (for performance reasons).
        """

        if column.dtype == object:  # `object` represents string type.
            if any(np.isin(column, self.parameters.index)):
                column_values = (
                    self.parameters.reindex(column).values
                )
                column_values[pd.isnull(column_values)] = (
                    pd.to_numeric(column[pd.isnull(column_values)])
                )
                column = column_values
            else:
                column = pd.to_numeric(column)

        # Explicitly parse to float, for consistent behavior independent of specific values.
        column = column.astype(float)

        return column

    def parse_parameters_dataframe(
            self,
            dataframe: pd.DataFrame,
            excluded_columns: list = None
    ):
        """Parse parameters into a dataframe.

        - Applies `parse_parameters_column` for all string columns.
        - Columns in `excluded_columns` are not parsed. By default this includes `_name`, `_type`, `_comment` columns.
        """

        # Define excluded columns. By default, all columns containing the following strings are excluded:
        # `_name`, `_type`, `connection`
        if excluded_columns is None:
            excluded_columns = ['parameter_set']
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_name')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_type')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('_comment')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.str.contains('timestep')])
        excluded_columns.extend(dataframe.columns[dataframe.columns.isin(
            ['parameter_set', 'time', 'time_period', 'timestep_start', 'timestep_end', 'timestep_interval', 'time_zone']
        )])

        # Select non-excluded, string columns and apply `parse_parameters_column`.
        selected_columns = (
            dataframe.columns[
                ~dataframe.columns.isin(excluded_columns)
                & (dataframe.dtypes == object)  # `object` represents string type.
            ]
        )
        for column in selected_columns:
            dataframe[column] = self.parse_parameters_column(dataframe[column].values)

        return dataframe
