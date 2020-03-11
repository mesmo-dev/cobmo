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

    building_scenarios: pd.Series
    building_parameters: pd.Series
    building_surfaces_adiabatic: pd.DataFrame
    building_surfaces_exterior: pd.DataFrame
    building_surfaces_interior: pd.DataFrame
    building_zones: pd.DataFrame

    @ multimethod
    def __init__(
            self,
            scenario_name: str
    ) -> None:

        # Obtain database connection.
        database_connection = connect_database()

        self.__init__(
            scenario_name,
            database_connection
        )

    @ multimethod
    def __init__(
            self,
            scenario_name: str,
            database_connection: sqlite3.Connection
    ) -> None:

        # Store scenario name.
        self.scenario_name = scenario_name

        # Obtain building data.
        self.building_scenarios = (
            pd.read_sql(
                """
                SELECT * FROM building_scenarios 
                JOIN buildings USING (building_name) 
                JOIN building_linearization_types USING (linearization_type) 
                LEFT JOIN building_initial_state_types USING (initial_state_type) 
                LEFT JOIN building_storage_types USING (building_storage_type) 
                WHERE scenario_name = ?
                """,
                con=database_connection,
                params=[self.scenario_name]
            ).iloc[0]  # Convert to Series for shorter indexing.
        )
        self.building_parameters = (
            pd.read_sql(
                """
                SELECT parameter_name, parameter_value FROM building_parameter_sets 
                WHERE parameter_set IN ('constants', ?)
                """,
                con=database_connection,
                params=[self.building_scenarios['parameter_set']],
                index_col='parameter_name'
            ).iloc[:, 0]  # Convert to Series for shorter indexing.
        )
        self.building_surfaces_adiabatic = (
            pd.read_sql(
                """
                SELECT * FROM building_surfaces_adiabatic 
                JOIN building_surface_types USING (surface_type) 
                LEFT JOIN building_window_types USING (window_type) 
                JOIN building_zones USING (zone_name, building_name) 
                WHERE building_name = ?
                """,
                con=database_connection,
                params=[self.building_scenarios['building_name']]
            )
        )
        self.building_surfaces_adiabatic.index = self.building_surfaces_adiabatic['surface_name']
        self.building_surfaces_exterior = (
            pd.read_sql(
                """
                SELECT * FROM building_surfaces_exterior 
                JOIN building_surface_types USING (surface_type) 
                LEFT JOIN building_window_types USING (window_type) 
                JOIN building_zones USING (zone_name, building_name) 
                WHERE building_name = ?
                """,
                con=database_connection,
                params=[self.building_scenarios['building_name']]
            )
        )
        self.building_surfaces_exterior.index = self.building_surfaces_exterior['surface_name']
        self.building_surfaces_interior = (
            pd.read_sql(
                """
                SELECT * FROM building_surfaces_interior 
                JOIN building_surface_types USING (surface_type) 
                LEFT JOIN building_window_types USING (window_type) 
                JOIN building_zones USING (zone_name, building_name) 
                WHERE building_name = ?
                """,
                con=database_connection,
                params=[self.building_scenarios['building_name']]
            )
        )
        self.building_surfaces_interior.index = self.building_surfaces_interior['surface_name']
        self.building_zones = (
            pd.read_sql(
                """
                SELECT * FROM building_zones 
                JOIN building_zone_types USING (zone_type) 
                JOIN building_internal_gain_types USING (internal_gain_type) 
                LEFT JOIN building_blind_types USING (blind_type) 
                LEFT JOIN building_hvac_generic_types USING (hvac_generic_type) 
                LEFT JOIN building_hvac_radiator_types USING (hvac_radiator_type) 
                LEFT JOIN building_hvac_ahu_types USING (hvac_ahu_type) 
                LEFT JOIN building_hvac_tu_types USING (hvac_tu_type) 
                WHERE building_name = ?
                """,
                con=database_connection,
                params=[self.building_scenarios['building_name']]
            )
        )
        self.building_zones.index = self.building_zones['zone_name']

        # Define parameter parsing utility function.
        @np.vectorize
        def parse_parameter(parameter_string):
            """Parse parameter utility function.
            - Convert given parameter string as `np.float`.
            - If the parameter string is the name of a parameter from `building_parameters`, return that parameter.
            - If the parameter string is `None`, `NaN` is returned.
            """
            try:
                return np.float(parameter_string)
            except ValueError:
                return self.building_parameters[parameter_string]
            except TypeError:
                return np.nan

        # Parse parameters.
        building_scenarios_numerical_columns = [
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
        self.building_scenarios.loc[building_scenarios_numerical_columns] = (
            self.building_scenarios.loc[building_scenarios_numerical_columns].apply(parse_parameter)
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
        self.building_surfaces_adiabatic.loc[:, building_surfaces_numerical_columns] = (
            self.building_surfaces_adiabatic.loc[:, building_surfaces_numerical_columns].apply(parse_parameter)
        )
        self.building_surfaces_exterior.loc[:, building_surfaces_numerical_columns] = (
            self.building_surfaces_exterior.loc[:, building_surfaces_numerical_columns].apply(parse_parameter)
        )
        self.building_surfaces_interior.loc[:, building_surfaces_numerical_columns] = (
            self.building_surfaces_interior.loc[:, building_surfaces_numerical_columns].apply(parse_parameter)
        )
        building_zones_numerical_columns = [
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
        self.building_zones.loc[:, building_zones_numerical_columns] = (
            self.building_zones.loc[:, building_zones_numerical_columns].apply(parse_parameter)
        )
