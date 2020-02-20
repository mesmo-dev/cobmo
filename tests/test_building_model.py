"""Tests for `cobmo.building`."""

import pandas as pd
import time
import unittest

import cobmo.building_model
import cobmo.config
import cobmo.database_interface

logger = cobmo.config.get_logger(__name__)


class TestBuilding(unittest.TestCase):

    def test_building_model_default_scenario(self):
        # Setup.
        scenario_name = 'scenario_default'

        # Get result.
        time_start = time.time()
        cobmo.building_model.BuildingModel(scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test BuildingModel for scenario '{scenario_name}': Completed in {time_duration:.6f} seconds.")

    def test_building_model_all_scenarios(self):
        # Setup.
        database_connection = cobmo.database_interface.connect_database()
        building_scenarios = (
            pd.read_sql(
                """
                SELECT scenario_name FROM building_scenarios 
                """,
                database_connection
            )
        )

        # Iterate through all scenarios.
        for scenario_name in building_scenarios['scenario_name'].tolist():

            # Get result.
            time_start = time.time()
            cobmo.building_model.BuildingModel(scenario_name, database_connection)
            time_duration = time.time() - time_start
            logger.info(f"Test BuildingModel for scenario '{scenario_name}': Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
