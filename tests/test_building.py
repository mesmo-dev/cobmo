"""Tests for `cobmo.building`."""

import pandas as pd
import time
import unittest

import cobmo.building
import cobmo.config
import cobmo.database_interface

logger = cobmo.config.get_logger(__name__)


class TestBuilding(unittest.TestCase):

    def test_building_default_scenario(self):
        # Define expected result.
        expected = cobmo.building.Building

        # Setup.
        database_connection = cobmo.database_interface.connect_database()
        scenario_name = 'scenario_default'

        # Get actual result.
        time_start = time.time()
        actual = type(cobmo.building.Building(database_connection, scenario_name))
        time_end = time.time()
        logger.info(
            "Test building with default scenario '{}': Completed in {} seconds.".format(
                scenario_name,
                round(time_end - time_start, 6)
            )
        )

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_building_all_scenarios(self):
        # Define expected result.
        expected = cobmo.building.Building

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

            # Get actual result.
            time_start = time.time()
            actual = type(cobmo.building.Building(database_connection, scenario_name))
            time_end = time.time()
            logger.info(
                "Test building with scenario '{}': Completed in {} seconds.".format(
                    scenario_name,
                    round(time_end - time_start, 6)
                )
            )

            # Compare expected and actual.
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
