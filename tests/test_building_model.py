"""Tests for building models."""

import pandas as pd
from parameterized import parameterized
import time
import unittest

import cobmo.building_model
import cobmo.config
import cobmo.database_interface

logger = cobmo.config.get_logger(__name__)

# Obtain database connection.
database_connection = cobmo.database_interface.connect_database()

# Obtain scenario names.
scenario_names = (
    pd.read_sql(
        """
        SELECT scenario_name FROM scenarios 
        """,
        database_connection
    )['scenario_name'].tolist()
)


class TestBuilding(unittest.TestCase):

    def test_building_model_default_scenario(self):
        # Get result.
        time_start = time.time()
        cobmo.building_model.BuildingModel(cobmo.config.test_scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test BuildingModel for default scenario: Completed in {time_duration:.6f} seconds.")

    @parameterized.expand(
        [(scenario_name,) for scenario_name in scenario_names])
    def test_building_model_any_scenario(self, scenario_name):
        # Get result.
        time_start = time.time()
        cobmo.building_model.BuildingModel(scenario_name)
        time_duration = time.time() - time_start
        logger.info(f"Test BuildingModel for scenario '{scenario_name}': Completed in {time_duration:.6f} seconds.")


if __name__ == '__main__':
    unittest.main()
