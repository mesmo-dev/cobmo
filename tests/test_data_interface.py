"""Test database interface."""

import os
import sqlite3
import time
import unittest

import cobmo.config
import cobmo.data_interface

logger = cobmo.config.get_logger(__name__)


class TestDatabaseInterface(unittest.TestCase):
    def test_recreate_database(self):
        # Get result.
        time_start = time.time()
        cobmo.data_interface.recreate_database()
        time_duration = time.time() - time_start
        logger.info(f"Test recreate_database: Completed in {time_duration:.6f} seconds.")

    def test_connect_database(self):
        # Define expected result.
        expected = sqlite3.dbapi2.Connection

        # Get actual result.
        time_start = time.time()
        actual = type(cobmo.data_interface.connect_database())
        time_duration = time.time() - time_start
        logger.info(f"Test connect_database: Completed in {time_duration:.6f} seconds.")

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_building_data(self):
        # Get result.
        time_start = time.time()
        cobmo.data_interface.BuildingData(cobmo.config.config["tests"]["scenario_name"])
        time_duration = time.time() - time_start
        logger.info(f"Test BuildingData: Completed in {time_duration:.6f} seconds.")


if __name__ == "__main__":
    unittest.main()
