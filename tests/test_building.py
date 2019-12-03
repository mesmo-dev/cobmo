"""Tests for `cobmo.building`."""

import time
import unittest

import cobmo.building
import cobmo.config
import cobmo.database_interface

logger = cobmo.config.get_logger(__name__)


class TestBuilding(unittest.TestCase):

    def test_building_init(self):
        # Define expected result.
        expected = cobmo.building.Building

        # Setup.
        database_connection = cobmo.database_interface.connect_database()
        scenario_name = 'scenario_default'

        # Get actual result.
        time_start = time.time()
        actual = type(cobmo.building.Building(database_connection, scenario_name))
        time_end = time.time()
        logger.info("Test equal: Completed in {} seconds.".format(round(time_end - time_start, 6)))

        # Compare expected and actual.
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
