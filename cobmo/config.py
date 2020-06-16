"""Configuration parameters."""

import datetime
import logging
import os
import pandas as pd

# Path definitions.
cobmo_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
data_path = os.path.join(cobmo_path, 'data')
database_path = os.path.join(data_path, 'database.sqlite')
results_path = os.path.join(cobmo_path, 'results')
supplementary_data_path = os.path.join(data_path, 'supplementary_data')

# Generate timestamp (for saving results with timestamp).
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Optimization solver settings.
solver_name = 'gurobi'  # Must be valid input string for Pyomo's `SolverFactory`.
solver_output = False  # If True, activate verbose solver output.

# Test settings.
test_scenario_name = 'create_level8_4zones_a'

# Pandas settings.
# - These settings ensure that that data frames are always printed in full, rather than cropped.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
try:
    pd.set_option('display.max_colwidth', None)
except ValueError:
    # This is provided for compatibility with older versions of Pandas.
    pd.set_option('display.max_colwidth', 0)

# Logger settings.
logging_level = logging.INFO
logging_handler = logging.StreamHandler()
logging_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(message)s'))
# logging_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging_handler)
    logger.setLevel(logging_level)
    return logger
