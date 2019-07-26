"""Configuration parameters."""

import os
import logging

cobmo_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
data_path = os.path.join(cobmo_path, 'data')

logging_level = logging.DEBUG
logging_handler = logging.StreamHandler()
logging_handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(message)s'))
# logging_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging_handler)
    logger.setLevel(logging_level)
    return logger
