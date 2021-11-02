"""Configuration module."""

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import psychrolib
import yaml


def get_config() -> dict:
    """Load the configuration dictionary.

    - Default configuration is obtained from `./cobmo/config_default.yml`.
    - Custom configuration is obtained from `./config.yml` and overwrites the respective default configuration.
    - `./` denotes the repository base directory.
    """

    # Load default configuration values.
    with open(os.path.join(base_path, 'cobmo', 'config_default.yml'), 'r') as file:
        default_config = yaml.safe_load(file)

    # Create local `config.yml` for custom configuration in base directory, if not existing.
    # - The additional data paths setting is added for reference.
    if not os.path.isfile(os.path.join(base_path, 'config.yml')):
        with open(os.path.join(base_path, 'config.yml'), 'w') as file:
            file.write(
                "# Local configuration values.\n"
                "# - Default values can be found in `cobmo/config_default.yml`\n"
                "paths:\n"
                "  additional_data: []\n"
            )

    # Load custom configuration values, overwriting the default values.
    with open(os.path.join(base_path, 'config.yml'), 'r') as file:
        custom_config = yaml.safe_load(file)

    # Define utility function to recursively merge default and custom configuration.
    def merge_config(default_values: dict, custom_values: dict) -> dict:
        full_values = default_values.copy()
        full_values.update({
            key: (
                merge_config(default_values[key], custom_values[key])
                if (
                        (key in default_values)
                        and isinstance(default_values[key], dict)
                        and isinstance(custom_values[key], dict)
                )
                else custom_values[key]
            )
            for key in custom_values.keys()
        })
        return full_values

    # Obtain complete configuration.
    if custom_config is not None:
        complete_config = merge_config(default_config, custom_config)
    else:
        complete_config = default_config

    # Define utility function to obtain full paths.
    # - Replace `./` with the base path and normalize paths.
    def get_full_path(path: str) -> str:
        if path.startswith('./'):
            # Replace only first occurrence.
            path = path.replace('./', base_path + os.path.sep, 1)
        return os.path.normpath(path)

    # Obtain full paths.
    complete_config['paths']['data'] = get_full_path(complete_config['paths']['data'])
    complete_config['paths']['additional_data'] = (
        [get_full_path(path) for path in complete_config['paths']['additional_data']]
    )
    complete_config['paths']['database'] = get_full_path(complete_config['paths']['database'])
    complete_config['paths']['results'] = get_full_path(complete_config['paths']['results'])
    complete_config['paths']['supplementary_data'] = get_full_path(complete_config['paths']['supplementary_data'])

    return complete_config


def get_logger(
        name: str
) -> logging.Logger:
    """Generate logger with given name."""

    logger = logging.getLogger(name)

    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(logging.Formatter(config['logs']['format']))
    logger.addHandler(logging_handler)

    if config['logs']['level'] == 'debug':
        logger.setLevel(logging.DEBUG)
    elif config['logs']['level'] == 'info':
        logger.setLevel(logging.INFO)
    elif config['logs']['level'] == 'warn':
        logger.setLevel(logging.WARN)
    elif config['logs']['level'] == 'error':
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(f"Unknown logging level: {config['logs']['level']}")

    return logger


# Obtain repository base directory path.
base_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))

# Obtain configuration dictionary.
config = get_config()

# Modify matplotlib default settings.
plt.style.use(config['plots']['matplotlib_style'])
pd.plotting.register_matplotlib_converters()  # Remove warning when plotting with pandas.

# Modify pandas default settings.
# - These settings ensure that that data frames are always printed in full, rather than cropped.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
try:
    pd.set_option('display.max_colwidth', None)
except ValueError:
    # For compatibility with older versions of pandas.
    pd.set_option('display.max_colwidth', 0)

# Modify plotly default settings.
pio.templates.default = go.layout.Template(pio.templates['simple_white'])
pio.templates.default.layout.update(
    font=go.layout.Font(
        family=config['plots']['plotly_font_family'],
        size=config['plots']['plotly_font_size']
    ),
    legend=go.layout.Legend(borderwidth=1),
    xaxis=go.layout.XAxis(showgrid=True),
    yaxis=go.layout.YAxis(showgrid=True)
)
if pio.kaleido.scope is not None:
    pio.kaleido.scope.default_width = config['plots']['plotly_figure_width']
    pio.kaleido.scope.default_height = config['plots']['plotly_figure_height']
pio.orca.config.default_width = config['plots']['plotly_figure_width']
pio.orca.config.default_height = config['plots']['plotly_figure_height']

# Modify optimization solver settings.
if config['optimization']['solver_name'] == 'osqp':
    solver_parameters = dict(max_iter=1000000)
else:
    solver_parameters = dict()

# Modify psychrolib settings.
psychrolib.SetUnitSystem(psychrolib.SI)
