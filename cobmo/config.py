"""Configuration module."""

import dynaconf
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import psychrolib


def get_config() -> dynaconf.Dynaconf:
    """Load the configuration dictionary.

    - Default configuration is obtained from `./cobmo/config_default.yml`.
    - Custom configuration is obtained from `./config.yml` and overwrites the respective default configuration.
    - `./` denotes the repository base directory.
    """

    # Create local `config.yml` for custom configuration in base directory, if not existing.
    # - The additional data paths setting is added for reference.
    if not (base_path / "config.yml").is_file():
        with open((base_path / "config.yml"), "w") as file:
            file.write(
                "# Local configuration parameters.\n"
                "# - Configuration parameters and their defaults are defined in `cobmo/config_default.yml`\n"
                "# - Copy from `cobmo/config_default.yml` and modify parameters here to set the local configuration.\n"
                "paths:\n"
                "  additional_data: []\n"
            )

    # Load configuration values.
    config_object = dynaconf.Dynaconf(
        settings_files=[
            base_path / "cobmo" / "config_default.yml",  # Default values are loaded first.
            base_path / "config.yml",  # Custom local values are loaded second and overwrite default values.
        ],
        merge_enabled=True,  # Merge nested configuration parameters.
    )

    # Obtain full paths.
    config_object['paths']['data'] = parse_path(config_object['paths']['data'])
    config_object['paths']['additional_data'] = (
        [parse_path(path) for path in config_object['paths']['additional_data']]
    )
    config_object['paths']['database'] = parse_path(config_object['paths']['database'])
    config_object['paths']['results'] = parse_path(config_object['paths']['results'])
    config_object['paths']['supplementary_data'] = parse_path(config_object['paths']['supplementary_data'])

    return config_object


def parse_path(path: str) -> pathlib.Path:
    """Parse path strings into pathlib objects. Replaces `./` with the repository base path and normalizes paths."""

    if path.startswith("./"):
        # Replace only first occurrence.
        return base_path / path.replace("./", "", 1)
    else:
        return pathlib.Path(path)


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
base_path = pathlib.Path(__file__).parent.parent

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
