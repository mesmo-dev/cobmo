"""Utility functions module."""

import cvxpy as cp
import datetime
import logging
import numpy as np
import os
import pandas as pd
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import psychrolib
import pvlib
import re
import scipy.sparse
import subprocess
import sys
import time
import typing

import cobmo.config

logger = cobmo.config.get_logger(__name__)

# Instantiate dictionary for execution time logging.
log_times = dict()


class OptimizationProblem(object):
    """Optimization problem object for use with CVXPY."""

    constraints: list
    objective: cp.Expression
    cvxpy_problem: cp.Problem

    def __init__(self):

        self.constraints = []
        self.objective = cp.Constant(value=0.0)

    def solve(
            self,
            keep_problem=False
    ):

        # Instantiate CVXPY problem object.
        if hasattr(self, 'cvxpy_problem') and keep_problem:
            pass
        else:
            self.cvxpy_problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

        # Solve optimization problem.
        self.cvxpy_problem.solve(
            solver=(
                cobmo.config.config['optimization']['solver_name'].upper()
                if cobmo.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=cobmo.config.config['optimization']['show_solver_output'],
            **cobmo.config.solver_parameters
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        try:
            assert self.cvxpy_problem.status == cp.OPTIMAL
        except AssertionError:
            logger.error(f"Solver termination status: {self.cvxpy_problem.status}")
            raise


class MatrixConstructor(object):
    """Matrix constructor object for more performant matrix construction operations.

    - Creates a matrix representation for given index, column sets.
    - Values can be added to the matrix using inplace addition ``+=`` and value setting ``=``. Note that
      value setting ``=`` behaves like inplace addition ``+=``, i.e. existing values are not overwritten but added up.
      True value setting is not available for performance reasons.
    - Value getting always returns 0. True value getting is not available for performance reasons.
    - Once the matrix construction is complete, the matrix representation can be converted to Scipy sparse CSR matrix
      with ``to_scipy_csr()`` or Pandas dataframe in sparse / dense format with ``to_dataframe_sparse()`` /
      ``to_dataframe_dense()``.

    :syntax:
        ``MatrixConstructor(index, columns)``: Instantiate matrix constructor object for given index (rows) and columns.
        ``matrix[index_key, column_key] += 1.0``: Add value 1.0 at given row / column key location.
        ``matrix[index_key, column_key] = 1.0``: Add value 1.0 at given row / column key location.

    Parameters:
        index (pd.Index): Index (row) key set.
        columns (pd.Index): Columns key set.

    Attributes:
        index (pd.Index): Index (row) key set.
        columns (pd.Index): Columns key set.
        data_index (list): List of data entry row locations as integer index.
        data_columns (list): List of data entry column locations as integer index.
        data_values (list): List of data entry values.
"""

    index: pd.Index
    columns: pd.Index
    data_index: list
    data_columns: list
    data_values: list

    def __init__(
            self,
            index: typing.Union[pd.Index],
            columns: typing.Union[pd.Index]
    ):

        self.index = index
        self.columns = columns
        self.data_index = list()
        self.data_columns = list()
        self.data_values = list()

    def __getitem__(
            self,
            key: typing.Tuple[any, any]
    ) -> float:

        # Always return 0, to enable inplace addition ``+=``.
        # - True value getting is not available for performance reasons.
        return 0.0

    def __setitem__(
            self,
            key: typing.Tuple[any, any],
            value: any
    ):

        # Assert that key has exactly 2 entries, otherwise raise error.
        if len(key) != 2:
            raise ValueError(f"Cannot use key with {len(key)} items. Only key with 2 items is valid.")

        # Append new values.
        # - Integer key locations are obtained from index sets.
        # - Note that existing values are not overwritten but added up.
        # - True value setting is not available for performance reasons.
        self.data_index.append(self.index.get_loc(key[0]))
        self.data_columns.append(self.columns.get_loc(key[1]))
        self.data_values.append(value)

    def to_scipy_csr(self) -> scipy.sparse.csr_matrix:
        """Obtain Scipy sparse CSR matrix."""

        return (
            scipy.sparse.csr_matrix(
                (self.data_values, (self.data_index, self.data_columns)),
                shape=(len(self.index), len(self.columns))
            )
        )

    def to_dataframe_sparse(self) -> pd.DataFrame:
        """Obtain Pandas dataframe in sparse format.

        - Reference for Pandas sparse dataframes: <https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html>
        """

        return (
            pd.DataFrame.sparse.from_spmatrix(
                scipy.sparse.coo_matrix(
                    (self.data_values, (self.data_index, self.data_columns)),
                    shape=(len(self.index), len(self.columns))
                ),
                index=self.index,
                columns=self.columns
            )
        )

    def to_dataframe_dense(self) -> pd.DataFrame:
        """Obtain Pandas dataframe in dense format."""

        return self.to_dataframe_sparse().sparse.to_dense()


def log_time(
        label: str,
        log_level: str = 'debug',
        logger_object: logging.Logger = logger
):
    """Log start / end message and time duration for given label.

    - When called with given label for the first time, will log start message.
    - When called subsequently with the same / previously used label, will log end message and time duration since
      logging the start message.
    - The log level for start / end messages can be given as keyword argument, By default, messages are logged as
      debug messages.
    - The logger object can be given as keyword argument. By default, uses ``utils.logger`` as logger.
    - Start message: "Starting ``label``."
    - End message: "Completed ``label`` in ``duration`` seconds."

    Arguments:
        label (str): Label for the start / end message.

    Keyword Arguments:
        log_level (str): Log level to which the start / end messages are output. Choices: 'debug', 'info'.
            Default: 'debug'.
        logger_object (logging.logger.Logger): Logger object to which the start / end messages are output. Default:
            ``utils.logger``.
    """

    time_now = time.time()

    if log_level == 'debug':
        logger_handle = lambda message: logger_object.debug(message)
    elif log_level == 'info':
        logger_handle = lambda message: logger_object.info(message)
    else:
        raise ValueError(f"Invalid log level: '{log_level}'")

    if label in log_times.keys():
        logger_handle(f"Completed {label} in {(time_now - log_times.pop(label)):.6f} seconds.")
    else:
        log_times[label] = time_now
        logger_handle(f"Starting {label}.")


def calculate_absolute_humidity_humid_air(
        temperature,  # In °C.
        relative_humidity  # In percent.
):
    absolute_humidity = (
        psychrolib.GetHumRatioFromRelHum(
            TDryBulb=temperature,  # In °C.
            RelHum=relative_humidity / 100.0,  # In [0,1].
            Pressure=101325.0  # In Pa.
        )
    )
    return absolute_humidity  # In kg(water)/kg(air).


def calculate_enthalpy_humid_air(
        temperature,  # In °C.
        absolute_humidity  # In kg(water)/kg(air).
):
    enthalpy = (
        psychrolib.GetMoistAirEnthalpy(
            TDryBulb=temperature,  # In °C.
            HumRatio=absolute_humidity  # In kg(water)/kg(air).
        )
    )
    return enthalpy  # In J/kg.


def calculate_dew_point_enthalpy_humid_air(
        temperature,  # In °C.
        relative_humidity  # In percent.
):
    enthalpy = (
        psychrolib.GetMoistAirEnthalpy(
            TDryBulb=psychrolib.GetTDewPointFromRelHum(
                TDryBulb=temperature,  # In °C.
                RelHum=relative_humidity / 100.0  # In [0,1].
            ),
            HumRatio=calculate_absolute_humidity_humid_air(
                temperature,  # In °C.
                relative_humidity  # In percent.
            )
        )
    )
    return enthalpy  # In J/kg.


def calculate_irradiation_surfaces(
        database_connection,
        weather_type='singapore_nus',
        irradiation_model='dirint'
):
    """Calculates irradiation for surfaces oriented towards east, south, west & north.

    - Operates on the database: Updates according columns in `weather_timeseries`.
    - Takes irradition_horizontal as measured global horizontal irradiation (ghi).
    - Based on pvlib-python toolbox: https://github.com/pvlib/pvlib-python
    """

    # Load weather data from database
    weather_types = pd.read_sql(
        """ 
        select * from weather_types  
        where weather_type='{}' 
        """.format(weather_type),
        database_connection
    )
    weather_timeseries = pd.read_sql(
        """ 
        select * from weather_timeseries  
        where weather_type='{}' 
        """.format(weather_type),
        database_connection
    )

    # Set time zone (required for pvlib solar position calculations).
    weather_timeseries.index = pd.to_datetime(weather_timeseries['time'])
    weather_timeseries.index = weather_timeseries.index.tz_localize(weather_types['time_zone'][0])

    # Extract global horizontal irradiation (ghi) from weather data.
    irradiation_ghi = weather_timeseries['irradiation_horizontal']

    # Calculate solarposition (zenith, azimuth).
    solarposition = pvlib.solarposition.get_solarposition(
        time=weather_timeseries.index,
        latitude=weather_types['latitude'][0],
        longitude=weather_types['longitude'][0]
    )

    # Calculate direct normal irradiation (dni) from global horizontal irradiation (ghi).
    irradiation_dni = pd.Series(index=weather_timeseries.index)
    if irradiation_model == 'disc':
        # ... via DISC model.
        irradiation_disc = pvlib.irradiance.disc(
            ghi=irradiation_ghi,
            solar_zenith=solarposition['zenith'],
            datetime_or_doy=weather_timeseries.index
        )
        irradiation_dni = irradiation_disc['dni']
    elif irradiation_model == 'erbs':
        # ... via ERBS model.
        irradiation_erbs = pvlib.irradiance.erbs(
            ghi=irradiation_ghi,
            zenith=solarposition['zenith'],
            datetime_or_doy=weather_timeseries.index
        )
        irradiation_dni = irradiation_erbs['dni']
    elif irradiation_model == 'dirint':
        # ... via DIRINT model.
        irradiation_dirint = pvlib.irradiance.dirint(
            ghi=irradiation_ghi,
            solar_zenith=solarposition['zenith'],
            times=weather_timeseries.index,
            temp_dew=np.vectorize(psychrolib.GetTDewPointFromHumRatio)(  # In °C.
                TDryBulb=weather_timeseries['ambient_air_temperature'].values,  # In °C.
                HumRatio=weather_timeseries['ambient_air_absolute_humidity'].values,  # In kg(water)/kg(air).
                Pressure=101325  # In Pa.
            )
        )
        irradiation_dni = irradiation_dirint

    # Replace NaNs (NaN means no irradiation).
    irradiation_dni.loc[irradiation_dni.isna()] = 0.0

    # Calculate diffuse horizontal irradiation (dhi).
    irradiation_dhi = pd.Series(
            irradiation_ghi
            - irradiation_dni
            * pvlib.tools.cosd(solarposition['zenith']),
    )

    # Define surface orientations.
    surface_orientations = pd.DataFrame(
        data=[0.0, 90.0, 180.0, 270.0],
        index=['north', 'east', 'south', 'west'],
        columns=['surface_azimuth']
    )

    # Calculate irradiation onto each surface.
    for index, row in surface_orientations.iterrows():
        irradiation_surface = pvlib.irradiance.get_total_irradiance(
            surface_tilt=90.0,
            surface_azimuth=row['surface_azimuth'],
            solar_zenith=solarposition['zenith'],
            solar_azimuth=solarposition['azimuth'],
            dni=irradiation_dni,
            ghi=irradiation_ghi,
            dhi=irradiation_dhi,
            surface_type='urban',
            model='isotropic'
        )
        weather_timeseries.loc[:, 'irradiation_' + index] = irradiation_surface['poa_global']

    # Update weather_timeseries in database.
    database_connection.cursor().execute(
        """ 
        delete from weather_timeseries  
        where weather_type='{}' 
        """.format(weather_type),
    )
    weather_timeseries.to_sql(
        'weather_timeseries',
        database_connection,
        if_exists='append',
        index=False
    )


def calculate_sky_temperature(
        database_connection,
        weather_type='singapore_nus'
):
    """ Calculates sky temperatures from ambient air temperature for tropical weather.

    - Ambient air temperature is decreased by 11K to get the sky temperature.
    - According to ISO 52016-1, Table B.19.
    """
    # Load weather data.
    weather_types = pd.read_sql(
        """ 
        select * from weather_types  
        where weather_type='{}' 
        """.format(weather_type),
        database_connection
    )
    weather_timeseries = pd.read_sql(
        """ 
        select * from weather_timeseries  
        where weather_type='{}' 
        """.format(weather_type),
        database_connection
    )
    weather_timeseries.index = pd.to_datetime(weather_timeseries['time'])

    # Get temperature difference between sky and ambient.
    temperature_difference = weather_types['temperature_difference_sky_ambient'][0]

    # Calculate sky temperature.
    weather_timeseries.loc[:, 'sky_temperature'] = \
        weather_timeseries.loc[:, 'ambient_air_temperature'] - temperature_difference

    # Update weather_timeseries in database.
    database_connection.cursor().execute(
        """ 
        delete from weather_timeseries  
        where weather_type='{}' 
        """.format(weather_type),
    )
    weather_timeseries.to_sql('weather_timeseries', database_connection, if_exists='append', index=False)


def calculate_error(
        expected_timeseries=pd.DataFrame(),
        predicted_timeseries=pd.DataFrame()
):
    """Computes the error between expected and predicted timeseries dataframes.

    - Note: This function doesn't check if the data format is valid.
    """

    # Instantiate error timeseries / summary dataframes.
    error_timeseries = pd.DataFrame(
        0.0,
        index=expected_timeseries.index,
        columns=expected_timeseries.columns
    )
    error_summary = pd.DataFrame(
        0.0,
        index=pd.Index(['mean_absolute_error', 'root_mean_squared_error'], name='error_type'),
        columns=expected_timeseries.columns
    )

    # Calculate error values.
    for index, row in error_timeseries.iterrows():
        error_timeseries.loc[index, :] = (
            predicted_timeseries.loc[index, :]
            - expected_timeseries.loc[index, :]
        )
    for column_name, column in error_summary.iteritems():
        error_summary.loc['mean_absolute_error', column_name] = (
            error_timeseries[column_name].abs().mean()
        )
        error_summary.loc['root_mean_squared_error', column_name] = (
            (error_timeseries[column_name] ** 2).mean() ** 0.5
        )

    return (
        error_summary,
        error_timeseries
    )


def calculate_tank_diameter_height(
        volume,
        aspect_ratio
):
    """Calculates diameter and height of storage tank based on volume and aspect ratio."""

    # Calculations.
    diameter = (volume / aspect_ratio * 4 / np.pi) ** (1 / 3)
    height = diameter * aspect_ratio

    return (
        diameter,
        height
    )


def calculate_discounted_payback_time(
        lifetime,
        investment_cost,
        operation_cost,
        operation_cost_baseline,
        interest_rate=0.06
):
    """Calculate simple / discounted payback time in years."""

    # Calculate annual cost savings.
    operation_cost_savings_annual = (operation_cost_baseline - operation_cost) / lifetime

    if (operation_cost_savings_annual <= 0.0) or (investment_cost == 0.0):
        # Return `None` if no savings observed.
        simple_payback_time = None
        discounted_payback_time = None

    else:
        # Calculate simple payback time.
        simple_payback_time = int(np.ceil(investment_cost / operation_cost_savings_annual))
        if simple_payback_time >= lifetime:
            # If simple payback time is greater than lifetime, return None.
            simple_payback_time = None

        # Calculate discounted payback time in years.
        year = 0
        annual_discounted_savings = np.zeros(int(np.ceil(lifetime)) + 1)
        cumulative_discounted_savings = np.zeros(int(np.ceil(lifetime)) + 1)
        while cumulative_discounted_savings[year] < investment_cost:
            year += 1
            discount_factor = (1.0 + interest_rate) ** (-year)
            annual_discounted_savings[year] = operation_cost_savings_annual * discount_factor
            cumulative_discounted_savings[year] = cumulative_discounted_savings[year - 1] + annual_discounted_savings[year]

            # Discontinue calculations if payback is not reached within lifetime, return None.
            if year >= lifetime:
                year = None
        discounted_payback_time = year

    return (
        simple_payback_time,
        discounted_payback_time,
    )


def get_results_path(
        base_name: str,
        scenario_name: str = None
) -> str:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given base name, scenario name and current timestamp. The new subfolder is
    created on disk along with this.

    - Non-alphanumeric characters are removed from `base_name` and `scenario_name`.
    - If is a script file path or `__file__` is passed as `base_name`, the base file name without extension
      will be taken as base name.
    """

    # Preprocess results path name components, including removing non-alphanumeric characters.
    base_name = re.sub(r"\W-+", "", pathlib.Path(base_name).stem) + "_"
    scenario_name = '' if scenario_name is None else re.sub(r'\W+', '', scenario_name) + '_'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    # Obtain results path.
    results_path = cobmo.config.config['paths']['results'] / f'{base_name}{scenario_name}{timestamp}'

    # Instantiate results directory.
    # TODO: Catch error if dir exists.
    os.mkdir(results_path)

    return results_path


def get_alphanumeric_string(
        string: str
):
    """Create lowercase alphanumeric string from given string, replacing non-alphanumeric characters with underscore."""

    return re.sub(r'\W+', '_', string).strip('_').lower()


def launch(path: pathlib.Path):
    """Launch the file at given path with its associated application. If path is a directory, open in file explorer."""

    if not path.exists():
        raise FileNotFoundError(f"Cannot launch file or directory that does not exist: {path}")

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.Popen(['xdg-open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def write_figure_plotly(
        figure: go.Figure,
        results_path: pathlib.Path,
        file_format=cobmo.config.config['plots']['file_format']
):
    """Utility function for writing / storing plotly figure to output file. File format can be given with
    `file_format` keyword argument, otherwise the default is obtained from config parameter `plots/file_format`.

    - `results_path` should be given as file name without file extension, because the file extension is appended
      automatically based on given `file_format`.
    - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'
    """

    if file_format in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(figure, f"{results_path}.{file_format}")
    elif file_format in ['html']:
        pio.write_html(figure, f"{results_path}.{file_format}")
    elif file_format in ['json']:
        pio.write_json(figure, f"{results_path}.{file_format}")
    else:
        logger.error(
            f"Invalid `file_format` for `write_figure_plotly`: {file_format}"
            f" - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'"
        )
        raise ValueError
