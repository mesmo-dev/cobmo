"""Utility functions module."""

from CoolProp.HumidAirProp import HAPropsSI as humid_air_properties
import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pvlib
import re
import seaborn
import subprocess
import sys
import typing

import cobmo.config

logger = cobmo.config.get_logger(__name__)


def starmap(
        function: typing.Callable,
        argument_sequence: typing.List[tuple]
) -> list:
    """Utility function to execute a function for a sequence of arguments, effectively replacing a for-loop.
    Allows running repeated function calls in-parallel, based on Python's `multiprocessing` module.

    - If configuration parameter `run_parallel` is set to True, execution is passed to `starmap`
      of `multiprocessing.Pool`, hence running the function calls in parallel.
    - Otherwise, execution is passed to `itertools.starmap`, which is the non-parallel equivalent.
    """

    if cobmo.config.config['multiprocessing']['run_parallel']:
        if cobmo.config.parallel_pool is None:
            cobmo.config.parallel_pool = cobmo.config.get_parallel_pool()
        results = cobmo.config.parallel_pool.starmap(function, argument_sequence)
    else:
        results = itertools.starmap(function, argument_sequence)

    return results


def calculate_absolute_humidity_humid_air(
        temperature,  # In °C.
        relative_humidity  # In percent.
):
    absolute_humidity = (
        humid_air_properties(
            'W',
            'T', temperature + 273.15,  # In K.
            'R', relative_humidity / 100.0,  # In percent/100.
            'P', 101325.0  # In Pa.
        )
    )
    return absolute_humidity  # In kg(water)/kg(air).


def calculate_enthalpy_humid_air(
        temperature,  # In °C.
        absolute_humidity  # In kg(water)/kg(air).
):
    enthalpy = (
        humid_air_properties(
            'H',
            'T', temperature + 273.15,  # In K.
            'W', absolute_humidity,  # In kg(water)/kg(air).
            'P', 101325.0  # In Pa.
        )
    )
    return enthalpy  # In J/kg.


def calculate_dew_point_enthalpy_humid_air(
        temperature,  # In °C.
        relative_humidity  # In percent.
):
    enthalpy = (
        humid_air_properties(
            'H',
            'T',
            humid_air_properties(
                'D',
                'T', temperature + 273.15,  # In K.
                'R', relative_humidity / 100.0,  # In percent/100.
                'P', 101325.0  # In Pa.
            ),
            'W',
            calculate_absolute_humidity_humid_air(
                temperature,  # In °C.
                relative_humidity  # In percent.
            ),
            'P', 101325.0  # In Pa.
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
            temp_dew=humid_air_properties(
                'D',
                'T', weather_timeseries['ambient_air_temperature'].values + 273.15,
                'W', weather_timeseries['ambient_air_humidity_ratio'].values,
                'P', 101325
            ) - 273.15  # Use CoolProps toolbox to calculate dew point temperature.
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
    - TODO: According to ISO ???
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


def get_battery_parameters(
        battery_parameters_path=(
            os.path.join(cobmo.config.config['paths']['data'], 'supplementary_data', 'storage', 'battery_storage_parameters.csv')
        ),
        cost_conversion_factor=1.37  # USD to SGD (as of October 2019).
):
    """Load battery parameters from CSV, apply `cost_conversion_factor` to cost columns and return as dataframe."""

    # Load battery parameters from CSV.
    battery_parameters = pd.read_csv(battery_parameters_path, index_col=[0, 1, 2])

    # Apply cost conversion factor.
    for column in battery_parameters.columns:
        if 'cost' in column:
            battery_parameters[column] *= cost_conversion_factor

    return battery_parameters


def calculate_discounted_payback_time(
        lifetime,
        investment_cost,
        operation_cost,
        operation_cost_baseline,
        interest_rate=0.06,
        investment_type='',
        save_plots=False,
        results_path=cobmo.config.config['paths']['results'],
        file_id=''
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
                print("Discounted payback time surpassed the technology lifetime.")
                year = None
                break
        discounted_payback_time = year

        # TODO: Move plotting functionality to plots.
        if save_plots:
            # Prepare arrays for plotting.
            if discounted_payback_time is not None:
                plot_array_size = discounted_payback_time
            else:
                plot_array_size = int(np.ceil(lifetime))
            years_array = np.arange(1, plot_array_size + 1)
            investment_cost_array = investment_cost * np.ones(plot_array_size)
            cumulative_discounted_savings = cumulative_discounted_savings[1:plot_array_size + 1]
            annual_discounted_savings = annual_discounted_savings[1:plot_array_size + 1]

            # Define plot settings.
            seaborn.set()
            plt.rcParams['font.serif'] = 'Arial'
            plt.rcParams['font.family'] = 'serif'

            # Create plot.
            (fig, ax) = plt.subplots(1, 1)
            if simple_payback_time is not None:
                ax.scatter(
                    simple_payback_time,
                    investment_cost / 1000.0,
                    marker='o',
                    facecolors='none',
                    edgecolors='r',
                    s=100,
                    label="Simple payback",
                    zorder=10
                )
            ax.plot(
                years_array,
                investment_cost_array / 1000.0,
                linestyle='--',
                color='black',
                alpha=0.7,
                label="Investment"
            )
            ax.plot(
                years_array,
                annual_discounted_savings / 1000.0,
                linestyle='-',
                color='#64BB8E',
                marker='^',
                alpha=1.0,
                label="Annual discounted savings"
            )
            ax.plot(
                years_array,
                cumulative_discounted_savings / 1000.0,
                linestyle='-',
                color='#0074BD',
                marker='s',
                alpha=1.0,
                label="Cumulative discounted savings"
            )

            # Modify plot appearance.
            ax.set_ylabel("Cost in thousand SGD")
            ax.set_xlabel("Time in years")
            ax.set_ylim(bottom=0.0)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.grid(True, which='both')
            ax.grid(which='minor', alpha=0.2)
            fig.legend(loc='center right', fontsize=9)

            # Modify plot title.
            if discounted_payback_time is None:
                title = "Not paying back"
            else:
                title = "Payback year: {:d}".format(discounted_payback_time)
            if 'sensible_thermal_storage' in investment_type:
                title = "Sensible thermal storage | " + title
            elif 'battery_storage' in investment_type:
                title = "Battery storage | " + title
            fig.suptitle(title)

            # Save plot.
            plt.savefig(os.path.join(results_path, 'payback' + file_id + '.svg'))

    return (
        simple_payback_time,
        discounted_payback_time,
    )


def get_alphanumeric_string(
        string: str
):
    """Create lowercase alphanumeric string from given string, replacing non-alphanumeric characters with underscore."""

    return re.sub(r'\W+', '_', string).strip('_').lower()


def get_timestamp(
        time: datetime.datetime = None
) -> str:
    """Generate formatted timestamp string, e.g., for saving results with timestamp."""

    if time is None:
        time = datetime.datetime.now()

    return time.strftime('%Y-%m-%d_%H-%M-%S')


def get_results_path(
        name: str,
) -> str:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given name string and current timestamp. The new subfolder is
    created on disk along with this.
    """

    # Obtain results path.
    results_path = (
        os.path.join(
            cobmo.config.config['paths']['results'],
            # Remove non-alphanumeric characters, except `_`, then append timestamp string.
            re.sub(r'\W+', '', f'{name}_') + cobmo.utils.get_timestamp()
        )
    )

    # Instantiate results directory.
    # TODO: Catch error if dir exists.
    os.mkdir(results_path)

    return results_path


def launch(path):
    """Launch the file at given path with its associated application. If path is a directory, open in file explorer."""

    try:
        assert os.path.exists(path)
    except AssertionError:
        logger.error(f'Cannot launch file or directory that does not exist: {path}')

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.Popen(['xdg-open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def write_figure_plotly(
        figure: go.Figure,
        results_path: str,
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
