"""
Building model utility function definitions
"""

import os
import sqlite3
import csv
import glob
import pandas as pd
import pvlib
# Using CoolProp for calculating humid air properties: http://www.coolprop.org/fluid_properties/HumidAir.html
from CoolProp.HumidAirProp import HAPropsSI as humid_air_properties


def create_database(
        sqlite_path,
        sql_path,
        csv_path
):
    """
    Create SQLITE database from SQL (schema) file and CSV files
    """
    # Connect SQLITE database (creates file, if none)
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Remove old data, if any
    cursor.executescript("""
        PRAGMA writable_schema = 1;
        DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger');
        PRAGMA writable_schema = 0;
        VACUUM;
        """)

    # Recreate SQLITE database (schema) from SQL file
    cursor.executescript(open(sql_path, 'r').read())
    conn.commit()

    # Import CSV files into SQLITE database
    conn.text_factory = str  # allows utf-8 data to be stored
    cursor = conn.cursor()
    for file in glob.glob(os.path.join(csv_path, '*.csv')):
        table_name = os.path.splitext(os.path.basename(file))[0]

        with open(file, 'r') as file:
            first_row = True
            for row in csv.reader(file):
                if first_row:
                    cursor.execute("delete from {}".format(table_name))
                    insert_sql_query = \
                        "insert into {} VALUES ({})".format(table_name, ', '.join(['?' for column in row]))

                    first_row = False
                else:
                    cursor.execute(insert_sql_query, row)
            conn.commit()
    cursor.close()
    conn.close()


def calculate_irradiation_surfaces(
        conn,
        weather_type='singapore_nus',
        irradiation_model='dirint'
):
    """ Calculates irradiation for surfaces oriented towards east, south, west & north.

    - Operates on the database: Updates according columns in weather_timeseries
    - Takes irradition_horizontal as measured global horizontal irradiation (ghi)
    - Based on pvlib-python toolbox: https://github.com/pvlib/pvlib-python
    """

    # Load weather data from database
    weather_types = pd.read_sql(
        """
        select * from weather_types 
        where weather_type='{}'
        """.format(weather_type),
        conn
    )
    weather_timeseries = pd.read_sql(
        """
        select * from weather_timeseries 
        where weather_type='{}'
        """.format(weather_type),
        conn
    )

    # Set time zone (required for pvlib solar position calculations)
    weather_timeseries.index = pd.to_datetime(weather_timeseries['time'])
    weather_timeseries.index = weather_timeseries.index.tz_localize(weather_types['time_zone'][0])

    # Extract global horizontal irradiation (ghi) from weather data
    irradiation_ghi = weather_timeseries['irradiation_horizontal']

    # Calculate solarposition (zenith, azimuth)
    solarposition = pvlib.solarposition.get_solarposition(
        time=weather_timeseries.index,
        latitude=weather_types['latitude'][0],
        longitude=weather_types['longitude'][0]
    )

    # Calculate direct normal irradiation (dni) from global horizontal irradiation (ghi)
    irradiation_dni = pd.Series(index=weather_timeseries.index)
    if irradiation_model == 'disc':
        # ... via DISC model
        irradiation_disc = pvlib.irradiance.disc(
            ghi=irradiation_ghi,
            solar_zenith=solarposition['zenith'],
            datetime_or_doy=weather_timeseries.index
        )
        irradiation_dni = irradiation_disc['dni']
    elif irradiation_model == 'erbs':
        # ... via ERBS model
        irradiation_erbs = pvlib.irradiance.erbs(
            ghi=irradiation_ghi,
            zenith=solarposition['zenith'],
            doy=weather_timeseries.index
        )
        irradiation_dni = irradiation_erbs['dni']
    elif irradiation_model == 'dirint':
        # ... via DIRINT model
        irradiation_dirint = pvlib.irradiance.dirint(
            ghi=irradiation_ghi,
            solar_zenith=solarposition['zenith'],
            times=weather_timeseries.index,
            temp_dew=humid_air_properties(
                'D',
                'T', weather_timeseries['ambient_air_temperature'].values + 273.15,
                'W', weather_timeseries['ambient_air_humidity_ratio'].values,
                'P', 101325
            ) - 273.15  # Use CoolProps toolbox to calculate dew point temperature
        )
        irradiation_dni = irradiation_dirint

    # Replace NaNs (NaN means no irradiation)
    irradiation_dni.loc[irradiation_dni.isna()] = 0.0

    # Calculate diffuse horizontal irradiation (dhi)
    irradiation_dhi = pd.Series(
            irradiation_ghi
            - irradiation_dni
            * pvlib.tools.cosd(solarposition['zenith']),
    )

    # Define surface orientations
    surface_orientations = pd.DataFrame(
        data=[0.0, 90.0, 180.0, 270.0],
        index=['north', 'east', 'south', 'west'],
        columns=['surface_azimuth']
    )

    # Calculate irradiation onto each surface
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

    # Update weather_timeseries in database
    conn.cursor().execute(
        """
        delete from weather_timeseries 
        where weather_type='{}'
        """.format(weather_type),
    )
    weather_timeseries.to_sql(
        'weather_timeseries',
        conn,
        if_exists='append',
        index=False
    )

def calculate_sky_temperature(conn, weather_type='singapore_nus'):
    """
    - Calculates sky temperatures from ambient air temperature for tropical weather
    - ambient air temperature is decreased by 11K to get the sky temperature
    """
    # Load weather data
    weather_types = pd.read_sql(
        """
        select * from weather_types 
        where weather_type='{}'
        """.format(weather_type),
        conn
    )
    weather_timeseries = pd.read_sql(
        """
        select * from weather_timeseries 
        where weather_type='{}'
        """.format(weather_type),
        conn
    )
    weather_timeseries.index = pd.to_datetime(weather_timeseries['time'])

    # Get temperature difference between sky and ambient
    temperature_difference = weather_types['temperature_difference_sky_ambient'][0]

    # Calculate sky temperature
    weather_timeseries.loc[:, 'sky_temperature'] = \
        weather_timeseries.loc[:, 'ambient_air_temperature'] - temperature_difference

    # Update weather_timeseries in database
    conn.cursor().execute(
        """
        delete from weather_timeseries 
        where weather_type='{}'
        """.format(weather_type),
    )

    weather_timeseries.to_sql('weather_timeseries', conn, if_exists='append', index=False)


def calculate_error(
        expected_timeseries=pd.DataFrame(),
        predicted_timeseries=pd.DataFrame()
):
    """Computes the error between expected and predicted timeseries dataframes.

    - Note: This function doesn't check if the data format is valid.
    """
    error_timeseries = pd.DataFrame(
        0.0,
        expected_timeseries.index,
        expected_timeseries.columns
    )
    for index, row in expected_timeseries.iterrows():
        error_timeseries.loc[index, :] = (
            predicted_timeseries.loc[index, :]
            - expected_timeseries.loc[index, :]
        ).abs()

    error_mean = error_timeseries.mean()

    return (
        error_mean,
        error_timeseries
    )
