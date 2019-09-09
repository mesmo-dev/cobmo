"""
Building model utility function definitions
"""

import os
import sqlite3
import csv
from math import fabs
import glob
import pandas as pd
import pvlib
# Using CoolProp for calculating humid air properties: http://www.coolprop.org/fluid_properties/HumidAir.html
from CoolProp.HumidAirProp import HAPropsSI as humid_air_properties
# Import for infeasibility analysis
from pyomo.core import Constraint, Var, value, TraversalStrategy
import logging
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

""" 
Module with diagnostic utilities for infeasible models. 
>>> https://github.com/Pyomo/pyomo/blob/master/pyomo/util/infeasible.py 
"""
logger = logging.getLogger('pyomo.util.infeasible')
logger.setLevel(logging.INFO)


def retrieve_battery_parameters():
    cobmo_path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
    data_path = os.path.join(cobmo_path, 'data')
    storage_data_path = os.path.join(data_path, 'storage_data/')
    file_battery_parameters = storage_data_path + 'battery_storage_types.csv'

    battery_params = pd.read_csv(file_battery_parameters, index_col=0)
    columns = battery_params.columns

    # 2016
    bool_2016 = pd.Series(columns.str.contains('2016'))
    columns_2016 = pd.Index(
        pd.concat(
            [
                pd.Series('round_trip_efficiency'),
                pd.Series('depth_of_discharge'),
                pd.Series(columns[bool_2016])
            ]
        )
    )
    battery_params_2016 = battery_params.loc[:, columns_2016]

    # 2020
    bool_2020 = pd.Series(columns.str.contains('2020'))
    columns_2020 = pd.Index(
        pd.concat(
            [
                pd.Series('round_trip_efficiency'),
                pd.Series('depth_of_discharge'),
                pd.Series(columns[bool_2020])
            ]
        )
    )
    battery_params_2020 = battery_params.loc[:, columns_2020]

    # 2025
    bool_2025 = pd.Series(columns.str.contains('2025'))
    columns_2025 = pd.Index(
        pd.concat(
            [
                pd.Series('round_trip_efficiency'),
                pd.Series('depth_of_discharge'),
                pd.Series(columns[bool_2025])
            ]
        )
    )
    battery_params_2025 = battery_params.loc[:, columns_2025]

    # 2030
    bool_2030 = pd.Series(columns.str.contains('2030'))
    columns_2030 = pd.Index(
        pd.concat(
            [
                pd.Series('round_trip_efficiency'),
                pd.Series('depth_of_discharge'),
                pd.Series(columns[bool_2030])
            ]
        )
    )
    battery_params_2030 = battery_params.loc[:, columns_2030]

    # Converting values from USD to SGD. Change rate of 1.39
    # 2016
    usd2sgd = 1.39
    battery_params_2016.loc[:, 'energy_installation_cost_2016'] = (
            battery_params_2016.loc[:, 'energy_installation_cost_2016'] * usd2sgd
    )
    battery_params_2016.loc[:, 'power_installation_cost_2016'] = (
            battery_params_2016.loc[:, 'power_installation_cost_2016'] * usd2sgd
    )

    # 2020
    battery_params_2020.loc[:, 'energy_installation_cost_2020'] = (
            battery_params_2020.loc[:, 'energy_installation_cost_2020'] * usd2sgd
    )
    battery_params_2020.loc[:, 'power_installation_cost_2020'] = (
            battery_params_2020.loc[:, 'power_installation_cost_2020'] * usd2sgd
    )

    # 2025
    battery_params_2025.loc[:, 'energy_installation_cost_2025'] = (
            battery_params_2025.loc[:, 'energy_installation_cost_2025'] * usd2sgd
    )
    battery_params_2025.loc[:, 'power_installation_cost_2025'] = (
            battery_params_2025.loc[:, 'power_installation_cost_2025'] * usd2sgd
    )

    # 2030
    battery_params_2030.loc[:, 'energy_installation_cost_2030'] = (
            battery_params_2030.loc[:, 'energy_installation_cost_2030'] * usd2sgd
    )
    battery_params_2030.loc[:, 'power_installation_cost_2030'] = (
            battery_params_2030.loc[:, 'power_installation_cost_2030'] * usd2sgd
    )

    # Energy, power and lifetime dataframes
    energy_cost = battery_params.loc[:, columns[columns.str.contains('energy')]]
    power_cost = battery_params.loc[:, columns[columns.str.contains('power')]]
    lifetime = battery_params.loc[:, columns[columns.str.contains('lifetime')]]
    efficiency_dod = battery_params.iloc[:, [0, 1]]

    return (
        battery_params_2016,
        battery_params_2020,
        battery_params_2025,
        battery_params_2030,
        energy_cost,
        power_cost,
        lifetime,
        efficiency_dod
    )


def plot_params(
        building,
        param_name='',
        readfile_path=''
):
    sns.set()
    pricing_method = building.electricity_prices['price_type'][0]
    prices = building.electricity_prices['price'].values
    print(pricing_method)
    timesteps = building.electricity_prices.index
    print(timesteps)

    fig, pr = plt.subplots(1, 1)
    pr.plot(timesteps, prices)
    plt.show()

def discounted_payback_time(
        building,
        storage_size,
        savings_day,
        save_plot_on_off,
        save_path,
        plotting_on_off,
        storage,
        pricing_method,
        interest_rate=0.06
):
    # Activating seaborn theme
    sns.set()

    # Storage characteristics
    storage_investment_per_unit = building.building_scenarios['storage_investment_sgd_per_unit'][0]
    SGD_per_X = building.building_scenarios['investment_sgd_per_X'][0]
    storage_lifetime = building.building_scenarios['storage_lifetime'][0]

    # DISCOUNTED PAYBACK
    start_date = dt.date(2019, 1, 1)
    end_date = dt.date(2019, 12, 31)
    working_days = np.busday_count(start_date, end_date)

    pvaf = (1 - (1 + interest_rate) ** (-storage_lifetime)) / interest_rate  # Present value Annuity factor

    rt_efficiency = building.building_scenarios['storage_round_trip_efficiency'][0]
    economic_horizon = 1000
    cumulative_discounted_savings = np.zeros(economic_horizon)
    yearly_discounted_savings = np.zeros(economic_horizon)

    savings_one_year = 0.0
    investment_cost = 0.0
    simple_payback_time = 0.0
    if savings_day != 0.0:
        savings_one_year = savings_day * working_days
        investment_cost = float(storage_size) * float(storage_investment_per_unit)
        simple_payback_time = np.ceil(investment_cost / savings_one_year)

    year = 0
    while cumulative_discounted_savings[year] < investment_cost:
        year += 1  # increment defined here to end the while at the right year (instead of 1 year more)
        discount_factor = (1 + interest_rate) ** (-year)
        yearly_discounted_savings[year] = savings_one_year * discount_factor
        cumulative_discounted_savings[year] = cumulative_discounted_savings[year - 1] + yearly_discounted_savings[year]
        # print("\nat year %i the cumulative is >> %.2f" % (year, cumulative_discounted_savings[year]))
        if year == 70:
            print('\nDISCOUNTED PAYBACK IS TOO HIGH! reached 70 years')
            year = 0
            break

    discounted_payback = year
    years_array = np.arange(1, discounted_payback + 1)
    investment_cost_array = investment_cost * np.ones(discounted_payback)  # array with constant value = investment
    cumulative_discounted_savings = cumulative_discounted_savings[1:discounted_payback + 1]
    yearly_discounted_savings = yearly_discounted_savings[1:discounted_payback + 1]
    # discounted_total_savings_at_payback = cumulative_discounted_savings[-1]


    payback_df = pd.DataFrame(
        np.column_stack(
            (
                years_array,
                investment_cost_array,
                cumulative_discounted_savings,
                yearly_discounted_savings

            )
        )
    )

    if plotting_on_off == 1:
        # Plotting
        # Change default font of plots
        # (http://jonathansoma.com/lede/data-studio/matplotlib/changing-fonts-in-matplotlib/)
        # usable fonts: Control Panel\Appearance and Personalization\Fonts
        plt.rcParams['font.serif'] = "Palatino Linotype"
        plt.rcParams['font.family'] = "serif"
        date_main = datetime.datetime.now()

        fig, pb = plt.subplots(1, 1)

        pb.axvline(x=storage_lifetime, color='b', linestyle='--', alpha=0.5, label='Storage Lifetime = %i' % int(storage_lifetime))
        pb.scatter(simple_payback_time, investment_cost, marker='o', facecolors='none', edgecolors='r', s=100,
                   label='Simple payback = %.0f' % simple_payback_time, zorder=10)
        pb.plot(years_array, investment_cost_array, linestyle='--', color='black', alpha=0.7,
                label='Investment')
        pb.plot(years_array, yearly_discounted_savings, linestyle='-', color='#64BB8E', marker='^', alpha=1.0,
                label='Yearly Disc. Savings')
        pb.plot(years_array, cumulative_discounted_savings, linestyle='-', color='#0074BD', marker='s', alpha=1.0,
                label='Cumulative Disc. Savings')

        pb.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        pb.set_ylabel('SGD')
        pb.set_xlabel('year')
        # pb.xaxis.set_major_locator(MultipleLocator(5))
        pb.xaxis.set_major_formatter(FormatStrFormatter('%i'))
        pb.xaxis.set_minor_locator(MultipleLocator(1))

        if discounted_payback != 0:
            major_ticks = np.arange(years_array[0], years_array[-1]+1, 1)
            # minor_ticks = np.arange(years_array[0], years_array[-1], 0.2)
            pb.set_xticks(major_ticks)
            # pb.set_xticks(minor_ticks, minor=True)
            # pb.set_yticks(major_ticks)
            # pb.set_yticks(minor_ticks, minor=True)

        # fig.legend(loc='center right', fontsize=9)

        pb.set_ylim([0.0, 20000.0])

        pb.grid(True, which='both')
        pb.grid(which='minor', alpha=0.2)
        # pb.grid(which='major', alpha=0.5)

        # pb.text(
        #     1, investment_cost*0.7,  # *2.5/4
        #     'storage tech = %s'
        #     '\ninterest rate = %.2f'
        #     '\nstorage size = %.1f'
        #     '\nefficiency = %.2f'
        #     '\nsavings/year = %.1f SGD'
        #     '\nstorage investment= %.1f SGD/%s'
        #     % (
        #         building.building_scenarios['storage_technology'][0], interest_rate, storage_size, float(rt_efficiency),
        #         savings_one_year, float(storage_investment_per_unit), SGD_per_X
        #        ),
        #     # style='italic',
        #     fontsize=9,
        #     bbox={'facecolor': '#0074BD', 'alpha': 0.2, 'pad': 5})

        title = 'Neither Sensible nor battery'
        filename = 'no_tech_specified'
        if storage == 'sensible':
            if year == 70.0:
                title = 'Sensible Thermal Storage — ' + pricing_method + ' — Not paying back'
            else:
                title = 'Sensible Thermal Storage — ' + pricing_method + ' — payback year = %i' % discounted_payback
            filename = ('discounted_payback_STES'  # + building.building_scenarios['building_name'][0]
                        + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                            date_main.year, date_main.month, date_main.day,
                            date_main.hour, date_main.minute, date_main.second)
                        )
        elif storage == 'battery':
            if year == 70.0:
                title = 'Battery Storage — ' + pricing_method + ' — Not paying back'
            else:
                title = 'Battery Storage — ' + pricing_method + ' — payback year = %i' % discounted_payback
            filename = ('discounted_payback_BES'  # + building.building_scenarios['building_name'][0]
                        + '_{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}'.format(
                            date_main.year, date_main.month, date_main.day,
                            date_main.hour, date_main.minute, date_main.second)
                        )

        fig.suptitle(title)

        fig.set_size_inches([7, 5])
        if save_plot_on_off == 1:
            plt.savefig(save_path + filename + '.svg', format='svg', dpi=1200)
            # plt.savefig('figs/discounted_payback.svg', format='svg', dpi=1200)

        plt.show()

    return (
        simple_payback_time,
        discounted_payback,
    )


def log_infeasible_constraints(m, tol=1E-6, logger=logger):
    """Print the infeasible constraints in the model.
    Uses the current model state. Uses pyomo.util.infeasible logger unless one
    is provided.
    Args:
        m (Block): Pyomo block or model to check
        tol (float): feasibility tolerance
    """
    for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        # if constraint is an equality, handle differently
        if constr.equality and fabs(value(constr.lower - constr.body)) >= tol:
            logger.info('CONSTR {}: {} != {}'.format(
                constr.name, value(constr.body), value(constr.lower)))
            continue
        # otherwise, check LB and UB, if they exist
        if constr.has_lb() and value(constr.lower - constr.body) >= tol:
            logger.info('CONSTR {}: {} < {}'.format(
                constr.name, value(constr.body), value(constr.lower)))
        if constr.has_ub() and value(constr.body - constr.upper) >= tol:
            logger.info('CONSTR {}: {} > {}'.format(
                constr.name, value(constr.body), value(constr.upper)))


def log_infeasible_bounds(m, tol=1E-6, logger=logger):
    """Print the infeasible variable bounds in the model.
    Args:
        m (Block): Pyomo block or model to check
        tol (float): feasibility tolerance
    """
    for var in m.component_data_objects(
            ctype=Var, descend_into=True):
        if var.has_lb() and value(var.lb - var) >= tol:
            logger.info('VAR {}: {} < LB {}'.format(
                var.name, value(var), value(var.lb)))
        if var.has_ub() and value(var - var.ub) >= tol:
            logger.info('VAR {}: {} > UB {}'.format(
                var.name, value(var), value(var.ub)))


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


def tank_geometry(
        volume,
        aspect_ratio
):
    diameter = (volume / aspect_ratio * 4 / 3.14159265359) ** (1 / 3)
    height = diameter * aspect_ratio

    return (
        diameter,
        height
    )


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
        index=expected_timeseries.index,
        columns=expected_timeseries.columns
    )
    error_summary = pd.DataFrame(
        0.0,
        index=pd.Index(['mean_absolute_error', 'root_mean_squared_error'], name='error_type'),
        columns=expected_timeseries.columns
    )

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
