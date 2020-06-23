import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn import linear_model, metrics
from statsmodels.tsa.arima.model import ARIMA
import forecast.format_data


def main(df=None, start_timestep=None, end_timestep=None):
    if df is None:
        df = forecast.format_data.main()
    if (start_timestep is not None) and (end_timestep is not None):
        df = df.loc[df['timestep'] < start_timestep]
        no_timesteps = int((end_timestep-start_timestep)/dt.timedelta(minutes=30) + 1)

    # Long-term trend removal: calculate annual averages of the log electricity prices.
    mean_by_year = df.groupby('YEAR')['log_price'].mean()
    x = mean_by_year.index.values.reshape(-1, 1)
    y = mean_by_year.values
    lm_lt = obtain_linear_model(x, y)
    df['log_price_no_ltt'] = df['log_price'] - lm_lt.predict(df['YEAR'].values.reshape(-1, 1))

    # Weekly cycle removal: absolute sinusoidal function.
    df['period_of_week'] = df.apply(lambda x: x['DATE'].weekday() * 48 + x['PERIOD'], axis=1)
    mean_by_pow = df.groupby('period_of_week')['log_price_no_ltt'].mean()
    min_period = mean_by_pow.index[mean_by_pow == mean_by_pow.min()][0]
    df['t'] = df.index.values + 1
    df['sin_func'] = df['t'].apply(calculate_absolute_sin_function, args=(min_period,))
    x = df['sin_func'].values.reshape(-1, 1)
    y = df['log_price_no_ltt']
    lm_week = obtain_linear_model(x, y)
    df['log_price_no_ltt_no_wt'] = df['log_price_no_ltt'] - lm_week.predict(
        df['sin_func'].values.reshape(-1, 1))

    # Daily cycle removal: calculate daily averages of the log electricity prices.
    mean_by_dow = df.groupby('DAY_OF_WEEK')['log_price_no_ltt_no_wt'].mean()
    df['residuals'] = df.apply(lambda x: x['log_price_no_ltt_no_wt'] - mean_by_dow[x['DAY_OF_WEEK']],
                                             axis=1)

    # Construct forecast model
    model = ARIMA(df['residuals'], order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast future prices
    # date_forecast = df.iloc[-1,0].date() + dt.timedelta(days=1)
    date_forecast = start_timestep.date()
    forecast_prices = obtain_forecast_prices(model_fit, date_forecast, lm_lt, lm_week, mean_by_dow, steps=no_timesteps)
    lower_prices = obtain_forecast_prices(model_fit, date_forecast, lm_lt, lm_week, mean_by_dow, mode='lower', steps=no_timesteps)
    upper_prices = obtain_forecast_prices(model_fit, date_forecast, lm_lt, lm_week, mean_by_dow, mode='upper', steps=no_timesteps)
    forecast_df = pd.concat([forecast_prices, upper_prices, lower_prices], axis=1)
    forecast_df.rename(
        {0: 'expected_price', 'upper residuals': 'upper_limit', 'lower residuals': 'lower_limit'},
        axis=1,
        inplace=True
        )
    return forecast_df, df


def obtain_forecast_prices(model, date, lt_model, week_model, daily_mean, steps=49, mode='expected'):
    if mode == 'expected':
        residuals = model.forecast(steps)
    elif mode == 'lower':
        residuals = model.get_forecast(steps).conf_int()['lower residuals']
    else:
        residuals = model.get_forecast(steps).conf_int()['upper residuals']
    return np.exp(residuals
                  + lt_model.predict(np.array([2020] * steps).reshape(-1, 1))
                  + week_model.predict(pd.Series(residuals.index.values).apply(
                                calculate_absolute_sin_function, args=(9,)).values.reshape(-1,1))
                  + daily_mean[date.weekday()]
                  )


def obtain_linear_model(x, y):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model


def calculate_absolute_sin_function(t, min_period):
    phase_shift = (193-min_period-1)/336*math.pi
    return np.abs(np.sin(math.pi * t / 336 + phase_shift))

if __name__ == '__main__':
    main()