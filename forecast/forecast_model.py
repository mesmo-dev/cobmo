import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn import linear_model, metrics
from statsmodels.tsa.arima.model import ARIMA
import forecast.format_data


class forecastModel(object):

    def __init__(self, df=None, start_timestep=None):
        self.expected_prices = None
        self.lower_prices = None
        self.upper_prices = None

        if df is None:
            self.df = forecast.format_data.main()

        if start_timestep is not None:
            self.df = self.df.loc[self.df['timestep'] < start_timestep]

        # Long-term trend removal: calculate annual averages of the log electricity prices.
        mean_by_year = self.df.groupby('YEAR')['log_price'].mean()
        x = mean_by_year.index.values.reshape(-1, 1)
        y = mean_by_year.values
        self.lm_lt = obtain_linear_model(x, y)
        self.df['log_price_no_ltt'] = self.df['log_price'] - self.lm_lt.predict(self.df['YEAR'].values.reshape(-1, 1))

        # Weekly cycle removal: absolute sinusoidal function.
        self.df['period_of_week'] = self.df.apply(lambda x: x['DATE'].weekday() * 48 + x['PERIOD'], axis=1)
        mean_by_pow = self.df.groupby('period_of_week')['log_price_no_ltt'].mean()
        self.min_period = mean_by_pow.index[mean_by_pow == mean_by_pow.min()][0]
        self.df['t'] = self.df.index.values + 1
        self.df['sin_func'] = self.df['t'].apply(calculate_absolute_sin_function, args=(self.min_period,))
        x = self.df['sin_func'].values.reshape(-1, 1)
        y = self.df['log_price_no_ltt']
        self.lm_week = obtain_linear_model(x, y)
        self.df['log_price_no_ltt_no_wt'] = self.df['log_price_no_ltt'] - self.lm_week.predict(
            self.df['sin_func'].values.reshape(-1, 1))

        # Daily cycle removal: calculate daily averages of the log electricity prices.
        self.mean_by_dow = self.df.groupby('DAY_OF_WEEK')['log_price_no_ltt_no_wt'].mean()
        self.df['residuals'] = self.df.apply(lambda x: x['log_price_no_ltt_no_wt'] - self.mean_by_dow[x['DAY_OF_WEEK']],
                                             axis=1)

        # Construct forecast model
        self.model = ARIMA(self.df['residuals'], order=(1, 1, 1))
        self.model_fit = self.model.fit()

    def update_model(self, new_price, timestep):
        new_df = pd.DataFrame(0, index=[self.df.index[-1]+1], columns=self.df.columns)
        new_df['USEP ($/MWh)'] = new_price
        new_df['timestep'] = timestep
        new_df['YEAR'] = timestep.date().year
        new_df['t'] = new_df.index.values + 1
        new_df['sin_func'] = new_df['t'].apply(calculate_absolute_sin_function, args=(self.min_period,))
        new_df['DAY_OF_WEEK'] = timestep.date().weekday()
        new_df['log_price'] = math.log(new_price)

        # Concatenate new data into the existing DataFrame
        self.df = pd.concat((self.df, new_df))

        self.df['log_price_no_ltt'] = self.df['log_price'] - self.lm_lt.predict(self.df['YEAR'].values.reshape(-1, 1))
        self.df['log_price_no_ltt_no_wt'] = self.df['log_price_no_ltt'] - self.lm_week.predict(
            self.df['sin_func'].values.reshape(-1, 1))
        self.df['residuals'] = self.df.apply(lambda x: x['log_price_no_ltt_no_wt'] - self.mean_by_dow[x['DAY_OF_WEEK']],
                                             axis=1)

        # Construct forecast model
        self.model = ARIMA(self.df['residuals'], order=(1, 1, 1))
        self.model_fit = self.model.fit()

    def forecast_prices(self, steps=48):
        forecast_date = (self.df['timestep'].iloc[-1] + dt.timedelta(minutes=30)).date()
        self.expected_prices = obtain_forecast_prices(self.model_fit, forecast_date, self.lm_lt, self.lm_week, self.mean_by_dow,
                                                      self.min_period, steps=steps)
        self.lower_prices = obtain_forecast_prices(self.model_fit, forecast_date, self.lm_lt, self.lm_week, self.mean_by_dow,
                                                   self.min_period, mode='lower', steps=steps)
        self.upper_prices = obtain_forecast_prices(self.model_fit, forecast_date, self.lm_lt, self.lm_week, self.mean_by_dow,
                                                   self.min_period, mode='upper', steps=steps)
        forecast_df = pd.concat([self.expected_prices, self.upper_prices, self.lower_prices], axis=1)
        forecast_df.rename(
            {0: 'expected_price', 'upper residuals': 'upper_limit', 'lower residuals': 'lower_limit'},
            axis=1,
            inplace=True
        )
        return forecast_df


def obtain_forecast_prices(model, date, lt_model, week_model, daily_mean, min_period, steps=49, mode='expected'):
    if mode == 'expected':
        residuals = model.forecast(steps)
    elif mode == 'lower':
        residuals = model.get_forecast(steps).conf_int()['lower residuals']
    else:
        residuals = model.get_forecast(steps).conf_int()['upper residuals']
    return np.exp(residuals
                  + lt_model.predict(np.array([2020] * steps).reshape(-1, 1))
                  + week_model.predict(pd.Series(residuals.index.values).apply(
                                calculate_absolute_sin_function, args=(min_period,)).values.reshape(-1,1))
                  + daily_mean[date.weekday()]
                  )


def obtain_linear_model(x, y):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model


def calculate_absolute_sin_function(t, min_period):
    phase_shift = (193-min_period-1)/336*math.pi
    return np.abs(np.sin(math.pi * t / 336 + phase_shift))