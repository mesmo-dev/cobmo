import forecast.forecast_model
import datetime as dt
import cobmo.config
import os
import pandas as pd

import cobmo.utils


def main():

    results_path = cobmo.utils.get_results_path(__file__)
    price_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'clearing_price')

    forecast_model = forecast.forecast_model.forecastModel()
    init_timestep = forecast_model.df['timestep'].iloc[-1]
    timesteps = pd.date_range(init_timestep + dt.timedelta(minutes=30), periods=49, freq='30T')
    clearing_prices = pd.read_csv(os.path.join(price_data_path, 'Jan_2020.csv'), index_col=0)
    clearing_prices.index = timesteps
    clearing_prices['expected'] = 0.0
    clearing_prices['upper_limit'] = 0.0
    clearing_prices['lower_limit'] = 0.0
    for timestep in timesteps:
        forecast_df = forecast_model.forecast_prices()
        forecast_df.index = pd.date_range(timestep, periods=48, freq='30T')
        forecast_df.to_csv(os.path.join(results_path, f'forecast_{timestep}.csv'.replace(':', '_')))
        clearing_prices.loc[timestep, 'expected'] = forecast_df.loc[timestep, 'expected_price']*1000
        clearing_prices.loc[timestep, 'upper_limit'] = forecast_df.loc[timestep, 'upper_limit'] * 1000
        clearing_prices.loc[timestep, 'lower_limit'] = forecast_df.loc[timestep, 'lower_limit'] * 1000
        forecast_model.update_model(clearing_prices.loc[timestep, 'clearing_price'], timestep)

    clearing_prices.to_csv(os.path.join(results_path, 'summary.csv'))


if __name__ == '__main__':
    main()