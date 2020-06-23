import forecast.forecast_model
import datetime as dt
import cobmo.config
import os
import pandas as pd

import cobmo.utils


def main():

    results_path = cobmo.utils.get_results_path(f'run_forecast_model')
    price_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'clearing_price')

    forecast_model = forecast.forecast_model.forecastModel()
    timestep = forecast_model.df['timestep'].iloc[-1]
    clearing_prices = pd.read_csv(os.path.join(price_data_path, 'Jan_2020.csv'), index_col=0)
    for price in clearing_prices['clearing_price']:
        forecast_df = forecast_model.forecast_prices()
        timestep += dt.timedelta(minutes=30)
        forecast_df.index = pd.date_range(timestep, periods=48, freq='30T')
        forecast_df.to_csv(os.path.join(results_path, f'forecast_{timestep}.csv'.replace(':', '_')))
        forecast_model.update_model(price, timestep)


if __name__ == '__main__':
    main()