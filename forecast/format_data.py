import pandas as pd
import numpy as np
import forecast.load_data
import datetime as dt
import math


def main():
    df = forecast.load_data.main()
    df.loc[df['USEP ($/MWh)'].isna(), 'USEP ($/MWh)'] = \
        df.loc[df['USEP ($/MWh)'].isna(), 'PRICE ($/MWh)']
    df.drop(['PRICE ($/MWh)', 'LCP ($/MWh)', 'TCL (MW)', 'INFORMATION TYPE', 'PRICE TYPE'], axis=1, inplace=True)
    df['DATE'] = df['DATE'].apply(lambda x: dt.datetime.strptime(x, '%d %b %Y'))
    df.sort_values(['DATE', 'PERIOD'], ignore_index=True, inplace=True)
    df['timestep'] = df['DATE'] + (df['PERIOD']-1) * dt.timedelta(minutes=30)
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    df['DAY_OF_WEEK'] = df['DATE'].dt.weekday
    df['log_price'] = df['USEP ($/MWh)'].apply(get_log)
    return df


def get_log(value):
    try:
        return math.log(value)
    except:
        return math.log(30)


if __name__ == '__main__':
    main()