import os
import pandas as pd
import cobmo.config


def main():
    forecast_path = os.path.join(cobmo.config.base_path, 'forecast')
    data_path = os.path.join(forecast_path, 'data')
    file_list = os.listdir(data_path)
    df = pd.DataFrame()
    for file in file_list:
        temp_df = pd.read_csv(os.path.join(data_path, file))
        df = pd.concat([df, temp_df], ignore_index=True)
    return df


if __name__ == '__main__':
    main()