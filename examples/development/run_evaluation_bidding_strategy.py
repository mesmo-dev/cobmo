"""Run script for formulating bidding strategies."""

import numpy as np
import os
import pandas as pd
import random
import datetime as dt

import cobmo.building_model
import cobmo.config
import cobmo.optimization_problem
import cobmo.data_interface
import cobmo.utils

import forecast.build_model, forecast.forecast_model


def main():

    # Settings.
    scenario_name = '43755562'
    results_path = cobmo.utils.get_results_path(f'run_evaluation_bidding_strategy_{scenario_name}')
    price_data_path = os.path.join(cobmo.config.config['paths']['supplementary_data'], 'clearing_price')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    cobmo.data_interface.recreate_database()

    # Obtain building model.
    building = cobmo.building_model.BuildingModel(scenario_name)
    timesteps = building.timesteps

    # Obtain price forecast.
    forecast_model = forecast.forecast_model.forecastModel()#start_timestep=timesteps[0])
    price_forecast = forecast_model.forecast_prices(steps=len(timesteps))
    price_forecast.index = timesteps
    forecast_timestep = forecast_model.df['timestep'].iloc[-1]

    # Create a placeholder DataFrame to store actual dispatch quantities
    actual_dispatch = pd.DataFrame(0, timesteps, ['clearing_price', 'actual_dispatch'])

    # Load actual clearing prices
    clearing_prices = pd.read_csv(os.path.join(price_data_path, 'Jan_2020.csv'), index_col=0)
    print(clearing_prices.index)
    actual_dispatch['clearing_price'] = clearing_prices['clearing_price'].values
    # actual_dispatch['clearing_price'] = price_forecast['expected_price'].copy()

    def determine_dispatch_quantity(bids, actual_price):
        if actual_price < bids.index[0]:
            dispatch_quantity = bids.loc[bids.index[0], 'P']
        elif actual_price > bids.index[-1]:
            dispatch_quantity = bids.loc[bids.index[-1], 'P']
        for i in range(len(bids) - 1):
            if bids.index[i] <= actual_price <= bids.index[i + 1]:
                price_ceiling = bids.index[i + 1]
                price_floor = bids.index[i]
                # dispatch_quantity = bids.loc[price_floor, 'P'] + (actual_price-price_floor)/(price_ceiling-price_floor)*(bids.loc[price_ceiling, 'P']-bids.loc[price_floor, 'P'])
                dispatch_quantity = bids.loc[price_ceiling, 'P']
        return dispatch_quantity  # Return minimum consumption if price is higher than upper bound

    # Obtain and solve baseline optimization problem.
    baseline_problem = cobmo.optimization_problem.OptimizationProblem(
        building,
        problem_type='load_minimization'
    )
    (
        control_vector_baseline,
        state_vector_baseline,
        output_vector_baseline,
        baseline_operation_cost,
        baseline_investment_cost,  # Zero when running (default) operation problem.
        baseline_storage_size  # Zero when running (default) operation problem.
    ) = baseline_problem.solve()

    actual_dispatch['actual_dispatch_flat'] = output_vector_baseline['grid_electric_power'].copy()
    output_vector_baseline.to_csv(os.path.join(results_path, 'output_vector_baseline.csv'))

    # Calculate baseline daily cost
    daily_cost_baseline = (
        actual_dispatch['clearing_price'] * actual_dispatch['actual_dispatch_flat'].sum()
        * building.timestep_interval.seconds / 3600 / 1e6
    )
    print(f'Daily cost (baseline): {daily_cost_baseline} $')

    # Obtain and solve optimization problem with bidding strategy
    for timestep in timesteps:
        price_forecast.to_csv(os.path.join(results_path, f'price_forecast_{timestep}.csv'.replace(':', '_')))
        lower_price_limit = price_forecast.at[timestep, 'lower_limit']
        upper_price_limit = price_forecast.at[timestep, 'upper_limit']
        price_points = np.linspace(lower_price_limit, upper_price_limit, 4)
        bids = pd.DataFrame(0, price_points, ['P'])
        output_vectors_path = os.path.join(results_path, f'output_vectors_{timestep}'.replace(':', '_'))
        os.mkdir(output_vectors_path)
        for price in price_points:
            optimization_problem = cobmo.optimization_problem.OptimizationProblem(
                building,
                problem_type='rolling_forecast',
                price_forecast=price_forecast,
                price_scenario_timestep=timestep,
                price_point=price,
                actual_dispatch=actual_dispatch
            )
            (
                control_vector_optimization,
                state_vector_optimization,
                output_vector_optimization,
                operation_cost,
                investment_cost,  # Zero when running (default) operation problem.
                storage_size  # Zero when running (default) operation problem.
            ) = optimization_problem.solve()

            output_vector_optimization.to_csv(os.path.join(output_vectors_path, f'output_vector_optimization_{price}.csv'))
            bids.loc[price, 'P'] = output_vector_optimization.at[timestep, 'grid_electric_power']
        bids.to_csv(os.path.join(results_path, f'bids_{timestep}.csv'.replace(':','_')))

        # TODO: pass bids to market clearing engine and obtain clearing price

        # Update actual dispatch quantity to be used in the next timestep
        dispatch_quantity = determine_dispatch_quantity(bids, actual_dispatch.at[timestep, 'clearing_price'])
        print(timestep, dispatch_quantity)
        actual_dispatch.loc[timestep, 'actual_dispatch'] = dispatch_quantity

        # Update forecast, skip if at the last timestep
        if timestep == timesteps[-1]:
            continue
        new_timesteps = timesteps[timesteps > timestep]
        forecast_timestep += dt.timedelta(minutes=30)
        forecast_model.update_model(actual_dispatch.loc[timestep, 'clearing_price'], forecast_timestep)
        price_forecast = forecast_model.forecast_prices(steps=len(new_timesteps))
        price_forecast.index = new_timesteps

    daily_cost_optimized = (
        actual_dispatch['clearing_price'] * actual_dispatch['actual_dispatch'].sum()
        * building.timestep_interval.seconds / 3600 / 1e6
    )
    print(f'Daily cost (optimized): {daily_cost_optimized} $')

    actual_dispatch.to_csv(os.path.join(results_path, 'actual_dispatch.csv'))

    # Print results path.
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
