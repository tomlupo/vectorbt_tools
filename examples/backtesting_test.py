import sys
import os
# Add the parent directory to the Python path to find the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import vectorbt as vbt
from vectorbt_tools import run_backtest

# Example usage of rebalancing bands
if __name__ == "__main__":    

    # %% settings
    num_tests = 10

    # %% load data
    prices = pd.read_csv('data/sample_data.csv',index_col=0).iloc[:100]

    # %% generate random weights
    np.random.seed(vbt.settings.portfolio['seed'])
    weights = {}
    for i in range(num_tests):
        w = np.random.random_sample(prices.shape[1])
        w = w / np.sum(w)
        weights[f'strategy_{i}'] = pd.DataFrame(w, columns=[prices.index[0]],index=prices.columns).T.reindex(prices.index).ffill()

    import time  # Import time module for execution measurement

    # Initialize a dictionary to store execution times
    execution_times = {}

    # %% Run a standard backtest with no rebalancing
    start_time = time.time()  # Start timing
    pf = run_backtest(prices, weights, rebalancing_freq=20)
    execution_times['no_rebalancing'] = time.time() - start_time  # Store execution time
    print("Standard backtest (no rebalancing):")
    print(f"Number of orders: {len(pf.orders)}")
    print(f"Sharpe ratio: {pf.sharpe_ratio()}")
    pf.save('data/sample_backtest.pkl')

    # %% Run a standard backtest with no rebalancing, force using order_func
    start_time = time.time()
    pf_order_func = run_backtest(prices, weights, rebalancing_freq=20, use_order_func=True)
    execution_times['no_rebalancing_order_func'] = time.time() - start_time
    print("Standard backtest (no rebalancing, use order_func):")
    print(f"Number of orders: {len(pf_order_func.orders)}")
    print(f"Sharpe ratio: {pf_order_func.sharpe_ratio()}")

    # %% Run a backtest with threshold rebalancing (0)
    start_time = time.time()
    pf_bands0 = run_backtest(prices, weights, rebalancing_freq=20, threshold=0)
    execution_times['threshold_0'] = time.time() - start_time
    print("\nBacktest with threshold rebalancing (0%):")
    print(f"Number of orders: {len(pf_bands0.orders)}")
    print(f"Sharpe ratio: {pf_bands0.sharpe_ratio()}")

    # %% Run a backtest with relative threshold rebalancing (2%)
    start_time = time.time()
    pf_bands2 = run_backtest(prices, weights, rebalancing_freq=20, threshold=0.02)
    execution_times['threshold_2'] = time.time() - start_time
    print("\nBacktest with relative threshold rebalancing (2%):")
    print(f"Number of orders: {len(pf_bands2.orders)}")
    print(f"Sharpe ratio: {pf_bands2.sharpe_ratio()}")

    # %% Run a backtest with relative threshold rebalancing (2%)
    start_time = time.time()
    pf_bands2_disable_numba = run_backtest(prices, weights, rebalancing_freq=20, threshold=0.02, use_numba=False)
    execution_times['threshold_2_disable_numba'] = time.time() - start_time
    print("\nBacktest with relative threshold rebalancing (2%), disable numba:")
    print(f"Number of orders: {len(pf_bands2_disable_numba.orders)}")
    print(f"Sharpe ratio: {pf_bands2_disable_numba.sharpe_ratio()}")

    # Print execution time summary
    print("\nExecution Time Summary:")
    for test_name, exec_time in execution_times.items():
        print(f"{test_name}: {exec_time:.4f} seconds")

    # Summary
    print('Sharpe ratio:')
    print(
        pd.concat(
            [
                pf.sharpe_ratio(),
                pf_order_func.sharpe_ratio(),
                pf_bands0.sharpe_ratio(),
                pf_bands2.sharpe_ratio(),
                pf_bands2_disable_numba.sharpe_ratio()
            ],
            axis=1,
            keys=[
                'no_bands',
                'no_bands_order_func',
                'bands0',
                'bands2',
                'bands2_disable_numba'
            ]
        )
    )
