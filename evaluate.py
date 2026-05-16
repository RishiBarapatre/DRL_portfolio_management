import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
import os
import time

from config import *
from portfolio_env import PortfolioEnv
from model import CustomCNN 

def get_benchmark_data():
    """
    Robustly loads or downloads the benchmark data.
    """
    try:
        print(f"Loading benchmark data from file: {BENCHMARK_PATH}...")
        try:
            benchmark_data = pd.read_csv(BENCHMARK_PATH, header=2, index_col=0, parse_dates=True)
            if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                raise ValueError("Index not dates")
        except:
            benchmark_data = pd.read_csv(BENCHMARK_PATH, index_col=0, parse_dates=True)

        benchmark_data.index.name = 'Date'
        benchmark_data.sort_index(inplace=True)
        return benchmark_data

    except Exception:
        print("Downloading Nifty 50 benchmark data directly...")
        time.sleep(1)
        try:
            benchmark_data = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE)
            if 'Adj Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data[['Adj Close']]
            else:
                benchmark_prices = benchmark_data[['Close']]
            
            benchmark_prices.to_csv(BENCHMARK_PATH)
            return benchmark_prices
        except Exception as e:
            print(f"Failed to download benchmark: {e}")
            return None

def run_evaluation():
    print("--- Running Evaluation ---")
    
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, index_col='Date', parse_dates=True)
    
    print("Loading model...")
    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=FEATURES_DIM))
    model = PPO.load(MODEL_PATH, policy_kwargs=policy_kwargs)
    
    price_cols = TICKERS
    signal_cols = [col for col in test_df.columns if col not in price_cols]

    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH, index_col='Date', parse_dates=True)
    
    print("--- VERIFYING DATA LENGTH ---")
    print("Row count in test file:", len(test_df))
    print("Date range of test file:", test_df.index.min(), "to", test_df.index.max())
    print("-----------------------------")
    
    test_env = PortfolioEnv(
        data_df=test_df, 
        ticker_list=price_cols, 
        signal_list=signal_cols,
        window_length=WINDOW_LENGTH,
        start_date_index=WINDOW_LENGTH-1,
        steps=len(test_df) - WINDOW_LENGTH - 1
    )

    print("--- DIAGNOSTIC CHECK ---")
    print("WINDOW_LENGTH variable is:", WINDOW_LENGTH)
    print("test_df shape is:", test_df.shape)
    print("Initial observation shape from env reset is:", test_env.reset()[0].shape)
    print("------------------------")

    print("Running evaluation backtest...")
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    results_df = pd.DataFrame(test_env.info_list).set_index('date')
    
    print("Extracting portfolio weights...")
    weight_data = []
    for row in test_env.info_list:
        if 'weights' in row:
            w_dict = {f'weight_{ticker}': row['weights'][i+1] for i, ticker in enumerate(TICKERS)}
            w_dict['weight_CASH'] = row['weights'][0]
            weight_data.append(w_dict)
    
    if weight_data:
        weights_df = pd.DataFrame(weight_data, index=results_df.index)
        results_df = results_df.join(weights_df)

    benchmark_prices = get_benchmark_data()
    
    if benchmark_prices is not None:
        benchmark_test = benchmark_prices.reindex(test_df.index, method='ffill').bfill()
        
        benchmark_returns = benchmark_test.pct_change().iloc[:, 0].fillna(0)
        benchmark_value = (1 + benchmark_returns).cumprod()
        
        benchmark_to_plot = benchmark_value.loc[results_df.index[0]:]
        benchmark_to_plot = benchmark_to_plot / benchmark_to_plot.iloc[0]
        
        benchmark_to_plot = benchmark_to_plot.iloc[:len(results_df)]
        # ----------------------------------------
        
        results_df['benchmark_value'] = benchmark_to_plot.values

    results_df.to_csv(RESULTS_PATH)
    print(f"Evaluation complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()