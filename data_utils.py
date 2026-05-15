import pandas as pd
import yfinance as yf
import os
import technical_indicators as ti
from sklearn.preprocessing import StandardScaler
from config import * 
import time
import numpy as np
import requests_cache # Optional but highly recommended
import requests
from requests_cache import CachedSession
# session = CachedSession('yfinance.cache')
# Use this session in yf.download(..., session=session)

# At the top of your script
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})


def download_stock_data():
    """
    Downloads and caches the 50 stock tickers IN BATCHES to avoid rate limits.
    """
    if os.path.exists(RAW_DATA_PATH):
        print(f"Loading stock data from local cache: {RAW_DATA_PATH}")
        return pd.read_csv(RAW_DATA_PATH, index_col='Date', parse_dates=True)
    
    print("Starting batched download for all 50 tickers...")
    
    all_chunks = []
    batch_size = 5
    num_batches = int(np.ceil(len(TICKERS) / batch_size))
    ticker_batches = np.array_split(TICKERS, num_batches)
    
    for i, batch in enumerate(ticker_batches):
        try:
            print(f" Downloading batch {i + 1}/{num_batches}...")
            batch_tickers = list(batch)
        
            raw_data = yf.download(batch_tickers, start=START_DATE, end=END_DATE, threads=False)
        
            if not raw_data.empty:
                # Extract only the Close prices available
                if 'Adj Close' in raw_data.columns:
                    data_chunk = raw_data['Adj Close']
                else:
                    data_chunk = raw_data['Close']
            
                # This ensures we only keep tickers that actually returned data
                valid_tickers_in_batch = data_chunk.columns[data_chunk.notna().any()].tolist()
                all_chunks.append(data_chunk[valid_tickers_in_batch])
            
                print(f" ...Success. Found {len(valid_tickers_in_batch)}/{len(batch_tickers)} tickers.")
        
            time.sleep(60)
                
        except Exception as e:
            print(f" --- Batch {i + 1} failed entirely: {e}")

    if not all_chunks:
        print("\nERROR: No stock data could be downloaded. Exiting.")
        return pd.DataFrame()

    all_data = pd.concat(all_chunks, axis=1)
    
    all_data.dropna(axis=1, how='all', inplace=True)
    
    all_data.to_csv(RAW_DATA_PATH)
    print(f"\nStock data saved successfully to '{RAW_DATA_PATH}'")
    return all_data

def download_benchmark_data():
    """
    Downloads and caches the Nifty 50 Index data.
    Waits 60 seconds *before* downloading to be safe.
    """
    if os.path.exists(BENCHMARK_PATH):
        print(f"Loading benchmark data from local cache: {BENCHMARK_PATH}")
        return pd.read_csv(BENCHMARK_PATH, index_col='Date', parse_dates=True)
    
    print("Waiting 60 seconds before downloading benchmark to be safe...")
    time.sleep(60)
    
    print(f"Downloading benchmark ticker {BENCHMARK_TICKER}...")
    try:
        data = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE)
        if data.empty: raise Exception("No data returned for benchmark.")
        
        if 'Adj Close' in data.columns:
            benchmark_data = data[['Adj Close']]
        else:
            benchmark_data = data[['Close']]
            
        benchmark_data.to_csv(BENCHMARK_PATH)
        print(f"Benchmark data saved successfully to '{BENCHMARK_PATH}'")
        return benchmark_data
        
    except Exception as e:
        print(f"ERROR downloading benchmark data: {e}")
        return pd.DataFrame()

def create_features(all_data):
    print("Calculating technical indicators...")
    feature_list = [] # Store individual Series here
    
    for ticker in all_data.columns:
        # Create a temp dict or list for this ticker's features
        rsi = ti.calculate_rsi(all_data[ticker]).rename(f'RSI_{ticker}')
        sma = ti.calculate_sma(all_data[ticker]).rename(f'SMA_{ticker}')
        upper, middle, lower = ti.calculate_bbands(all_data[ticker])
        
        feature_list.extend([
            rsi, 
            sma, 
            upper.rename(f'BB_Upper_{ticker}'), 
            middle.rename(f'BB_Middle_{ticker}'), 
            lower.rename(f'BB_Lower_{ticker}')
        ])
    
    # Combine everything at once (much faster and no warnings)
    signals_df = pd.concat(feature_list, axis=1)
    
    print(f"Engineered {signals_df.shape[1]} total features.")
    return signals_df

def process_and_save_data():
    """Main function to run the full data pipeline for the stocks."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_data = download_stock_data()
    if all_data.empty: return

    signals_df = create_features(all_data)
    
    final_df = all_data.join(signals_df)
    final_df.dropna(inplace=True)
    
    train_size = int(len(final_df) * TRAIN_TEST_SPLIT)
    train_df = final_df.iloc[:train_size].copy()
    test_df = final_df.iloc[train_size:].copy()
    
    print("Scaling features... This may take a moment.")
    price_cols = [col for col in all_data.columns if col in TICKERS]
    signal_cols = [col for col in signals_df.columns]
    
    scaler = StandardScaler()
    scaler.fit(train_df[signal_cols])
    
    train_df[signal_cols] = scaler.transform(train_df[signal_cols])
    test_df[signal_cols] = scaler.transform(test_df[signal_cols])
    
    train_df.to_csv(TRAIN_DATA_PATH)
    test_df.to_csv(TEST_DATA_PATH)
    
    print(f"Processed stock data saved to {TRAIN_DATA_PATH} and {TEST_DATA_PATH}")

if __name__ == "__main__":
    process_and_save_data()
    download_benchmark_data()