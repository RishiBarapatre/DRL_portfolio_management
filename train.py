# train.py
# This script runs the full training process.

import pandas as pd
import os
from stable_baselines3 import PPO

from config import *
from portfolio_env import PortfolioEnv
from model import CustomCNN

def run_training():
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
        
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH, index_col='Date', parse_dates=True)
    
    price_cols = TICKERS
    signal_cols = [col for col in train_df.columns if col not in price_cols]
    
    print(f"Instantiating training environment with {len(signal_cols)} signals...")
    train_env = PortfolioEnv(
        data_df=train_df, 
        ticker_list=price_cols, 
        signal_list=signal_cols,
        window_length=WINDOW_LENGTH
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )
    
    print("Defining the PPO agent...")
    model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR
    )
    
    print(f"Starting training for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    
    print(f"Training complete. Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    run_training()