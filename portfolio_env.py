import gymnasium as gym
import gymnasium.spaces
import os
import numpy as np
import pandas as pd

EPS = 1e-8


class PortfolioEnv(gym.Env):
    """
    A financial market environment for a portfolio management RL agent.

    Args:
        data_df (pd.DataFrame): A DataFrame containing the market data (prices and signals).
        steps (int): The number of steps (days) in a single episode.
        trading_cost (float): The cost of a trade as a percentage.
        window_length (int): The number of past observations to include in the state.
        start_date_index (int, optional): The starting index for the episode. If None, a random start is chosen.
    """
    
class PortfolioEnv(gym.Env):
    
    def __init__(self, data_df, ticker_list, signal_list, steps=505, trading_cost=0.001, window_length=5, start_date_index=None):
    
        self.data_df = data_df.copy()
        self.tickers = ticker_list
        self.signal_cols = signal_list
        
        self.info_list = []
        self.portfolio_value = 1.0
        self.step_number = 0
        self.weights = np.insert(np.zeros(len(self.tickers)), 0, 1.0)

        self.trading_cost = trading_cost
        self.window_length = window_length
        self.steps = steps
        self.start_date_index = start_date_index
        
        prices = self.data_df[self.tickers].values
        self.gain = np.hstack((np.ones((prices.shape[0] - 1, 1)), prices[1:] / (prices[:-1] + EPS)))
        self.dates = self.data_df.index.values[1:]
        self.n_dates = self.dates.shape[0]
        self.n_tickers = len(self.tickers)

        self.signals = self.data_df[self.signal_cols].T.values[..., np.newaxis]
        self.n_signals = self.signals.shape[0]
        
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_tickers,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_signals, self.window_length, 1), 
            dtype=np.float32
        )

        self.reset()
    
        
    def step(self, action):
        """Steps the environment forward one day."""
        self.step_number += 1
    
        w1 = np.clip(action, a_min=0, a_max=1)
        w1 = np.insert(w1, 0, np.clip(1 - w1.sum(), a_min=0, a_max=1))
        w1 /= (w1.sum() + EPS)

        t = self.start_date_index + self.step_number
        y1 = self.gain[t]
        w0 = self.weights
        p0 = self.portfolio_value

        dw1 = (y1 * w0) / (np.dot(y1, w0) + EPS)
        mu1 = self.trading_cost * (np.abs(dw1 - w1)).sum()
    
        p1 = p0 * (1 - mu1) * np.dot(y1, w1)
        p1 = np.clip(p1, 0, np.inf)
    
        reward = np.log((p1 + EPS) / (p0 + EPS))

        self.weights = w1
        self.portfolio_value = p1

        t0 = t - self.window_length + 1
        observation = self.signals[:, t0:t+1, :]
    
        info = {
            'reward': reward,
            'portfolio_value': p1,
            'date': self.dates[t],
            'cost': mu1,
            'weights': w1
        }
        self.info_list.append(info)

        terminated = (p1 <= 0.5)
        truncated = (self.step_number >= self.steps)

        if terminated or truncated:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            results_path = os.path.join(output_dir, f'results_episode_{self.start_date_index}.csv')
            pd.DataFrame(self.info_list).set_index('date').to_csv(results_path)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to a new starting state."""
        super().reset(seed=seed)
        self.info_list = []
        self.weights = np.insert(np.zeros(self.n_tickers), 0, 1.0)
        self.portfolio_value = 1.0
        self.step_number = 0

        self.steps = min(self.steps, self.n_dates - self.window_length - 1)

        if self.start_date_index is None:
            self.start_date_index = self.np_random.integers(
                self.window_length - 1, self.n_dates - self.steps - 1
            )
    
        t = self.start_date_index + self.step_number
        t0 = t - self.window_length + 1
        observation = self.signals[:, t0:t+1, :]
    
        info = {}
    
        return observation, info