import pandas as pd

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data (pd.Series): A pandas Series of prices (e.g., closing prices).
        window (int): The lookback period for the RSI calculation.

    Returns:
        pd.Series: A pandas Series containing the RSI values.
    """
    delta = data.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(0)

def calculate_sma(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculates the Simple Moving Average (SMA)."""
    return data.rolling(window=window).mean().fillna(0)

def calculate_bbands(data: pd.Series, window: int = 20, std_dev: int = 2) -> tuple:
    """
    Calculates Bollinger Bands.

    Returns:
        A tuple of pandas Series: (Upper_Band, Middle_Band, Lower_Band)
    """
    middle_band = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = (middle_band + std_dev * std).fillna(0)
    lower_band = (middle_band - std_dev * std).fillna(0)
    
    return upper_band, middle_band, lower_band