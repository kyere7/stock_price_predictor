"""
This script provides functions to compute various technical indicators for stock market analysis. 
"""

# Import necessary libraries
import pandas as pd


# RSI
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# MACD
def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
    ) -> pd.DataFrame:
    """Compute MACD and Signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return pd.DataFrame({
        "MACD": macd,
        "MACD_Signal": macd_signal,
        "MACD_Hist": macd_hist,
    })

# Bollinger Bands (20-day)
def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: int = 2
) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    upper_band = sma + num_std * std
    lower_band = sma - num_std * std

    return pd.DataFrame({
        "BB_Middle": sma,
        "BB_Upper": upper_band,
        "BB_Lower": lower_band,
    })


# Add the indcators to the DataFrame
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, and Bollinger Bands."""
    df = df.copy()

    df["RSI_14"] = compute_rsi(df["Adj Close"])

    macd_df = compute_macd(df["Adj Close"])
    df = df.join(macd_df)

    bb_df = compute_bollinger_bands(df["Adj Close"])
    df = df.join(bb_df)

    df = df.dropna()
    return df