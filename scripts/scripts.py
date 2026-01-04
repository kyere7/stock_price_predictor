"""
Other scripts and functions 
"""

import pandas as pd
import numpy as np


#train-test-split function for a single DataFrame
def train_test_split(df: pd.DataFrame, train_size: float = 0.8):
  """
  Splits a single DataFrame into training and testing sets based on chronological order.

  Args:
      df (pd.DataFrame): The DataFrame to split.
      train_size (float): The proportion of the DataFrame to use for training.

  Returns:
      tuple: (df_train, df_test) DataFrames.
  """
  train_split_point = int(len(df) * train_size)
  df_train = df.iloc[:train_split_point]
  df_test = df.iloc[train_split_point:]
  return df_train, df_test


# function to create sequences for LSTM model
def create_sequences(features, targets, window=30):
  X, y = [], []
  for i in range(window, len(features)):
      X.append(features[i-window:i])
      y.append(targets[i])
  return np.array(X), np.array(y)