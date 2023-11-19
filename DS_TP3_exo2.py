import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to generate synthetic data
def generate_synthetic_data(start_date, end_date):
    start_date = pd.to_datetime(start_date, format='%d/%m/%Y')
    end_date = pd.to_datetime(end_date, format='%d/%m/%Y')

    date_range = pd.date_range(start=start_date, end=end_date)
    temperature_data = [random.randint(20, 35) for _ in range(len(date_range))]

    df = pd.DataFrame({'Date': date_range, 'Temperature': temperature_data})
    return df

# Function to calculate moving average
def moving_average(data, window_size):
    return data['Temperature'].rolling(window=window_size, min_periods=1).mean()

# Function to calculate weighted moving average
def weighted_moving_average(data, weights):
    return data['Temperature'].rolling(window=len(weights), min_periods=1).apply(lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]), raw=True)

# Function to visualize the data and smoothing effects
def visualize_smoothing(data, window_sizes, weights):
    plt.figure(figsize=(12, 6))

    # Original data
    plt.plot(data['Date'], data['Temperature'], label='Original Data', color='black', linestyle='--')

    # Moving averages with different window sizes
    for window_size in window_sizes:
        ma_data = moving_average(data, window_size)
        plt.plot(data['Date'], ma_data, label=f'Moving Average (Window Size = {window_size})')

    # Weighted moving averages with different weights
    for weight in weights:
        wma_data = weighted_moving_average(data, weight)
        plt.plot(data['Date'], wma_data, label=f'Weighted Moving Average (Weights = {weight})')

    plt.title('Smoothing Techniques Comparison')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

# Example usage
start_date = '01/01/2023'
end_date = '31/01/2023'
window_sizes = [3, 7, 14]  # Different window sizes for moving average
weights = [[0.2, 0.3, 0.5], [0.1, 0.2, 0.3, 0.4], [0.05, 0.1, 0.2, 0.3, 0.4]]  # Different weights for weighted moving average

# Generate synthetic data
synthetic_data = generate_synthetic_data(start_date, end_date)

# Visualize the data and smoothing effects
visualize_smoothing(synthetic_data, window_sizes, weights)
