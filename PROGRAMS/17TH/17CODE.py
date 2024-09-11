import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    'battery_capacity': np.random.normal(4000, 1000, n_samples),
    'ram_gb': np.random.choice([2, 3, 4, 6, 8, 12, 16], n_samples),
    'internal_memory_gb': np.random.choice([16, 32, 64, 128, 256, 512], n_samples),
    'screen_size_inches': np.random.normal(6, 0.5, n_samples),
    'camera_mp': np.random.normal(20, 10, n_samples),
    'front_camera_mp': np.random.normal(10, 5, n_samples),
    'processor_cores': np.random.choice([4, 6, 8], n_samples),
    'processor_speed_ghz': np.random.normal(2.2, 0.3, n_samples),
    'has_5g': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'has_nfc': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'has_infrared': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
})

data['battery_capacity'] = np.clip(data['battery_capacity'], 2000, 7000)
data['screen_size_inches'] = np.clip(data['screen_size_inches'], 5, 7.5)
data['camera_mp'] = np.clip(data['camera_mp'], 8, 108)
data['front_camera_mp'] = np.clip(data['front_camera_mp'], 5, 32)
data['processor_speed_ghz'] = np.clip(data['processor_speed_ghz'], 1.5, 3.0)

price_score = (
    data['battery_capacity'] * 0.01
    + data['ram_gb'] * 20
    + data['internal_memory_gb'] * 0.5
    + (data['screen_size_inches'] - 6) * 100
    + data['camera_mp'] * 2
    + data['front_camera_mp'] * 3
    + data['processor_cores'] * 30
    + (data['processor_speed_ghz'] - 2) * 200
    + data['has_5g'] * 150
    + data['has_nfc'] * 50
    + data['has_infrared'] * 30
)

data['price_category'] = pd.cut(price_score, 
                                bins=[-np.inf, 300, 500, 700, np.inf],
                                labels=['Budget', 'Mid-range', 'High-end', 'Flagship'])

data.to_csv('mobile_price_data.csv', index=False)
print("Mobile price classification dataset saved as 'mobile_price_data.csv'")
print(data.head())
print("\nPrice Category Distribution:")
print(data['price_category'].value_counts(normalize=True))
