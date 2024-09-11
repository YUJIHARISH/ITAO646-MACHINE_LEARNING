import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    'year': np.random.randint(2000, 2024, n_samples),
    'mileage': np.random.normal(50000, 30000, n_samples),
    'fuel_type': np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], n_samples),
    'transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
    'engine_size': np.random.normal(2.0, 0.5, n_samples),
    'horsepower': np.random.normal(200, 50, n_samples),
    'num_doors': np.random.choice([2, 4, 5], n_samples),
    'weight': np.random.normal(3000, 500, n_samples)
})

data['mileage'] = np.abs(data['mileage'])
data['engine_size'] = np.clip(data['engine_size'], 0.8, 5.0)
data['horsepower'] = np.clip(data['horsepower'], 50, 500)
data['weight'] = np.clip(data['weight'], 1500, 5000)

base_price = 20000
price = (
    base_price
    + (data['year'] - 2000) * 500
    - data['mileage'] * 0.05
    + (data['engine_size'] - 2.0) * 5000
    + (data['horsepower'] - 200) * 100
    + (data['weight'] - 3000) * 2
)

price += np.where(data['fuel_type'] == 'Electric', 10000, 0)
price += np.where(data['fuel_type'] == 'Hybrid', 5000, 0)
price += np.where(data['transmission'] == 'Automatic', 2000, 0)

data['price'] = np.abs(price + np.random.normal(0, 2000, n_samples))

data.to_csv('car_price_data.csv', index=False)
print("Car price dataset saved as 'car_price_data.csv'")
print(data.head())
print("\nPrice Statistics:")
print(data['price'].describe())
