import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    'area_sqft': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
    'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples),
    'stories': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.6, 0.1]),
    'garage': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.7, 0.1]),
    'year_built': np.random.randint(1950, 2024, n_samples),
    'lot_size': np.random.normal(8000, 2000, n_samples),
    'basement': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'renovation_year': np.random.choice([0] + list(range(1980, 2024)), n_samples)
})

data['area_sqft'] = np.clip(data['area_sqft'], 500, 5000)
data['lot_size'] = np.clip(data['lot_size'], 2000, 20000)

base_price = 200000
price = (
    base_price
    + data['area_sqft'] * 100
    + data['bedrooms'] * 20000
    + data['bathrooms'] * 15000
    + data['stories'] * 25000
    + data['garage'] * 15000
    + (2024 - data['year_built']) * -500
    + data['lot_size'] * 2
    + data['basement'] * 30000
)

price += np.where(data['renovation_year'] > 0, (2024 - data['renovation_year']) * 1000, 0)

data['price'] = np.abs(price + np.random.normal(0, 20000, n_samples))

data.to_csv('house_price_data.csv', index=False)
print("House price dataset saved as 'house_price_data.csv'")
print(data.head())
print("\nPrice Statistics:")
print(data['price'].describe())
