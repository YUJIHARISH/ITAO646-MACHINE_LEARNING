import pandas as pd
import numpy as np

np.random.seed(42)

date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
n_samples = len(date_range)

data = pd.DataFrame({
    'date': date_range,
    'base_sales': np.random.normal(1000, 200, n_samples)
})

data['trend'] = np.linspace(0, 500, n_samples)

data['yearly_seasonality'] = np.sin(2 * np.pi * data.index / 365) * 200
data['weekly_seasonality'] = np.sin(2 * np.pi * data.index / 7) * 50

christmas_dates = pd.to_datetime(['2021-12-25', '2022-12-25', '2023-12-25'])
data['holiday_effect'] = np.where(data['date'].isin(christmas_dates), 500, 0)

data['sales'] = data['base_sales'] + data['trend'] + data['yearly_seasonality'] + data['weekly_seasonality'] + data['holiday_effect']

data['sales'] += np.random.normal(0, 50, n_samples)

data['sales'] = np.maximum(data['sales'], 0)

data = data[['date', 'sales']]

data.to_csv('future_sales_data.csv', index=False)
print("Future sales prediction dataset saved as 'future_sales_data.csv'")
print(data.head())
print("\nSales Statistics:")
print(data['sales'].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['sales'])
plt.title('Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.savefig('sales_time_series.png')
plt.close()

print("\nTime series plot saved as 'sales_time_series.png'")
