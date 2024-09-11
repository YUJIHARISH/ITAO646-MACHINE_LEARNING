import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'annual_income': np.random.normal(50000, 20000, n_samples),
    'credit_card_debt': np.random.normal(5000, 3000, n_samples),
    'num_credit_cards': np.random.randint(0, 6, n_samples),
    'num_late_payments': np.random.poisson(1, n_samples),
    'credit_utilization': np.random.beta(2, 5, n_samples) * 100,
    'years_of_credit_history': np.random.randint(1, 30, n_samples),
    'num_loan_accounts': np.random.randint(0, 5, n_samples)
})

data['credit_card_debt'] = np.abs(data['credit_card_debt'])

credit_score = (
    data['annual_income'] / 1000 
    - data['credit_card_debt'] / 1000 
    + data['num_credit_cards'] * 20
    - data['num_late_payments'] * 50
    - data['credit_utilization'] 
    + data['years_of_credit_history'] * 10
    - data['num_loan_accounts'] * 5
)

data['credit_score'] = pd.cut(credit_score, 
                              bins=[-np.inf, -200, 0, 200, np.inf],
                              labels=['Poor', 'Fair', 'Good', 'Excellent'])

data.to_csv('credit_score_data.csv', index=False)
print("Credit score dataset saved as 'credit_score_data.csv'")
print(data.head())
print("\nCredit Score Distribution:")
print(data['credit_score'].value_counts(normalize=True))
