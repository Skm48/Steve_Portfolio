import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("projects/early-fraud-detection-ml/data/raw/fraud_data.csv")
print("Original shape:", data.shape)

# -------------------------
# DATA CLEANING
# -------------------------

# Remove invalid vehicle prices
data = data[(data['Price'] > 0) & (data['Price'] < 1000000)]

# Remove invalid mileage
data = data[(data['Runned_Miles'] >= 0) & (data['Runned_Miles'] <= 500000)]

# Remove unrealistic repair costs
data = data[(data['repair_cost'] >= 0) & (data['repair_cost'] <= 50000)]

# Remove unrealistic repair hours
data = data[(data['repair_hours'] >= 0) & (data['repair_hours'] <= 500)]

# Remove invalid registration years
data = data[(data['Reg_year'] >= 1990) & (data['Reg_year'] <= 2021)]

# Remove invalid months
data = data[(data['Adv_month'] >= 1) & (data['Adv_month'] <= 12)]

print("Cleaned shape:", data.shape)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# Cost per hour — inflated labour rates are a primary fraud mechanism
data['cost_per_hour'] = data['repair_cost'] / data['repair_hours']
data = data[data['cost_per_hour'] <= 500]

# Price-cost ratio — repair cost relative to vehicle value
data['price_cost_ratio'] = data['repair_cost'] / data['Price']
data = data[data['price_cost_ratio'] <= 100]

# Repair lag — days between breakdown and repair
data['breakdown_date'] = pd.to_datetime(data['breakdown_date'])
data['repair_date']    = pd.to_datetime(data['repair_date'])
data['repair_lag_days'] = (data['repair_date'] - data['breakdown_date']).dt.days

# Save cleaned data
data.to_csv("projects/early-fraud-detection-ml/data/cleaned/cleaned_data.csv", index=False)
print("✅ Cleaning complete")
print("Final shape:", data.shape)
print(f"Fraud rate: {data['anomaly_present'].mean() * 100:.2f}%")
