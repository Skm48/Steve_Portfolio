import pandas as pd

# Load data
data = pd.read_csv("projects/fraud-risk-segmentation/data/raw/fraud_data.csv")

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

# Remove invalid years
data = data[(data['Reg_year'] >= 1990) & (data['Reg_year'] <= 2021)]

# Remove invalid months
data = data[(data['Adv_month'] >= 1) & (data['Adv_month'] <= 12)]

print("Cleaned shape:", data.shape)

# Save cleaned data
data.to_csv("projects/fraud-risk-segmentation/data/cleaned/cleaned_data.csv", index=False)

print("✅ Cleaning complete")
