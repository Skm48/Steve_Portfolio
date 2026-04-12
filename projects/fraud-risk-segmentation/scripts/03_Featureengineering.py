import pandas as pd

# Load cleaned data
data = pd.read_csv("projects/fraud-risk-segmentation/data/cleaned/cleaned_data.csv")

print("Starting shape:", data.shape)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# Repair cost relative to vehicle value (%)
data['price_cost_ratio'] = (data['repair_cost'] / data['Price']) * 100

# Cost per repair hour (labour efficiency)
data['cost_per_hour'] = data['repair_cost'] / data['repair_hours']

# Convert dates
data['breakdown_date'] = pd.to_datetime(data['breakdown_date'])
data['repair_date'] = pd.to_datetime(data['repair_date'])

# Time between breakdown and repair
data['repair_lag_days'] = (data['repair_date'] - data['breakdown_date']).dt.days

# -------------------------
# CLEAN NEW FEATURES
# -------------------------

# Remove extreme labour rates
data = data[data['cost_per_hour'] <= 1000]

# Remove unrealistic ratios
data = data[data['price_cost_ratio'] <= 100]

print("Final shape:", data.shape)

# Save featured data
data.to_csv("projects/fraud-risk-segmentation/data/cleaned/featured_data.csv", index=False)

print("✅ Feature engineering complete")
