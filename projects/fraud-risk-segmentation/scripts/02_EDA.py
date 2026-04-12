import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
data = pd.read_csv("projects/fraud-risk-segmentation/data/cleaned/cleaned_data.csv")

# -------------------------
# FEATURE ENGINEERING 
# -------------------------

data['price_cost_ratio'] = (data['repair_cost'] / data['Price']) * 100
data['cost_per_hour'] = data['repair_cost'] / data['repair_hours']

data['breakdown_date'] = pd.to_datetime(data['breakdown_date'])
data['repair_date'] = pd.to_datetime(data['repair_date'])
data['repair_lag_days'] = (data['repair_date'] - data['breakdown_date']).dt.days

# -------------------------
# DISTRIBUTION ANALYSIS 
# -------------------------

# Price-to-cost ratio 
plt.figure()
sns.histplot(data['price_cost_ratio'], bins=50)
plt.title("Price-to-Cost Ratio Distribution")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/ratio_distribution.png")
plt.close()

# Cost per hour
plt.figure()
sns.histplot(data['cost_per_hour'], bins=50)
plt.title("Cost per Repair Hour Distribution")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/cost_per_hour.png")
plt.close()

# Repair delay
plt.figure()
sns.histplot(data['repair_lag_days'], bins=50)
plt.title("Repair Lag Distribution")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/repair_lag.png")
plt.close()

# -------------------------
# RELATIONSHIPS 
# -------------------------

# Ratio vs Price
plt.figure()
sns.scatterplot(x='Price', y='price_cost_ratio', data=data)
plt.title("Price vs Price-Cost Ratio")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/ratio_vs_price.png")
plt.close()

# Cost per hour vs repair cost
plt.figure()
sns.scatterplot(x='repair_cost', y='cost_per_hour', data=data)
plt.title("Repair Cost vs Cost per Hour")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/cost_relationship.png")
plt.close()

# -------------------------
# CORRELATION 
# -------------------------

corr = data[['Price', 'repair_cost', 'repair_hours',
             'price_cost_ratio', 'cost_per_hour', 'repair_lag_days']].corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/correlation.png")
plt.close()

print("✅ EDA complete")
