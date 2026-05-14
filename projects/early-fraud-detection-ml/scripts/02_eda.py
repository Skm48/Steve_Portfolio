import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned data (from 01_data_cleaning.py)
data = pd.read_csv('projects/early-fraud-detection-ml/data/cleaned/cleaned_data.csv')
print(f"Dataset shape: {data.shape}")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")

# Output figures folder
figures_path = 'projects/early-fraud-detection-ml/outputs/figures'
os.makedirs(figures_path, exist_ok=True)

# Show first few rows
data.head(4).T

# -------------------------
# EXPLORE DATA
# -------------------------

data.info()
data.isna().sum()

# -------------------------
# DESCRIPTIVE STATISTICS
# -------------------------

data.describe().T.round(2)

# -------------------------
# VISUALIZATION
# -------------------------

# --- 1. Fraud Vs Legitimate distribution analysis ---
print(data['anomaly_present'].value_counts())
print(f"anomaly_present fraud cases : {data['anomaly_present'].mean()*100:.1f}%")
print(data['category_anomaly'].value_counts())
print(f"category_anomaly fraud cases: {data['category_anomaly'].mean()*100:.1f}%")

plt.figure(figsize=(8, 5))
plt.bar(['Legitimate', 'Fraudulent'], data['anomaly_present'].value_counts())
plt.title('Fraud vs Legitimate Claims', fontsize=14, fontweight='bold')
plt.ylabel('Number of Claims', fontsize=12)
plt.tight_layout()
plt.savefig(f'{figures_path}/01_fraud_vs_legitimate.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 2. Fraud rate by different Maker analysis ---
fraud_by_maker = data.groupby('Maker').agg({
    'anomaly_present': ['sum', 'count', 'mean']}).round(4)
fraud_by_maker.columns = ['Fraud_Count', 'Total_Claims', 'Fraud_Rate']
fraud_by_maker['Fraud_Rate'] = fraud_by_maker['Fraud_Rate'] * 100
fraud_by_maker = fraud_by_maker[fraud_by_maker['Total_Claims'] >= 500]
fraud_by_maker = fraud_by_maker.sort_values('Fraud_Rate', ascending=False)
print(f"Number of makers with 500+ claims: {len(fraud_by_maker)}")
print(f"\nFraud rate range: {fraud_by_maker['Fraud_Rate'].min():.1f}% - {fraud_by_maker['Fraud_Rate'].max():.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_10 = fraud_by_maker.head(10)
axes[0].barh(range(len(top_10)), top_10['Fraud_Rate'], color='darkred', alpha=0.7)
axes[0].set_yticks(range(len(top_10)))
axes[0].set_yticklabels(top_10.index, fontsize=12, fontweight='bold')
axes[0].set_xlabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Makers: Highest Fraud Rates', fontsize=16, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
axes[0].axvline(18.1, color='blue', linestyle='--', linewidth=2, label=f'18.1%')
axes[0].legend()

bottom_10 = fraud_by_maker.tail(10)
axes[1].barh(range(len(bottom_10)), bottom_10['Fraud_Rate'], color='green', alpha=0.7)
axes[1].set_yticks(range(len(bottom_10)))
axes[1].set_yticklabels(bottom_10.index, fontsize=12, fontweight='bold')
axes[1].set_xlabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Bottom 10 Makers: Lowest Fraud Rates', fontsize=16, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)
axes[1].axvline(18.1, color='blue', linestyle='--', linewidth=2, label=f'18.1%')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{figures_path}/02_fraud_by_maker.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 3. Fraud Rate by Fuel Type ---
fraud_by_fuel = data.groupby('Fuel_type_grouped')['anomaly_present'].agg(['count', 'sum', 'mean'])
fraud_by_fuel.columns = ['Total_Claims', 'Fraud_Count', 'Fraud_Rate']
fraud_by_fuel['Fraud_Rate'] = fraud_by_fuel['Fraud_Rate'] * 100
fraud_by_fuel = fraud_by_fuel.sort_values('Fraud_Rate', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['darkred' if rate > 18.1 else 'green' for rate in fraud_by_fuel['Fraud_Rate']]
bars = axes[0].bar(range(len(fraud_by_fuel)), fraud_by_fuel['Fraud_Rate'],
                   color=colors, alpha=0.7, edgecolor='black')
axes[0].set_xticks(range(len(fraud_by_fuel)))
axes[0].set_xticklabels(fraud_by_fuel.index, rotation=45, ha='right', fontsize=12)
axes[0].set_ylabel('Fraud Rate (%)', fontsize=14, fontweight='bold')
axes[0].set_title('Fraud Rate by Fuel Type', fontsize=16, fontweight='bold')
axes[0].axhline(18.1, color='blue', linestyle='--', linewidth=2, label=f'Overall: 18.1%')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].barh(range(len(fraud_by_fuel)), fraud_by_fuel['Total_Claims'],
             color='steelblue', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(fraud_by_fuel)))
axes[1].set_yticklabels(fraud_by_fuel.index, fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Claims', fontsize=14, fontweight='bold')
axes[1].set_title('Sample Size by Fuel Type', fontsize=16, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/03_fraud_by_fuel_type.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 4. Fraud Rate by price_cost_ratio ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

data.boxplot(column='price_cost_ratio', by='anomaly_present', ax=axes[0])
axes[0].set_title('Price-Cost Ratio by Fraud Status', fontsize=14)
axes[0].set_xlabel('Claim Type')
axes[0].set_ylabel('Price-Cost Ratio (%)')
plt.sca(axes[0])
plt.xticks([1, 2], ['Legitimate', 'Fraudulent'])
axes[0].get_figure().suptitle('')

fraud_by_bin = data.groupby('ratio_bin')['anomaly_present'].agg(['count', 'mean'])
fraud_by_bin['fraud_rate'] = fraud_by_bin['mean'] * 100
labels = ['0-5%', '5-10%', '10-20%', '20-40%', '40-60%', '60-100%']
colors = ['green' if r < 18.1 else 'darkred' for r in fraud_by_bin['fraud_rate']]
axes[1].bar(range(len(fraud_by_bin)), fraud_by_bin['fraud_rate'],
            color=colors, alpha=0.7, edgecolor='black')
axes[1].set_xticks(range(len(fraud_by_bin)))
axes[1].set_xticklabels(labels, rotation=45, ha='right')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].set_xlabel('Repair Cost as % of Vehicle Price')
axes[1].set_title('Fraud Rate Increases with Ratio', fontsize=14)
axes[1].axhline(18.1, color='blue', linestyle='--', linewidth=2, label='Overall: 18.1%')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/04_fraud_by_price_cost_ratio.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 5. Fraud Rate by labour cost per hour ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cph_fraud = data.groupby('cph_bin')['anomaly_present'].agg(['count', 'mean'])
cph_fraud['fraud_rate'] = cph_fraud['mean'] * 100
labels = ['£0-50', '£50-100', '£100-150', '£150-200', '£200-300', '£300-500', '£500-700', '£700-1000']

axes[0].plot(range(len(cph_fraud)), cph_fraud['fraud_rate'],
             linewidth=3, color='darkred', marker='o', markersize=10,
             markeredgecolor='black', markeredgewidth=1.5)
axes[0].axhline(18.1, color='blue', linestyle='--', linewidth=2, label='Overall: 18.1%', alpha=0.7)
axes[0].fill_between(range(len(cph_fraud)), 18.1, cph_fraud['fraud_rate'],
                     where=(cph_fraud['fraud_rate'] > 18.1),
                     alpha=0.3, color='red', label='Above average')
axes[0].set_xticks(range(len(cph_fraud)))
axes[0].set_xticklabels(labels, rotation=45, ha='right')
axes[0].set_ylabel('Fraud Rate (%)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Cost Per Hour Range', fontweight='bold', fontsize=12)
axes[0].set_title('Fraud Rate Escalates with Labor Rate', fontweight='bold', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, max(cph_fraud['fraud_rate']) + 5)

axes[1].barh(range(len(cph_fraud)), cph_fraud['count'],
             color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
axes[1].set_yticks(range(len(cph_fraud)))
axes[1].set_yticklabels(labels, fontsize=10)
axes[1].set_xlabel('Number of Claims', fontweight='bold', fontsize=12)
axes[1].set_title('Sample Size by Rate Range', fontweight='bold', fontsize=13)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/05_fraud_by_cost_per_hour.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 6. Fraud Rate by repair complexity ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

complexity_fraud = data.groupby('repair_complexity')['anomaly_present'].agg(['count', 'sum', 'mean'])
complexity_fraud['fraud_rate'] = complexity_fraud['mean'] * 100

axes[0].plot(complexity_fraud.index, complexity_fraud['fraud_rate'],
             linewidth=3, color='darkred', marker='o', markersize=12,
             markeredgecolor='black', markeredgewidth=2)
axes[0].axhline(18.1, color='blue', linestyle='--', linewidth=2, label='Overall: 18.1%', alpha=0.7)
axes[0].fill_between(complexity_fraud.index, 18.1, complexity_fraud['fraud_rate'],
                     where=(complexity_fraud['fraud_rate'] > 18.1),
                     alpha=0.3, color='red', label='Above average')
axes[0].set_xlabel('Repair Complexity Level', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Fraud Rate (%)', fontweight='bold', fontsize=12)
axes[0].set_title('Fraud Rate by Complexity', fontweight='bold', fontsize=13)
axes[0].set_xticks(complexity_fraud.index)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

axes[1].bar(complexity_fraud.index, complexity_fraud['count'],
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Repair Complexity Level', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Number of Claims', fontweight='bold', fontsize=12)
axes[1].set_title('Claim Volume by Complexity', fontweight='bold', fontsize=13)
axes[1].set_xticks(complexity_fraud.index)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/06_fraud_by_repair_complexity.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 7. Repair hour distribution ---
fig = plt.figure(figsize=(14, 12))

ax1 = plt.subplot(2, 1, 1)
fraud_groups = [data[data['anomaly_present']==0]['repair_hours'],
                data[data['anomaly_present']==1]['repair_hours']]
ax1.boxplot(fraud_groups, labels=['Legitimate', 'Fraudulent'])
ax1.set_title('Repair Hours by Fraud Status', fontweight='bold')
ax1.set_xlabel('Claim Type', fontweight='bold')
ax1.set_ylabel('Repair Hours', fontweight='bold')
ax1.grid(alpha=0.3)

hours_fraud = data.groupby('hours_bin')['anomaly_present'].agg(['count', 'mean'])
hours_fraud['fraud_rate'] = hours_fraud['mean'] * 100
labels = ['0-5h', '5-10h', '10-20h', '20-50h', '50-100h', '100-500h']

ax2 = plt.subplot(2, 2, 3)
ax2.plot(range(len(hours_fraud)), hours_fraud['fraud_rate'],
         linewidth=3, color='darkred', marker='o', markersize=10,
         markeredgecolor='black', markeredgewidth=1.5)
ax2.axhline(18.1, color='blue', linestyle='--', linewidth=2, label='Overall: 18.1%', alpha=0.7)
ax2.fill_between(range(len(hours_fraud)), 18.1, hours_fraud['fraud_rate'],
                 where=(hours_fraud['fraud_rate'] > 18.1),
                 alpha=0.3, color='red', label='Above average')
ax2.set_xticks(range(len(hours_fraud)))
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.set_ylabel('Fraud Rate (%)', fontweight='bold')
ax2.set_xlabel('Repair Hours Range', fontweight='bold')
ax2.set_title('Fraud Rate by Repair Duration', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

ax3 = plt.subplot(2, 2, 4)
ax3.barh(range(len(hours_fraud)), hours_fraud['count'],
         color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(hours_fraud)))
ax3.set_yticklabels(labels)
ax3.set_xlabel('Number of Claims', fontweight='bold')
ax3.set_title('Sample Size by Hours Range', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/07_repair_hours_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 8. Correlation Heatmap ---
numerical_features = [
    'Price', 'Runned_Miles', 'repair_cost', 'repair_hours',
    'repair_complexity', 'cost_per_hour', 'price_cost_ratio',
    'repair_lag_days', 'anomaly_present']

corr_matrix = data[numerical_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax, annot_kws={'fontsize': 9})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(f'{figures_path}/08_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ EDA complete")
print(f"Figures saved to: {figures_path}")
