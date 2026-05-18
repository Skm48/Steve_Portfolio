import pandas as pd
import matplotlib.pyplot as plt

# Load clustered data
data = pd.read_csv("projects/fraud-risk-segmentation/data/cleaned/clustered_data.csv")

print("Data loaded:", data.shape)


# -------------------------
# CLUSTER DISTRIBUTION
# -------------------------

cluster_counts = data['cluster'].value_counts().sort_index()

plt.figure()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel("Cluster")
plt.ylabel("Number of Claims")
plt.title("Cluster Distribution")

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/cluster_distribution.png")
plt.close()

# -------------------------
# CLUSTER SUMMARY (FROM REPORT)
# -------------------------

cluster_summary = data.groupby('cluster').agg(
    Size=('anomaly_present', 'count'),
    Fraud_Rate=('anomaly_present', 'mean'),
    Mean_Price=('Price', 'mean'),
    Mean_Mileage=('Runned_Miles', 'mean'),
    Mean_Age=('vehicle_age', 'mean'),
    Mean_Repair_Hours=('repair_hours', 'mean'),
    Mean_Complexity=('repair_complexity', 'mean'),
    Mean_Cost_Per_Hour=('cost_per_hour', 'mean'),
    Mean_Ratio=('price_cost_ratio', 'mean')
).round(2)

# Convert fraud rate to %
cluster_summary['Fraud_Rate'] = (cluster_summary['Fraud_Rate'] * 100).round(2)

print("\nCluster Summary:\n")
print(cluster_summary)

# Save table
cluster_summary.to_csv("projects/fraud-risk-segmentation/outputs/tables/cluster_summary.csv")

# -------------------------
# VISUALISATION
# -------------------------

clusters = cluster_summary.index

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Fraud rate
colors = ['green' if r < 18.5 else 'darkred' for r in cluster_summary['Fraud_Rate']]
axes[0,0].bar(range(6), cluster_summary['Fraud_Rate'], color=colors,
alpha=0.7, edgecolor='black')
axes[0,0].axhline(18.5, color='blue', linestyle='--', linewidth=2, label='Overall: 18.5%')
axes[0,0].set_xlabel('Cluster')
axes[0,0].set_ylabel('Fraud Rate (%)')
axes[0,0].set_title('Fraud Rate by Cluster')
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# Plot 2: Cluster size
axes[0,1].bar(range(6), cluster_summary['Size'], color='steelblue', alpha=0.7, edgecolor='black')
axes[0,1].set_xlabel('Cluster')
axes[0,1].set_ylabel('Number of Claims')
axes[0,1].set_title('Cluster Size')
axes[0,1].grid(axis='y', alpha=0.3)

# Plot 3: Mean price
axes[1,0].bar(range(6), cluster_summary['Mean_Price'], color='steelblue', 
              alpha=0.7, edgecolor='black')
axes[1,0].set_xlabel('Cluster')
axes[1,0].set_ylabel('Mean Price (£)')
axes[1,0].set_title('Mean Vehicle Price by Cluster')
axes[1,0].grid(axis='y', alpha=0.3)

# Plot 4: Repair hours
axes[1,1].bar(range(6), cluster_summary['Mean_Repair_Hours'], color='steelblue', 
              alpha=0.7, edgecolor='black')
axes[1,1].set_xlabel('Cluster')
axes[1,1].set_ylabel('Mean Repair Hours')
axes[1,1].set_title('Mean Repair Hours by Cluster')
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()

plt.savefig("projects/fraud-risk-segmentation/outputs/figures/cluster_profiles.png")
plt.close()

print("✅ Visualisation complete")
