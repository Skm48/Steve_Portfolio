import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("projects/fraud-risk-segmentation/data/cleaned/featured_data.csv")
data['vehicle_age'] = data['Adv_year'] - data['Reg_year']
# Select features
features = data[['Price', 'Runned_Miles', 'repair_complexity',
                    'repair_hours', 'cost_per_hour', 'price_cost_ratio', 'vehicle_age']]

# -------------------------
# SCALING
# -------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# -------------------------
# ELBOW METHOD
# -------------------------

inertia = []
K_range = range(2, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}")

# Plot elbow
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)', fontweight='bold')
ax.set_ylabel('Inertia', fontweight='bold')
ax.set_title('Elbow Method for Optimal K', fontsize=13, fontweight='bold')
 
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("projects/fraud-risk-segmentation/outputs/figures/elbow.png")
plt.close()

# -------------------------
# FINAL MODEL (k = 6)
# -------------------------

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered data
data.to_csv("projects/fraud-risk-segmentation/data/cleaned/clustered_data.csv", index=False)

print("✅ Clustering complete")
