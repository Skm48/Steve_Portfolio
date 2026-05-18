import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

figures_path = 'projects/early-fraud-detection-ml/outputs/figures'
os.makedirs(figures_path, exist_ok=True)

# -------------------------
# MODEL COMPARISON SUMMARY
# -------------------------

print("Model Comparison Summary (Test set: 77,718)")
print(f"{'Model':<25} {'Caught Fraud':>14} {'Missed Fraud':>14} {'Wrongly Flagged':>16} {'AUC':>8}")
print(f"{'Logistic Regression':<25} {4:>14} {14336:>14} {4:>16} {0.528:>8.3f}")
print(f"{'RF + SMOTE':<25} {3860:>14} {10480:>14} {4576:>16} {0.666:>8.3f}")
print(f"{'XGBoost':<25} {8361:>14} {5979:>14} {17705:>16} {0.696:>8.3f}")
print(f"{'MLP':<25} {4048:>14} {10292:>14} {6612:>16} {0.656:>8.3f}")

# -------------------------
# COMPARISON CHART
# -------------------------

models = ['Logistic\nRegression', 'RF +\nSMOTE', 'XGBoost', 'MLP']
caught_fraud    = [4,     3860,  8361,  4048]
missed_fraud    = [14336, 10480, 5979,  10292]
wrongly_flagged = [4,     4576,  17705, 6612]
auc_scores      = [0.528, 0.666, 0.696, 0.656]

x = np.arange(len(models))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Caught fraud vs wrongly flagged
bars1 = axes[0].bar(x - width, caught_fraud, width, label='Caught Fraud',
                    color='steelblue', alpha=0.7, edgecolor='black')
bars2 = axes[0].bar(x,          missed_fraud, width, label='Missed Fraud',
                    color='grey', alpha=0.7, edgecolor='black')
bars3 = axes[0].bar(x + width,  wrongly_flagged, width, label='Wrongly Flagged',
                    color='darkred', alpha=0.7, edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=11)
axes[0].set_ylabel('Number of Claims', fontweight='bold', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: AUC scores
colors = ['darkred' if a < 0.70 else 'green' for a in auc_scores]
axes[1].bar(x, auc_scores, color=colors, alpha=0.7, edgecolor='black')
axes[1].axhline(0.70, color='blue', linestyle='--', linewidth=2, label='AUC = 0.70 threshold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=11)
axes[1].set_ylabel('AUC Score', fontweight='bold', fontsize=12)
axes[1].set_title('AUC Score by Model', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/13_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# CONCLUSION
# -------------------------

print("""
=======================================================================
CONCLUSION
=======================================================================

This project investigated whether insurance fraud can be predicted at
the point of initial claim reporting, using only pre-repair vehicle
characteristics and estimated repair details.

Key findings:

1. Pre-repair features alone provide limited predictive power.
   All models fell below AUC 0.70, confirming that fraud signals are
   stronger in post-repair behavioural data.

2. Logistic Regression failed entirely (recall = 0.00), confirming
   the non-linear nature of fraud patterns identified in the EDA.

3. XGBoost achieved the highest recall (58%) but at a cost of 17,705
   false positives — a ratio of 2.1 innocent claimants investigated
   per fraudster caught.

4. RF + SMOTE offered the most balanced trade-off:
   - 27% recall
   - 4,576 false positives
   - 1.2 innocent claimants investigated per fraudster caught

5. MLP was comparable to RF + SMOTE but with more false positives
   (6,612), consistent with literature showing tree-based models
   outperform neural networks on tabular data.

Recommendation:
RF + SMOTE is the most suitable model for deployment — not as a
standalone decision tool, but as a claim prioritisation system that
flags suspicious cases for human review without automatically
delaying legitimate repairs.
=======================================================================
""")
