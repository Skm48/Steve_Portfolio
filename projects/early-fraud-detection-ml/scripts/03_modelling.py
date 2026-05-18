import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import seaborn as sns
import os

# Load cleaned data
Cleaned_data = pd.read_csv('projects/early-fraud-detection-ml/data/cleaned/cleaned_data.csv')
print(f"Dataset shape: {Cleaned_data.shape}")
print(f"Rows: {len(Cleaned_data)}")
print(f"Columns: {len(Cleaned_data.columns)}")
Cleaned_data.head(4).T

# Output figures folder
figures_path = 'projects/early-fraud-detection-ml/outputs/figures'
os.makedirs(figures_path, exist_ok=True)

# -------------------------
# STEP 1: Feature Selection and Preparation
# -------------------------

categorical_features = ['Maker', 'Bodytype', 'Fuel_type_grouped', 'Gearbox']
numerical_features = ['Price', 'Runned_Miles', 'Reg_year', 'issue_id', 'repair_complexity']

# Encode categorical features
le_dict = {}
Cleaned_data_model = Cleaned_data.copy()
for col in categorical_features:
    le_dict[col] = LabelEncoder()
    Cleaned_data_model[col + '_encoded'] = le_dict[col].fit_transform(Cleaned_data_model[col])

# Final feature list
feature_cols = [col + '_encoded' for col in categorical_features] + numerical_features
X = Cleaned_data_model[feature_cols]
y = Cleaned_data_model['anomaly_present']

print("Features used for early prediction:")
for f in feature_cols:
    print(f" {f}")
print(f"\nDataset size: {len(X)}")
print(f"Fraud rate: {y.mean()*100:.2f}%")

# -------------------------
# STEP 2: Train/Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      random_state=42,
                                                      stratify=y)
print(f"Training set: {len(X_train)} ({y_train.mean()*100:.2f}% fraud)")
print(f"Test set: {len(X_test)} ({y_test.mean()*100:.2f}% fraud)")

# -------------------------
# STEP 3: Baseline Model — Logistic Regression
# -------------------------

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

print("\nBaseline Model: Logistic Regression")
print(classification_report(y_test, y_pred_baseline, target_names=['Legitimate', 'Fraudulent']))
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print("Confusion Matrix:")
print(f" True Negatives (correct legitimate): {cm_baseline[0][0]}")
print(f" False Positives (wrongly flagged): {cm_baseline[0][1]}")
print(f" False Negatives (missed fraud): {cm_baseline[1][0]}")
print(f" True Positives (caught fraud): {cm_baseline[1][1]}")

# -------------------------
# STEP 4: Random Forest (no SMOTE)
# -------------------------

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Model")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraudulent']))
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(f" True Negatives (correct legitimate): {cm_rf[0][0]}")
print(f" False Positives (wrongly flagged): {cm_rf[0][1]}")
print(f" False Negatives (missed fraud): {cm_rf[1][0]}")
print(f" True Positives (caught fraud): {cm_rf[1][1]}")

# -------------------------
# STEP 5: Random Forest + SMOTE
# -------------------------

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: Legitimate={sum(y_train==0)}, Fraud={sum(y_train==1)}")
print(f"After SMOTE: Legitimate={sum(y_train_smote==0)}, Fraud={sum(y_train_smote==1)}")

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

print("\nRandom Forest with SMOTE")
print(classification_report(y_test, y_pred_smote, target_names=['Legitimate', 'Fraudulent']))
cm_smote = confusion_matrix(y_test, y_pred_smote)
print("Confusion Matrix:")
print(f" True Negatives (correct legitimate): {cm_smote[0][0]}")
print(f" False Positives (wrongly flagged): {cm_smote[0][1]}")
print(f" False Negatives (missed fraud): {cm_smote[1][0]}")
print(f" True Positives (caught fraud): {cm_smote[1][1]}")

# -------------------------
# STEP 6: XGBoost
# -------------------------

scale = sum(y_train == 0) / sum(y_train == 1)
xgb_model = XGBClassifier(n_estimators=100,
                           scale_pos_weight=scale,
                           random_state=42,
                           eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Model")
print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate', 'Fraudulent']))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print("Confusion Matrix:")
print(f" True Negatives (correct legitimate): {cm_xgb[0][0]}")
print(f" False Positives (wrongly flagged): {cm_xgb[0][1]}")
print(f" False Negatives (missed fraud): {cm_xgb[1][0]}")
print(f" True Positives (caught fraud): {cm_xgb[1][1]}")

# -------------------------
# STEP 7: MLP
# -------------------------

smote = SMOTE(random_state=42)
X_train_mlp, y_train_mlp = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: Legitimate={sum(y_train_mlp==0)}, Fraud={sum(y_train_mlp==1)}")

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    ))
])

print("\nTraining MLP (128 -> 64 -> 32)...")
mlp_pipeline.fit(X_train_mlp, y_train_mlp)
print(f"Training stopped after {mlp_pipeline.named_steps['clf'].n_iter_} epochs")

y_pred_mlp = mlp_pipeline.predict(X_test)

print("\nMLP Classifier Results")
print(classification_report(y_test, y_pred_mlp, target_names=['Legitimate', 'Fraudulent']))
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print("Confusion Matrix:")
print(f"True Negatives (correct legitimate): {cm_mlp[0][0]}")
print(f"False Positives (wrongly flagged): {cm_mlp[0][1]}")
print(f"False Negatives (missed fraud): {cm_mlp[1][0]}")
print(f"True Positives (caught fraud): {cm_mlp[1][1]}")
print(f"Epochs: {mlp_pipeline.named_steps['clf'].n_iter_}")

y_prob_mlp = mlp_pipeline.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
auc_mlp = auc(fpr_mlp, tpr_mlp)
print(f"MLP AUC: {auc_mlp:.3f}")

# -------------------------
# STEP 8: Confusion Matrices
# -------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_baseline), annot=True, fmt='d', cmap='Blues',
            ax=axes[0], xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
axes[0].set_title('Logistic Regression (Baseline)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontweight='bold')
axes[0].set_xlabel('Predicted', fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_smote), annot=True, fmt='d', cmap='Greens',
            ax=axes[1], xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
axes[1].set_title('Random Forest + SMOTE', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual', fontweight='bold')
axes[1].set_xlabel('Predicted', fontweight='bold')

sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Reds',
            ax=axes[2], xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
axes[2].set_title('XGBoost', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Actual', fontweight='bold')
axes[2].set_xlabel('Predicted', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{figures_path}/09_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# STEP 9: Feature Importance Comparison
# -------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rf_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_smote.feature_importances_
}).sort_values('Importance', ascending=True)

axes[0].barh(range(len(rf_imp)), rf_imp['Importance'],
             color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(rf_imp)))
axes[0].set_yticklabels(rf_imp['Feature'])
axes[0].set_xlabel('Importance', fontweight='bold')
axes[0].set_title('RF + SMOTE Feature Importance', fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

xgb_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=True)

axes[1].barh(range(len(xgb_imp)), xgb_imp['Importance'],
             color='darkred', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(xgb_imp)))
axes[1].set_yticklabels(xgb_imp['Feature'])
axes[1].set_xlabel('Importance', fontweight='bold')
axes[1].set_title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{figures_path}/10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# STEP 10: ROC Curves
# -------------------------

fig, ax = plt.subplots(figsize=(8, 6))

y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
auc_base = auc(fpr_base, tpr_base)
ax.plot(fpr_base, tpr_base, label=f'Logistic Regression (AUC = {auc_base:.3f})',
        color='grey', linewidth=2)

y_prob_smote = rf_smote.predict_proba(X_test)[:, 1]
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_prob_smote)
auc_smote = auc(fpr_smote, tpr_smote)
ax.plot(fpr_smote, tpr_smote, label=f'RF + SMOTE (AUC = {auc_smote:.3f})',
        color='steelblue', linewidth=2)

y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
auc_xgb = auc(fpr_xgb, tpr_xgb)
ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})',
        color='darkred', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curves: Model Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{figures_path}/11_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# STEP 11: False Positive Impact Analysis
# -------------------------

test_data = Cleaned_data_model.iloc[X_test.index].copy()
test_data['pred_smote'] = y_pred_smote
false_positives = test_data[(test_data['anomaly_present'] == 0) & (test_data['pred_smote'] == 1)]
true_negatives  = test_data[(test_data['anomaly_present'] == 0) & (test_data['pred_smote'] == 0)]

print("False Positive Impact Analysis (RF + SMOTE)")
print(f"Total false positives: {len(false_positives)}")
print(f"\nRepair Cost ")
print(f"FP mean repair cost: £{false_positives['repair_cost'].mean():.2f}")
print(f"TN mean repair cost: £{true_negatives['repair_cost'].mean():.2f}")
print(f"\nRepair Delay")
print(f"FP mean repair lag (days): {false_positives['repair_lag_days'].mean():.2f}")
print(f"TN mean repair lag (days): {true_negatives['repair_lag_days'].mean():.2f}")
print(f"\nRepair Complexity")
print(f"FP mean complexity: {false_positives['repair_complexity'].mean():.2f}")
print(f"TN mean complexity: {true_negatives['repair_complexity'].mean():.2f}")

print("\nModel Comparison Summary (Test set: 77,718)")
print(f"{'Model':<25} {'Caught Fraud':>14} {'Missed Fraud':>14} {'Wrongly Flagged':>16}")
print(f"{'Logistic Regression':<25} {cm_baseline[1][1]:>14} {cm_baseline[1][0]:>14} {cm_baseline[0][1]:>16}")
print(f"{'RF + SMOTE':<25} {cm_smote[1][1]:>14} {cm_smote[1][0]:>14} {cm_smote[0][1]:>16}")
print(f"{'XGBoost':<25} {cm_xgb[1][1]:>14} {cm_xgb[1][0]:>14} {cm_xgb[0][1]:>16}")

# False Positive Rate by Price Segment
test_data['pred_base'] = y_pred_baseline
test_data['pred_xgb']  = y_pred_xgb

print("False Positive Rate by Price Segment")
print(f"{'Segment':<25} {'Logistic':>10} {'RF+SMOTE':>10} {'XGBoost':>10}")
for seg in ['Budget (£0-5k)', 'Mid-range (£5k-15k)', 'Premium (£15k-30k)', 'Luxury (£30k+)']:
    legit = test_data[(test_data['anomaly_present'] == 0) & (test_data['price_segment'] == seg)]
    total = len(legit)
    base_fp  = len(legit[legit['pred_base']  == 1]) / total * 100
    smote_fp = len(legit[legit['pred_smote'] == 1]) / total * 100
    xgb_fp   = len(legit[legit['pred_xgb']   == 1]) / total * 100
    print(f"{seg:<25} {base_fp:>9.2f}% {smote_fp:>9.2f}% {xgb_fp:>9.2f}%")

smote_fp_vals = [6.06, 7.41, 7.38, 9.54]
xgb_fp_vals   = [23.69, 30.51, 27.94, 28.23]
segments      = ['Budget\n(£0-5k)', 'Mid-range\n(£5k-15k)', 'Premium\n(£15k-30k)', 'Luxury\n(£30k+)']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(segments))
width = 0.35
ax.bar(x - width/2, smote_fp_vals, width, label='RF + SMOTE',
       color='steelblue', alpha=0.7, edgecolor='black')
ax.bar(x + width/2, xgb_fp_vals, width, label='XGBoost',
       color='darkred', alpha=0.7, edgecolor='black')
ax.set_ylabel('False Positive Rate (%)', fontweight='bold')
ax.set_title('Legitimate Claims Wrongly Flagged by Price Segment',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(segments)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{figures_path}/12_false_positive_by_segment.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Modelling complete")
print(f"Figures saved to: {figures_path}")
