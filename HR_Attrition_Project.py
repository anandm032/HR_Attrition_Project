# HR Attrition Project
# =====================

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
df = pd.read_csv("hr_data.csv")

# Step 3: EDA / Visualization
print("\nDataset Info:\n", df.info())
print("\nFirst 5 Rows:\n", df.head())

# Attrition count
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Count")
plt.show()

# Department count
sns.countplot(x='Department', data=df)
plt.title("Department Count")
plt.show()

# JobRole distribution
sns.countplot(x='JobRole', data=df)
plt.title("Job Role Distribution")
plt.xticks(rotation=45)
plt.show()

# Overtime vs Attrition
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title("OverTime vs Attrition")
plt.show()

# =======================
# Step 4: Data Preprocessing
# =======================
label_enc = LabelEncoder()
for col in ['Department', 'JobRole', 'OverTime']:
    df[col] = label_enc.fit_transform(df[col])

# Encode target column Attrition
df['Attrition'] = label_enc.fit_transform(df['Attrition'])  # Yes=1, No=0

# Features & Target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# Step 5: Model Building & Evaluation
# =======================

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
# ================================
# STEP 6: MODEL EXPLAINABILITY
# ================================
import shap

print("\n" + "="*40)
print("STEP 6: MODEL EXPLAINABILITY")
print("="*40)

# 1. Feature Importance (Decision Tree)
feature_importances = dt_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance from Decision Tree:")
print(importance_df)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance (Decision Tree)")
plt.show()

# 2. SHAP Explainability
print("\nSHAP Explainability for Logistic Regression:")

# Use TreeExplainer for DecisionTreeClassifier
explainer_tree = shap.TreeExplainer(dt_model)
shap_values_tree = explainer_tree.shap_values(X_test)

# Summary plot for Decision Tree
shap.summary_plot(shap_values_tree, X_test, plot_type="bar")

# Logistic Regression (use KernelExplainer since it's linear)
explainer_lr = shap.KernelExplainer(log_reg.predict_proba, X_train, link="logit")
shap_values_lr = explainer_lr.shap_values(X_test[:50])  # only first 50 to save time

# Summary plot for Logistic Regression
shap.summary_plot(shap_values_lr, X_test[:50], plot_type="bar")
# ================================
# STEP 7: MODEL COMPARISON
# ================================
from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*40)
print("STEP 7: MODEL COMPARISON")
print("="*40)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))

# Accuracy Comparison
models = {
    "Logistic Regression": log_reg.score(X_test, y_test),
    "Decision Tree": dt_model.score(X_test, y_test),
    "Random Forest": rf_model.score(X_test, y_test)
}

print("\nModel Accuracy Comparison:")
for model, acc in models.items():
    print(f"{model}: {acc:.4f}")

# Bar Chart for Comparison
plt.figure(figsize=(6,4))
sns.barplot(x=list(models.keys()), y=list(models.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
# ================================
# STEP 8: SAVE & LOAD BEST MODEL
# ================================
import joblib

print("\n" + "="*40)
print("STEP 8: SAVE & LOAD MODEL")
print("="*40)

# Pick the best model based on accuracy
best_model_name = max(models, key=models.get)
print(f"Best Model: {best_model_name} with accuracy {models[best_model_name]:.4f}")

if best_model_name == "Logistic Regression":
    best_model = log_reg
elif best_model_name == "Decision Tree":
    best_model = dt_model
else:
    best_model = rf_model

# Save model
joblib.dump(best_model, "best_attrition_model.pkl")
print("✅ Model saved as best_attrition_model.pkl")

# Load model (test)
loaded_model = joblib.load("best_attrition_model.pkl")
sample_pred = loaded_model.predict(X_test[:5])
print("\nSample Prediction from Loaded Model:", sample_pred)
# ================================
# ================================
# STEP 9: EXPORT DATA FOR POWER BI
# ================================
print("\n" + "="*40)
print("STEP 9: EXPORT DATA FOR POWER BI")
print("="*40)

# Use df (your dataset variable) instead of data if you named it df
# ✅ Export cleaned dataset for Power BI
export_df = df[["Age", "Department", "MonthlyIncome", "YearsAtCompany", "OverTime", "Attrition"]]
export_df.to_csv("hr_attrition_export.csv", index=False)

print("Data exported successfully to hr_attrition_export.csv")


