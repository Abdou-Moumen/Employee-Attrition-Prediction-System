# =============================================================
# EMPLOYEE ATTRITION PREDICTION PROJECT
# Includes:
# 1. Data Loading
# 2. Outlier Detection
# 3. Data Cleaning
# 4. Feature Engineering
# 5. Ridge / Lasso Logistic Regression
# 6. Threshold Optimization
# 7. Risk Scoring
# 8. Save Model
# =============================================================


# =============================================================
# 1. IMPORT LIBRARIES
# =============================================================
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline


# =============================================================
# 2. LOAD DATASET
# =============================================================
data = pd.read_csv(
    r"C:\Users\Abdou\Desktop\supervised\HR.csv"
)

print("Dataset Shape:", data.shape)
data.head()


# =============================================================
# 3. OUTLIER DETECTION
# Using 4 methods:
# - IQR
# - Z-score
# - Isolation Forest
# - Local Outlier Factor
# =============================================================
df = data.copy()

# Select numeric columns
numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Remove useless columns
cols_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

numeric_cols = [
    col for col in numeric_cols
    if col not in cols_to_remove
]

# Fill missing values
X_num = df[numeric_cols].fillna(
    df[numeric_cols].median()
)

# Standardize values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)


# -------------------------------------------------------------
# IQR METHOD
# -------------------------------------------------------------
df["Outlier_IQR"] = False

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["Outlier_IQR"] = df["Outlier_IQR"] | (
        (df[col] < lower) |
        (df[col] > upper)
    )


# -------------------------------------------------------------
# Z-SCORE METHOD
# -------------------------------------------------------------
z_scores = np.abs(X_scaled)

df["Outlier_Zscore"] = (
    z_scores > 3
).any(axis=1)


# -------------------------------------------------------------
# ISOLATION FOREST
# -------------------------------------------------------------
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

iso_pred = iso.fit_predict(X_num)

df["Outlier_IForest"] = iso_pred == -1


# -------------------------------------------------------------
# LOCAL OUTLIER FACTOR
# -------------------------------------------------------------
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

lof_pred = lof.fit_predict(X_scaled)

df["Outlier_LOF"] = lof_pred == -1


# -------------------------------------------------------------
# FINAL OUTLIER DECISION
# Keep rows marked by 3+ methods
# -------------------------------------------------------------
outlier_cols = [
    "Outlier_IQR",
    "Outlier_Zscore",
    "Outlier_IForest",
    "Outlier_LOF"
]

df["OutlierVotes"] = df[outlier_cols].sum(axis=1)

df["FinalOutlier"] = df["OutlierVotes"] >= 3

print("Detected Outliers:")
print(df["FinalOutlier"].value_counts())


# =============================================================
# 4. REMOVE FINAL OUTLIERS
# =============================================================
clean_data = df[
    df["FinalOutlier"] == False
].copy()

print("Clean Dataset Shape:", clean_data.shape)


# =============================================================
# 5. FEATURE SELECTION
# =============================================================
features_to_exclude = [
    "EmployeeCount",
    "EmployeeNumber",
    "StandardHours",
    "StockOptionLevel",
    "Over18"
]

X_raw = clean_data.drop(
    columns=["Attrition"] + features_to_exclude,
    errors="ignore"
)

y = clean_data["Attrition"]


# =============================================================
# 6. FEATURE ENGINEERING
# =============================================================
X = X_raw.copy()

# Overtime binary
X["OverTime_binary"] = (
    X["OverTime"] == "Yes"
).astype(int)

# Satisfaction score
X["SatisfactionScore"] = (
    X["JobSatisfaction"] +
    X["EnvironmentSatisfaction"] +
    X["RelationshipSatisfaction"] +
    X["WorkLifeBalance"]
) / 4

# Tenure ratio
X["TenureRatio"] = (
    X["YearsAtCompany"] /
    (X["TotalWorkingYears"] + 1)
)

# Promotion stagnation
X["StagnationScore"] = (
    X["YearsSinceLastPromotion"] /
    (X["YearsInCurrentRole"] + 1)
)

# High risk employees
X["HighRiskFlag"] = (
    (X["OverTime"] == "Yes") &
    (X["MaritalStatus"] == "Single") &
    (X["JobLevel"] <= 2)
).astype(int)

# Income efficiency
X["IncomePerLevel"] = (
    X["MonthlyIncome"] /
    (X["JobLevel"] + 1)
)


# =============================================================
# 7. DEFINE COLUMN TYPES
# =============================================================
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# =============================================================
# 8. TRAIN / TEST SPLIT
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# =============================================================
# 9. PREPROCESSING
# =============================================================
preprocessor = ColumnTransformer(
    transformers=[

        ("num",
         StandardScaler(),
         numeric_features),

        ("cat",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)


# =============================================================
# 10. CHOOSE MODEL
# Ridge = L2
# Lasso = L1
# =============================================================

# -------- RIDGE MODEL --------
ridge_model = ImbPipeline(steps=[

    ("prep", preprocessor),

    ("smote", SMOTETomek(random_state=42)),

    ("clf", LogisticRegression(
        penalty="l2",
        C=0.1,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ))
])

# -------- LASSO MODEL --------
lasso_model = ImbPipeline(steps=[

    ("prep", preprocessor),

    ("smote", SMOTETomek(random_state=42)),

    ("clf", LogisticRegression(
        penalty="l1",
        C=0.1,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ))
])


# =============================================================
# 11. SELECT BEST MODEL
# =============================================================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Ridge": ridge_model,
    "Lasso": lasso_model
}

best_auc = 0
best_name = None
best_model = None

for name, model in models.items():

    probs = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    auc = roc_auc_score(
        (y == "Yes").astype(int),
        probs
    )

    print(name, "ROC AUC:", round(auc, 4))

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = model

print("\nBest Model:", best_name)


# =============================================================
# 12. FIND BEST THRESHOLD
# =============================================================
y_proba_oof = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

precision_arr, recall_arr, thresholds = precision_recall_curve(
    y,
    y_proba_oof,
    pos_label="Yes"
)

f1_scores = (
    2 * precision_arr * recall_arr /
    (precision_arr + recall_arr + 1e-9)
)

best_threshold = thresholds[np.argmax(f1_scores)]

print("Best Threshold:", round(best_threshold, 3))


# =============================================================
# 13. TRAIN FINAL MODEL
# =============================================================
best_model.fit(X_train, y_train)

y_proba_test = best_model.predict_proba(
    X_test
)[:, 1]

y_pred_test = np.where(
    y_proba_test >= best_threshold,
    "Yes",
    "No"
)


# =============================================================
# 14. FINAL EVALUATION
# =============================================================
print("=" * 55)
print("FINAL MODEL RESULTS")
print("=" * 55)

print("Model:", best_name)

print("ROC AUC:",
      round(roc_auc_score(y_test, y_proba_test), 3))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred_test,
    labels=["No", "Yes"]
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_test
))


# =============================================================
# 15. EMPLOYEE RISK SCORING
# =============================================================
risk_df = clean_data.copy()

risk_df["AttritionProbability"] = y_proba_oof

risk_df["Prediction"] = np.where(
    y_proba_oof >= best_threshold,
    "Yes",
    "No"
)

risk_df["RiskBand"] = pd.cut(
    y_proba_oof,
    bins=[0, 0.3, 0.5, 0.7, 1],
    labels=["Low", "Medium", "High", "Critical"]
)

print("\nRisk Summary:")
print(
    risk_df["RiskBand"].value_counts()
)


# =============================================================
# 16. SAVE MODEL
# =============================================================
with open("attrition_model.pkl", "wb") as f:

    pickle.dump({
        "model": best_model,
        "threshold": best_threshold,
        "model_name": best_name
    }, f)

print("\nModel Saved Successfully!")