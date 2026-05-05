import os
import pickle
import numpy as np
import pandas as pd

from django.conf import settings
from django.shortcuts import render

# ── Load model once when Django starts ───────────────────────────────────────
MODEL_PATH = os.path.join(settings.BASE_DIR, "attrition_model.pkl")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model     = saved["model"]
threshold = saved["threshold"]


def apply_feature_engineering(df):
    X = df.copy()

    features_to_exclude = [
        "EmployeeCount", "EmployeeNumber",
        "StandardHours", "StockOptionLevel",
        "Over18"
    ]
    X = X.drop(columns=[c for c in features_to_exclude if c in X.columns], errors="ignore")

    X["OverTime_binary"]   = (X["OverTime"] == "Yes").astype(int)
    X["SatisfactionScore"] = (X["JobSatisfaction"] +
                              X["EnvironmentSatisfaction"] +
                              X["RelationshipSatisfaction"] +
                              X["WorkLifeBalance"]) / 4
    X["TenureRatio"]       = X["YearsAtCompany"] / (X["TotalWorkingYears"] + 1)
    X["StagnationScore"]   = X["YearsSinceLastPromotion"] / (X["YearsInCurrentRole"] + 1)
    X["HighRiskFlag"]      = (
        (X["OverTime"] == "Yes") &
        (X["MaritalStatus"] == "Single") &
        (X["JobLevel"] <= 2)
    ).astype(int)
    X["IncomePerLevel"]    = X["MonthlyIncome"] / (X["JobLevel"] + 1)

    return X


def predict_csv(request):
    results        = None
    error          = None
    employee_count = 0
    summary        = None

    if request.method == "POST":
        csv_file = request.FILES.get("csv_file")

        if not csv_file:
            error = "Please upload a CSV file."

        elif not csv_file.name.endswith(".csv"):
            error = "Please upload a valid CSV file."

        else:
            try:
                df = pd.read_csv(csv_file)
                employee_count = len(df)

                if employee_count == 0:
                    error = "The CSV file is empty."

                else:
                    columns_to_remove = [
                        "Attrition", "Outlier_IQR", "Outlier_Zscore",
                        "Outlier_IsolationForest", "Outlier_LOF",
                        "OutlierVotes", "FinalOutlier",
                    ]
                    prediction_data = df.drop(
                        columns=[c for c in columns_to_remove if c in df.columns],
                        errors="ignore"
                    )

                    prediction_data = apply_feature_engineering(prediction_data)

                    probas = model.predict_proba(prediction_data)[:, 1]

                    results = [
                        {
                            "Prediction":  "Yes" if prob >= 0.5 else "No",
                            "Probability": f"{prob:.1%}",
                            "RiskBand": (
                                "Critical" if prob >= 0.7 else
                                "High"     if prob >= 0.5 else
                                "Medium"   if prob >= 0.3 else
                                "Low"
                            )
                        }
                        for prob in probas
                    ]

                    # ── Build summary ─────────────────────────────────────
                    summary = {
                        "total":    len(results),
                        "yes":      sum(1 for r in results if r["Prediction"] == "Yes"),
                        "no":       sum(1 for r in results if r["Prediction"] == "No"),
                        "critical": sum(1 for r in results if r["RiskBand"] == "Critical"),
                        "high":     sum(1 for r in results if r["RiskBand"] == "High"),
                        "medium":   sum(1 for r in results if r["RiskBand"] == "Medium"),
                        "low":      sum(1 for r in results if r["RiskBand"] == "Low"),
                    }

            except Exception as e:
                error = f"Prediction failed: {str(e)}"

    return render(request, "predict_csv.html", {
        "results":        results,
        "error":          error,
        "employee_count": employee_count,
        "summary":        summary,
    })