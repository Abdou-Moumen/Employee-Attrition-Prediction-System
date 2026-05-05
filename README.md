# GitHub README.md

# Employee Attrition Prediction System

Predict employee attrition before they resign and vanish into another company’s onboarding process.

This project uses Machine Learning to identify employees at risk of leaving based on HR data.

---

## Features

* Data cleaning and preprocessing
* Multi-method outlier detection
* Feature engineering
* Ridge vs Lasso Logistic Regression comparison
* Class imbalance handling with SMOTE + Tomek
* Threshold optimization using F1-score
* Risk segmentation (Low / Medium / High / Critical)
* Export trained model with Pickle

---

## Tech Stack

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

---

## Project Structure

```bash
employee-attrition/
│── data/
│   └── HR.csv
│── notebooks/
│   └── attrition.ipynb
│── model/
│   └── attrition_model.pkl
│── requirements.txt
│── README.md
```

---

# Full Setup Guide (0 to Hero)

## 1. Install Python

Download Python from:

https://www.python.org/downloads/

During installation:

* Check **Add Python to PATH**

Verify:

```bash
python --version
```

---

## 2. Clone Repository

```bash
git clone https://github.com/yourusername/employee-attrition.git
cd employee-attrition
```

---

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

If it works, your terminal becomes emotionally attached to `(venv)`.

---

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn imbalanced-learn jupyter matplotlib seaborn
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 5. Create requirements.txt

```bash
pip freeze > requirements.txt
```

---

## 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```bash
notebooks/attrition.ipynb
```

---

## 7. Run the Project

Inside notebook:

* Load dataset
* Detect outliers
* Train models
* Compare Ridge vs Lasso
* Evaluate metrics
* Save best model

---

## 8. Saved Model

After training:

```bash
attrition_model.pkl
```

Load later:

```python
import pickle

with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC

Because accuracy alone lies sometimes.

---

## Why This Project Matters

Replacing employees is expensive.

This system helps HR teams detect attrition risk early and take action before resignation emails start flying.

---

## Future Improvements

* XGBoost / LightGBM
* SHAP Explainability
* Streamlit Dashboard
* Real-time HR Analytics API

---

## Author

Built by someone who chose violence against messy datasets.
