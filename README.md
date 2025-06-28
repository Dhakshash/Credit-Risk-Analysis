# 🏦 Credit Risk Analysis with Machine Learning

This project applies machine learning techniques to predict borrower delinquency risk using anonymized financial data. It includes robust preprocessing, domain-relevant feature engineering using Weight of Evidence (WoE) and Information Value (IV), and model comparison across Logistic Regression, Random Forest, and LightGBM. The models are evaluated using metrics such as ROC-AUC, precision, recall, and F1-score—especially focusing on the minority (delinquent) class due to severe class imbalance.

---

## 📌 Problem Statement

The objective is to predict whether a borrower will default within the next two years (`SeriousDlqin2yrs = 1`), using features such as credit utilization, age, delinquency history, income, and credit lines. The challenge lies in handling an imbalanced dataset, extracting interpretable insights, and building models that are both predictive and compliant with financial risk practices.

---

## 🧪 Workflow Summary

1. **Exploratory Data Analysis (EDA)**
   - Identified skewed distributions (e.g., `MonthlyIncome`)
   - Detected missing values and outliers

2. **Data Preprocessing**
   - Imputed missing values:
     - `MonthlyIncome`: Median
     - `NumberOfDependents`: Mode
   - Treated outliers using WoE binning via `scorecardpy`

3. **Feature Engineering**
   - Applied WoE transformation to numerical variables
   - Calculated IV scores for all features
   - Selected features based on IV strength and credit domain relevance

4. **Model Development**
   - **Logistic Regression** (interpretable, regulator-friendly)
   - **Random Forest** (base and optimized with grid search)
   - **LightGBM** (fast, scalable, supports class imbalance)

5. **Model Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix
   - ROC-AUC and ROC curve visualizations

---

## 🔎 Feature Selection using Information Value (IV)

The IV scores guided the inclusion of features with strong discriminatory power between delinquent and non-delinquent borrowers.

| Feature                                 | IV Score | Predictive Power |
|----------------------------------------|----------|------------------|
| `RevolvingUtilizationOfUnsecuredLines` | 0.9878   | Strong           |
| `NumberOfTimes90DaysLate`              | 0.7795   | Strong           |
| `NumberOfTime30-59DaysPastDueNotWorse` | 0.6603   | Strong           |
| `NumberOfTime60-89DaysPastDueNotWorse` | 0.6344   | Strong           |
| `age`                                   | 0.2771   | Medium           |
| `MonthlyIncome`                         | 0.0922   | Weak (retained for context) |
| `NumberOfOpenCreditLinesAndLoans`      | 0.0635   | Weak (retained for coverage) |

Features such as `DebtRatio`, `NumberOfDependents`, and `NumberRealEstateLoansOrLines` were excluded due to weak IV.

---

## ⚖️ Class Imbalance Handling

| Model               | Technique Used                     |
|---------------------|------------------------------------|
| Logistic Regression | `class_weight='balanced'`         |
| Random Forest       | `class_weight='balanced'`         |
| LightGBM            | `scale_pos_weight` (auto-calculated) |

---

## 📊 Model Performance Summary

| Model                  | Accuracy | ROC-AUC | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) |
|------------------------|----------|---------|------------------|----------------------|--------------------|
| Logistic Regression    | 82.04%   | 0.88    | 81%              | 25%                  | 38%                |
| Random Forest (Base)   | 93.24%   | 0.845   | 8%               | 48%                  | 13%                |
| Random Forest (Tuned)  | 82.72%   | 0.88    | 76%              | 25%                  | 37%                |
| LightGBM               | 84.64%   | 0.87    | 71%              | 27%                  | 39%                |

📌 **Observation**:
- **Logistic Regression** provides transparent, audit-friendly predictions suitable for credit risk models.
- **LightGBM** achieved the best recall-performance balance on the minority class after tuning.
- **Random Forest (Tuned)** significantly reduced false negatives at the cost of false positives—an acceptable trade-off in credit risk.

---

## 📈 Confusion Matrix (Optimized Random Forest)

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **Actual 0**  |     1940    |     392     |
| **Actual 1**  |      40     |     128     |

✅ **True Positives (TP)**: 128  
✅ **True Negatives (TN)**: 1940  
❌ **False Positives (FP)**: 392  
❌ **False Negatives (FN)**: 40  

> ⚠️ The optimized model significantly improved recall on the minority (delinquent) class by reducing false negatives from 155 → 40, boosting detection of high-risk borrowers.

- Improved True Positives (TP)  
- Reduced False Negatives (FN) — critical in identifying risky borrowers

---

## 📁 Repository Structure

- `data/` – Input data and metadata  
  ◦ `credit_data.xlsx` – Anonymized credit dataset (safe for academic use)  
- `notebooks/Credit_Risk_Analysis.ipynb` – Full end-to-end Google collab notebook with EDA, feature engineering, modeling, and evaluation
- `requirements.txt` – List of Python dependencies for reproducing the analysis
- `.gitignore` – Git tracking exclusions (e.g., `.ipynb_checkpoints/`, `.csv`, `__pycache__/`, etc.)
- `README.md` – Project overview, results summary, and usage instructions






---

## 🧠 Technologies Used

- **Language**: Python 3.10+
- **Libraries**:
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – data visualization
  - `scikit-learn` – modeling and evaluation
  - `scorecardpy` – WoE binning and IV calculation
  - `lightgbm` – efficient gradient boosting

---

## 📌 Reproducibility Note
This notebook was originally developed as part of academic coursework using a seed tied to the student ID.
For privacy and reproducibility, the GitHub version now uses:
SEED = 42
Results may vary slightly when re-running, but the process and insights remain consistent.

📁 The dataset is included only for illustrative academic purposes. Do not redistribute commercially.

---

## 🤝 Connect
For questions, collaborations, or opportunities related to AI in finance and risk modeling, feel free to connect through dhakshashganesan@gmail.com or GitHub.

