
# Risk Assessment & Loan Default Prediction using Random Forest

This repository contains an end-to-end Machine Learning pipeline designed to identify high-risk individuals and predict credit defaults based on socio-economic, financial, and demographic data.

Due to severe class imbalance (where the minority class represents only about 12.3% of the data), this project leverages specialized preprocessing transformers, advanced categorical encoders, and oversampling/undersampling techniques embedded within rigorous cross-validation setups.

## 📌 Project Overview

* **Objective:** Predict whether an individual presents a high loan/credit risk (`Risk_Flag`).
* **Dataset Structure:** 252,000 entries across 13 features, including columns such as `Income`, `Age`, `Experience`, `Profession`, `CITY`, `STATE`, and `House_Ownership`.
* **Core Algorithms:** Multi-stage `RandomForestClassifier` combined with resampling pipelines (SMOTE, RandomUnderSampler) and `BalancedRandomForestClassifier`.

---

## 🛠️ Tech Stack & Requirements

The project uses the following libraries for data processing, evaluation, and modeling:

* **Data Manipulation & Visualization:** `pandas`, `seaborn`, `matplotlib`
* **Machine Learning & Pipelines:** `scikit-learn`, `imbalanced-learn` (`imblearn`)
* **Categorical Encoding:** `category_encoders` (specifically `TargetEncoder` for high-cardinality values like CITY and STATE)
* **Advanced Metrics:** `hmeasure` (for evaluating the `h_score`)

To install the required packages, run:

```bash
pip install pandas scikit-learn imbalanced-learn category_encoders hmeasure matplotlib seaborn

```

---

## 📊 Dataset Insights & Analysis

Initial exploratory data analysis reveals the following payload features:

* **Numerical Features:** `Income`, `Age`, `Experience`, `CURRENT_JOB_YRS`, `CURRENT_HOUSE_YRS`.
* **Categorical Features:** `Married/Single`, `House_Ownership`, `Car_Ownership`, `Profession`, `CITY`, `STATE`.
* **Data Completeness:** The dataset contains `0` missing or null values and no duplicates.
* **Target Distribution:** Significant class imbalance exists, with non-default (`0`) vastly outnumbering defaults (`1`), making standard model evaluation metrics like accuracy misleading.

---

## ⚙️ Machine Learning Pipeline Architecture

To handle the feature engineering and data resampling seamlessly without data leakage, the project wraps the logic into an `imPipeline`:

1. **Feature Engineering & Preprocessing (`ColumnTransformer`):**
* **Binary Columns:** Encoded via binary or ordinal schemes.
* **Categorical Features:** High-cardinality variables like `CITY`, `STATE`, and `Profession` are dynamically maps using a supervised `TargetEncoder` to optimize computational complexity.
* **Numerical Remainder:** Kept intact or scaled appropriately.


2. **Imbalance Handling:**
* Embedded synthetic oversampling via **SMOTE** and strategic downsampling using **RandomUnderSampler** directly inside cross-validation splits to maintain test integrity.
* Comparative application of an out-of-the-box **BalancedRandomForestClassifier**.


3. **Hyperparameter Optimization:**
* Fine-tuned parameters via `GridSearchCV` combined with `StratifiedKFold` to maximize the Area Under the ROC Curve (`roc_auc_score`) and the `h_score`.



---

## 🚀 Usage Guide

### 1. Training the Model

Ensure your training source file is named `Training Data.csv` and placed in the project root folder. Run the notebook cells sequentially to clean the data, generate feature importance mappings, and start model fitting; otherwise you need to modify the code.

### 2. Feature Importance Evaluation

After pipeline completion, you can extract feature contributions from the Random Forest estimator. The model relies heavily on geographic and financial context to determine risk:

| Feature Rank | Pipeline Component Feature | Relative Importance |
| --- | --- | --- |
| 1 | `cats__CITY` | ~18.49% |
| 2 | `remainder__Income` | ~16.46% |
| 3 | `remainder__Age` | ~13.61% |
| 4 | `cats__Profession` | ~13.27% |
| 5 | `remainder__Experience` | ~9.45% |

---

## 📈 Evaluation Performance Metrics

The model is evaluated using robust metrics resilient to skewed datasets:

* **Classification Report:** Precision, Recall, and F1-Scores generated for the default category class.
* **ROC-AUC & Curve:** Quantifies the probability that the model ranks a random positive default higher than a random negative non-default.
* **H-Measure (`h_score`):** Used to prevent the target class distribution assumptions built into traditional ROC metrics.

---

## 📂 Repository File Structure

```text
├──AnalyticPics #Plots
├── Data     # Datasets
    ├──Test Data.csv
    └──Training Data.csv
├── RandomForest.ipynb     # Main notebook containing data processing and models
└── README.md              # Documentation (This file)

```
