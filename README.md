# 📉 Customer Churn Prediction

> A portfolio-grade machine learning pipeline to identify customers at risk of leaving a telecom service — featuring end-to-end EDA, multi-model classification, evaluation dashboards, and business-ready risk tiering.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── churn_pipeline.py          # Full ML pipeline (data → EDA → models → evaluation)
├── README.md                  # Project documentation (this file)
│
├── outputs/
│   ├── fig1_eda_dashboard.png     # Exploratory Data Analysis — 6-panel signal dashboard
│   ├── fig2_model_evaluation.png  # Model metrics, ROC curves, confusion matrix, CV
│   ├── fig3_risk_tiers.png        # Business risk tier view (Low / Medium / High)
│   └── summary.json               # Final metrics for all models (machine-readable)
```

---

## 🎯 Problem Statement

Customer churn — the rate at which customers stop using a service — is one of the most costly problems in subscription-based businesses. Acquiring a new customer is 5–7× more expensive than retaining an existing one.

**Goal:** Build a binary classifier to predict which telecom customers are likely to churn, enabling proactive retention interventions.

---

## 📊 Dataset

| Property        | Detail                                  |
|-----------------|-----------------------------------------|
| Source          | Synthetic telecom cohort (realistic distributions) |
| Rows            | 7,000 customers                         |
| Features        | 10 input features + 1 target            |
| Churn Rate      | ~28.6% (class imbalance handled)        |
| Train / Test    | 80% / 20% stratified split              |

### Features

| Feature            | Type        | Description                                      |
|--------------------|-------------|--------------------------------------------------|
| `tenure`           | Numeric     | Months the customer has been with the service    |
| `monthly_charges`  | Numeric     | Monthly bill amount (USD)                        |
| `total_charges`    | Numeric     | Cumulative spend (USD)                           |
| `num_products`     | Numeric     | Number of services subscribed                    |
| `support_calls`    | Numeric     | Number of customer support contacts              |
| `contract`         | Categorical | Month-to-Month / One Year / Two Year             |
| `internet_service` | Categorical | Fiber / DSL / No                                 |
| `payment_method`   | Categorical | Electronic / Mailed / Bank Transfer / Credit Card|
| `senior_citizen`   | Binary      | 1 if customer is senior                          |
| `dependents`       | Binary      | 1 if customer has dependents                     |
| `churn`            | **Target**  | 1 = churned, 0 = retained                        |

---

## 🔍 Exploratory Data Analysis

**Figure 1 — EDA Signal Dashboard** covers six analytical panels:

| Panel | Insight |
|-------|---------|
| Churn Donut | 28.6% overall churn rate |
| Tenure Distribution | Churners heavily skew toward short tenures (< 12 months) |
| Monthly Charges KDE | Churners cluster in the $70–$100/month band |
| Contract Type | Month-to-Month churn is ~3× higher than Two Year contracts |
| Support Calls | Near-linear rise in churn probability per additional call |
| Correlation Heatmap | `tenure` is the strongest negative predictor; `support_calls` the strongest positive |

---

## 🤖 Models

Three classification models were trained and compared:

### 1. Logistic Regression
- Regularisation: `C=0.5` (L2)
- `class_weight="balanced"` to handle imbalance
- Features standardised with `StandardScaler`

### 2. Random Forest
- 300 estimators, `max_depth=10`, `min_samples_leaf=5`
- `class_weight="balanced"`
- No feature scaling required

### 3. XGBoost
- 300 rounds, `learning_rate=0.08`, `max_depth=5`
- `scale_pos_weight` set to ratio of negative/positive samples
- Subsample + column subsampling for regularisation

---

## 📈 Results

| Model                | Accuracy | Recall | F1    | ROC-AUC |
|----------------------|----------|--------|-------|---------|
| Logistic Regression  | 0.740    | **0.816** | 0.642 | **0.840** |
| Random Forest        | **0.774**| 0.741  | **0.653** | 0.829 |
| XGBoost              | 0.754    | 0.713  | 0.624 | 0.816 |

> ✅ **Best Model: Logistic Regression** — highest ROC-AUC (0.840) and recall (0.816)

### Why recall matters most here
In churn prediction, a **false negative** (missing a churner) is far more costly than a **false positive** (flagging a loyal customer). Logistic Regression's high recall of 81.6% means it catches the most at-risk customers for retention outreach.

### Cross-Validation
5-fold Stratified CV confirms model stability — no overfitting observed across folds.

---

## 🚦 Risk Tiering

Customers are scored and segmented into three actionable tiers:

| Tier         | P(churn) | Recommended Action                        |
|--------------|----------|-------------------------------------------|
| 🟢 Low Risk  | 0–30%    | No immediate action needed                |
| 🟡 Medium Risk | 30–60% | Proactive check-in, loyalty offer         |
| 🔴 High Risk | 60–100%  | Immediate retention call, discount/upgrade|

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn scipy
```

### Run the full pipeline

```bash
python churn_pipeline.py
```

This will:
1. Generate the synthetic telecom dataset
2. Run full EDA and save `fig1_eda_dashboard.png`
3. Train all three models
4. Save `fig2_model_evaluation.png` with all metrics
5. Save `fig3_risk_tiers.png` with business risk view
6. Write `summary.json` with final scores

### Requirements

| Package         | Version (tested) |
|-----------------|------------------|
| Python          | 3.10+            |
| pandas          | ≥ 2.0            |
| numpy           | ≥ 1.24           |
| scikit-learn    | ≥ 1.3            |
| xgboost         | ≥ 2.0            |
| matplotlib      | ≥ 3.7            |
| seaborn         | ≥ 0.13           |
| scipy           | ≥ 1.11           |

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Stratified train/test split | Preserves churn ratio in both sets |
| `class_weight="balanced"` | Prevents model bias toward majority (non-churn) class |
| Recall as primary metric | Missing a churner costs more than a false alarm |
| ROC-AUC as ranking metric | Threshold-independent; robust for imbalanced classes |
| 5-fold Stratified CV | Validates generalisation, not just test-set luck |
| Risk tier segmentation | Converts model output into actionable business intelligence |

---

## 📌 Top Churn Drivers (XGBoost Feature Importance)

1. **tenure** — Long-standing customers are far less likely to churn
2. **contract** — Month-to-Month customers are the most volatile segment
3. **support_calls** — Every additional support contact increases churn risk
4. **monthly_charges** — Higher bills correlate with higher churn propensity
5. **internet_service** — Fiber customers show elevated churn rates

---

## 💡 Business Recommendations

- **Lock in Month-to-Month customers** with contract upgrade incentives after month 3
- **Flag customers with 3+ support calls** for proactive outreach
- **New customers in their first 6 months** are the highest-risk cohort — prioritise onboarding quality
- **Fiber customers with high bills** represent the highest revenue at risk — target with loyalty discounts

---

## 👤 Author

**MockSphere** — Data Analytics & Engineering Portfolio  
Built as an internship evaluation project demonstrating end-to-end ML pipeline design.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
