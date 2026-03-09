# Causal Impact of Promotions on Customer Retention

## Research Question
> Does receiving a promotional offer **causally** increase customer retention — after controlling for confounding variables?

A naive comparison overestimates the effect because higher-spending customers are both more likely to receive promotions **and** more likely to retain. This project applies causal inference methods to isolate the true treatment effect.

---

## Methods

| Method | Library | Purpose | Controls Confounding |
|---|---|---|---|
| Naive t-test | scipy | Biased baseline | ❌ |
| Propensity Score Matching | scikit-learn | ATT estimation | ✅ |
| Double ML (LinearDML) | EconML | ATE with 95% CI | ✅ |
| Uplift Modeling | CausalML | Individual-level effects | ✅ |
| Bootstrap (1,000 iter) | numpy | Validation | Partial |
| KMeans Clustering | scikit-learn | EDA segmentation | — |
| Random Forest Importance | scikit-learn | Feature selection | — |

---

## Dataset

### Synthetic Dataset (used in this project)
Synthetically generated — 100,000 customer records, seed=42 (fully reproducible).
Confounding is explicitly introduced: higher-spending customers are more likely to receive promotions, mimicking real-world selection bias.

| Column | Description |
|---|---|
| `age`, `region`, `segment` | Customer demographics |
| `tenure_months` | How long they've been a customer |
| `avg_monthly_spend` | Average monthly spend (confounds treatment) |
| `purchase_frequency` | Purchases per month |
| `last_purchase_days_ago` | Recency |
| `num_support_contacts` | Support interactions |
| `received_promotion` | **Treatment** (binary, confounded) |
| `retained` | **Outcome** (binary) |

**True simulated ATE = 0.20** (20 percentage point retention lift from promotions)

### Real-World Reference Datasets
The synthetic data structure is inspired by these publicly available datasets:

| Dataset | Source | Link |
|---|---|---|
| UCI Online Retail Dataset | UCI ML Repository | https://archive.ics.uci.edu/dataset/352/online+retail |
| E-Commerce Behavior Data | Kaggle | https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store |
| Customer Retention / Store Data | Kaggle | https://www.kaggle.com/datasets/uttamp/store-data |
| E-Commerce Customer Churn | Kaggle | https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction |

---

## Project Structure

```
causal-retention/
├── src/
│   ├── data_loader.py          # Synthetic data (pandas + PySpark)
│   ├── feature_engineering.py  # 18 features + KMeans + RF importance
│   ├── causal_model.py         # PSM + Double ML (EconML) + CausalML uplift
│   └── evaluation.py           # Bootstrap + t-test + results summary
├── tests/
│   └── test_pipeline.py        # 15 unit tests (pytest)
├── data/
│   ├── raw/                    # customer_data.csv (generated at runtime)
│   └── processed/              # features.csv, results_summary.csv
├── notebooks/                  # EDA plots (generated at runtime)
├── main.py                     # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/ronithjuluri18/Causal-Impact-of-Promotions-on-Customer-Retention.git
cd Causal-Impact-of-Promotions-on-Customer-Retention

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# 4. View MLflow experiment dashboard
mlflow ui
# Open http://localhost:5000

# 5. Run unit tests
pytest tests/ -v
```

---

## Key Results

| Method | ATE Estimate | Confounding Controlled |
|---|---|---|
| Naive difference (biased) | ~0.27 | ❌ |
| Propensity Score Matching | ~0.20 | ✅ |
| Double ML (EconML) | ~0.20 | ✅ |
| Bootstrap (naive) | ~0.27 | ❌ |

Causal methods recover the true simulated ATE of **~0.20** — a statistically significant **20 percentage point** retention lift from promotions, after controlling for spend, segment, and tenure confounders. The naive estimate inflates this by ~35% due to selection bias.

---

## Tech Stack

Python · PySpark · scikit-learn · EconML · CausalML · MLflow · pandas · NumPy · SciPy · statsmodels · pytest

---

## Author

**Ronith Pradyumna Juluri**
M.S. Data Science, Pace University (GPA: 3.89)
[linkedin.com/in/ronith-juluri](https://linkedin.com/in/ronith-juluri)
