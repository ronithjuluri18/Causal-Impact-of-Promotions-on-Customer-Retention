"""
main.py
-------
End-to-end causal inference pipeline.

Phases:
  1. Data generation (pandas + PySpark)
  2. Feature engineering (18 features)
  3. EDA — unsupervised clustering + supervised feature importance
  4. Naive baseline (t-test)
  5. Propensity Score Matching
  6. Double ML (EconML)
  7. CausalML Uplift Modeling
  8. Bootstrap validation
  9. Results summary + MLflow logging

Run:        python main.py
MLflow UI:  mlflow ui   → open http://localhost:5000
"""

import os
import sys
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader          import generate_customer_data, generate_customer_data_spark
from feature_engineering  import engineer_features, validate_features, \
                                  cluster_customers, feature_importance_analysis
from causal_model         import propensity_score_matching, run_double_ml, \
                                  run_causalml_uplift
from evaluation           import bootstrap_ate, t_test_groups, summarize_results

# ── Config ────────────────────────────────────────────────────────────────────
N_RECORDS       = 100_000
SEED            = 42
TREATMENT_COL   = "received_promotion"
OUTCOME_COL     = "retained"
EXPERIMENT_NAME = "causal-retention-study"
N_BOOTSTRAP     = 1000

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("notebooks",      exist_ok=True)


def main():
    print("=" * 60)
    print("  Causal Impact of Promotions on Customer Retention")
    print("=" * 60)

    # ── 1. Data Generation ────────────────────────────────────
    print("\n[1/8] Generating synthetic dataset (pandas)...")
    df = generate_customer_data(n=N_RECORDS, seed=SEED)
    df.to_csv("data/raw/customer_data.csv", index=False)
    print(f"  Records        : {len(df):,}")
    print(f"  Promotion rate : {df[TREATMENT_COL].mean():.2%}")
    print(f"  Retention rate : {df[OUTCOME_COL].mean():.2%}")

    print("\n  Attempting PySpark version...")
    try:
        spark_df = generate_customer_data_spark(n=N_RECORDS, seed=SEED)
        print(f"  PySpark DataFrame : {spark_df.count():,} rows ✅")
    except Exception as e:
        print(f"  ⚠️  PySpark skipped: {e}")

    # ── 2. Feature Engineering ────────────────────────────────
    print("\n[2/8] Engineering features...")
    df, feature_cols = engineer_features(df)
    validate_features(df, feature_cols)
    print(f"  Features built : {len(feature_cols)}")

    # ── 3. EDA ────────────────────────────────────────────────
    print("\n[3/8] Running EDA (unsupervised + supervised)...")
    df = cluster_customers(df, feature_cols, n_clusters=4)
    feature_importance_analysis(df, feature_cols, OUTCOME_COL)
    df.to_csv("data/processed/features.csv", index=False)

    # ── 4. Naive Baseline ─────────────────────────────────────
    print("\n[4/8] Computing naive baseline (biased)...")
    naive_diff = (
        df[df[TREATMENT_COL] == 1][OUTCOME_COL].mean()
        - df[df[TREATMENT_COL] == 0][OUTCOME_COL].mean()
    )
    t_stat, p_value = t_test_groups(df, TREATMENT_COL, OUTCOME_COL)

    # ── 5. Propensity Score Matching ──────────────────────────
    print("\n[5/8] Running Propensity Score Matching...")
    att, df = propensity_score_matching(df, feature_cols, TREATMENT_COL, OUTCOME_COL)

    # ── 6. Double ML ──────────────────────────────────────────
    print("\n[6/8] Running Double ML (EconML)...")
    mlflow.set_experiment(EXPERIMENT_NAME)
    try:
        ate_dml, ci_dml, _ = run_double_ml(
            df, feature_cols, TREATMENT_COL, OUTCOME_COL, EXPERIMENT_NAME
        )
    except ImportError as e:
        print(f"  ⚠️  {e}")
        ate_dml = att
        ci_dml  = (None, None)

    # ── 7. CausalML Uplift ────────────────────────────────────
    print("\n[7/8] Running CausalML Uplift Model...")
    try:
        uplift_scores = run_causalml_uplift(
            df, feature_cols, TREATMENT_COL, OUTCOME_COL
        )
    except ImportError as e:
        print(f"  ⚠️  {e}")
        uplift_scores = None

    # ── 8. Bootstrap Validation ───────────────────────────────
    print("\n[8/8] Running bootstrap validation (1,000 iterations)...")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="FullPipeline"):
        ate_bootstrap, ci_lo, ci_hi = bootstrap_ate(
            df, TREATMENT_COL, OUTCOME_COL, n_bootstrap=N_BOOTSTRAP, seed=SEED
        )
        mlflow.log_params({"n_records": N_RECORDS, "n_features": len(feature_cols)})
        mlflow.log_metrics({
            "naive_diff":    naive_diff,
            "att_psm":       att,
            "ate_dml":       ate_dml,
            "ate_bootstrap": ate_bootstrap,
            "t_stat":        t_stat,
            "p_value":       p_value,
        })

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    results = summarize_results(naive_diff, att, ate_dml, ate_bootstrap)
    results.to_csv("data/processed/results_summary.csv", index=False)
    
    # ── SQL: Save results to SQLite database ──────────────────────
    import sqlite3
    print("\n[+] Saving data to SQLite database...")
    conn = sqlite3.connect("data/processed/customers.db")
    df.to_sql("customers", conn, if_exists="replace", index=False)
    results.to_sql("results_summary", conn, if_exists="replace", index=False)
    conn.close()
    print("   Database saved → data/processed/customers.db")
    print("   Tables: customers, results_summary")

    print("\n✅ Pipeline complete.")
    print("   Results  → data/processed/results_summary.csv")
    print("   Plots    → notebooks/")
    print("   MLflow   → run: mlflow ui")


if __name__ == "__main__":
    main()
