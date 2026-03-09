"""
evaluation.py
-------------
Statistical validation: bootstrap confidence intervals, t-test, results summary.
"""

import numpy as np
import pandas as pd
import mlflow
from scipy import stats
from typing import Tuple


def bootstrap_ate(
    df: pd.DataFrame,
    treatment_col: str = "received_promotion",
    outcome_col:   str = "retained",
    n_bootstrap:   int = 1000,
    seed:          int = 42,
) -> Tuple[float, float, float]:
    """Estimate ATE and 95% CI via bootstrap resampling (1,000 iterations)."""
    np.random.seed(seed)
    ates = []

    for _ in range(n_bootstrap):
        sample = df.sample(frac=1, replace=True)
        ate_i  = (
            sample[sample[treatment_col] == 1][outcome_col].mean()
            - sample[sample[treatment_col] == 0][outcome_col].mean()
        )
        ates.append(ate_i)

    ates     = np.array(ates)
    mean_ate = float(np.mean(ates))
    ci_lower = float(np.percentile(ates, 2.5))
    ci_upper = float(np.percentile(ates, 97.5))

    print(f"\n📊 Bootstrap Validation ({n_bootstrap:,} iterations)")
    print(f"   Mean ATE : {mean_ate:.4f}")
    print(f"   95% CI   : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   Std Dev  : {np.std(ates):.4f}")

    with mlflow.start_run(run_name="Bootstrap", nested=True):
        mlflow.log_params({"n_bootstrap": n_bootstrap})
        mlflow.log_metrics({
            "bootstrap_mean_ate": mean_ate,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "bootstrap_std":      float(np.std(ates)),
        })

    return mean_ate, ci_lower, ci_upper


def t_test_groups(
    df: pd.DataFrame,
    treatment_col: str = "received_promotion",
    outcome_col:   str = "retained",
) -> Tuple[float, float]:
    """
    Two-sample Welch t-test comparing retention between treatment groups.
    Naive baseline — does not control for confounders.
    """
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]
    t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)

    print(f"\n📊 T-Test (Naive — no confounder control)")
    print(f"   Treated mean : {treated.mean():.4f}")
    print(f"   Control mean : {control.mean():.4f}")
    print(f"   Naive diff   : {treated.mean() - control.mean():.4f}")
    print(f"   T-statistic  : {t_stat:.4f}")
    print(f"   P-value      : {p_value:.6f}")
    print(f"   Significant  : {'Yes ✅' if p_value < 0.05 else 'No ❌'}")

    return float(t_stat), float(p_value)


def summarize_results(
    naive_diff:    float,
    att_psm:       float,
    ate_dml:       float,
    ate_bootstrap: float,
) -> pd.DataFrame:
    """Print and return a comparison table of all estimation methods."""
    results = pd.DataFrame({
        "Method": [
            "Naive Difference (biased)",
            "Propensity Score Matching (ATT)",
            "Double ML / LinearDML (ATE)",
            "Bootstrap ATE (naive)",
        ],
        "Estimate": [naive_diff, att_psm, ate_dml, ate_bootstrap],
        "Controls for Confounding": ["No", "Yes", "Yes", "No"],
    })
    results["Estimate"] = results["Estimate"].round(4)
    print("\n📊 Results Summary")
    print(results.to_string(index=False))
    return results
