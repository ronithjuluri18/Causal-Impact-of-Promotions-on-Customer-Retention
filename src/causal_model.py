"""
causal_model.py
---------------
Causal inference models for estimating the Average Treatment Effect (ATE)
of promotional campaigns on customer retention.

Methods:
  1. Propensity Score Matching (PSM)     — ATT estimation (scikit-learn)
  2. Double ML via EconML LinearDML      — ATE with 95% CI
  3. Uplift Modeling via CausalML        — individual-level treatment effects
"""

import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Any


# ── 1. Propensity Score Matching ──────────────────────────────────────────────

def propensity_score_matching(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "received_promotion",
    outcome_col:   str = "retained",
) -> Tuple[float, pd.DataFrame]:
    """
    Estimate ATT using 1:1 nearest-neighbor propensity score matching.
    Controls for confounding by matching treated customers to similar controls.
    """
    X = df[feature_cols].values
    T = df[treatment_col].values

    # Fit propensity model (P(T=1|X))
    ps_model = LogisticRegression(max_iter=1000, solver="lbfgs")
    ps_model.fit(X, T)

    df = df.copy()
    df["propensity_score"] = ps_model.predict_proba(X)[:, 1]

    # Trim to common support
    treated_ps = df[df[treatment_col] == 1]["propensity_score"]
    control_ps = df[df[treatment_col] == 0]["propensity_score"]
    lo = max(treated_ps.min(), control_ps.min())
    hi = min(treated_ps.max(), control_ps.max())

    df_trim = df[(df["propensity_score"] >= lo) & (df["propensity_score"] <= hi)]
    treated  = df_trim[df_trim[treatment_col] == 1].copy()
    control  = df_trim[df_trim[treatment_col] == 0].copy()

    # 1:1 nearest-neighbour match on propensity score
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(control[["propensity_score"]].values)
    _, indices = nn.kneighbors(treated[["propensity_score"]].values)
    matched_control = control.iloc[indices.flatten()]

    att = treated[outcome_col].mean() - matched_control[outcome_col].mean()

    print(f"\n📊 Propensity Score Matching")
    print(f"   Common support     : [{lo:.3f}, {hi:.3f}]")
    print(f"   Treated n          : {len(treated):,}")
    print(f"   Matched control n  : {len(matched_control):,}")
    print(f"   Treated retention  : {treated[outcome_col].mean():.4f}")
    print(f"   Control retention  : {matched_control[outcome_col].mean():.4f}")
    print(f"   ATT estimate       : {att:.4f}")

    return att, df


# ── 2. Double ML (EconML) ─────────────────────────────────────────────────────

def run_double_ml(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col:   str = "received_promotion",
    outcome_col:     str = "retained",
    experiment_name: str = "causal-retention-study",
) -> Tuple[float, Tuple[float, float], Any]:
    """
    Estimate ATE using Double ML (EconML LinearDML).
    Residualizes both outcome and treatment on covariates separately,
    then regresses outcome residuals on treatment residuals.
    All metrics logged to MLflow.
    """
    try:
        from econml.dml import LinearDML
    except ImportError:
        raise ImportError("Run: pip install econml")

    X = df[feature_cols].values
    T = df[treatment_col].values.astype(float)
    Y = df[outcome_col].values.astype(float)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="LinearDML"):
        mlflow.log_params({
            "model_type": "LinearDML",
            "n_samples":  len(df),
            "n_features": len(feature_cols),
            "treatment":  treatment_col,
            "outcome":    outcome_col,
            "cv_folds":   5,
        })

        dml = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
            model_t=GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
            discrete_treatment=True,
            cv=5,
            random_state=42,
        )
        dml.fit(Y, T, X=X)

        ate      = float(dml.ate(X))
        ci       = dml.ate_interval(X, alpha=0.05)
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])

        mlflow.log_metrics({
            "ATE":         ate,
            "CI_lower_95": ci_lower,
            "CI_upper_95": ci_upper,
        })

        print(f"\n📊 Double ML (EconML LinearDML)")
        print(f"   ATE estimate : {ate:.4f}")
        print(f"   95% CI       : [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"   MLflow run logged ✅")

    return ate, (ci_lower, ci_upper), dml


# ── 3. CausalML Uplift Modeling ───────────────────────────────────────────────

def run_causalml_uplift(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str = "received_promotion",
    outcome_col:   str = "retained",
) -> np.ndarray:
    """
    Estimate individual-level uplift using CausalML (Uber's library).
    Uplift = P(retained | treated) - P(retained | control) per customer.

    Used as cross-validation against Double ML estimates.
    """
    try:
        from causalml.inference.tree import UpliftRandomForestClassifier
    except ImportError:
        raise ImportError("Run: pip install causalml")

    X = df[feature_cols].values
    # CausalML expects string labels for treatment
    T = df[treatment_col].apply(
        lambda x: "treatment" if x == 1 else "control"
    ).values
    Y = df[outcome_col].values

    model = UpliftRandomForestClassifier(
        control_name="control",
        n_estimators=50,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, T, Y)

    # Uplift scores — positive = promotion helps retention
    uplift_scores = model.predict(X)
    # UpliftRandomForest returns array of shape (n, n_treatments)
    if uplift_scores.ndim > 1:
        uplift_scores = uplift_scores[:, 0]

    mean_uplift = float(np.mean(uplift_scores))
    pct_positive = float(np.mean(uplift_scores > 0) * 100)

    print(f"\n📊 CausalML Uplift Model")
    print(f"   Mean uplift score    : {mean_uplift:.4f}")
    print(f"   % customers +uplift  : {pct_positive:.1f}%")
    print(f"   (Positive = promotion increases retention probability)")

    return uplift_scores
