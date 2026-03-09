"""
feature_engineering.py
-----------------------
Feature construction and preprocessing for causal retention analysis.
Includes EDA helpers for supervised/unsupervised learning during feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, List


# ── 1. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build 18 features from raw customer data.

    Returns
    -------
    df           : DataFrame with engineered features added
    feature_cols : list of feature column names for modeling
    """
    df = df.copy()

    # Encode categoricals
    df["region_enc"]  = LabelEncoder().fit_transform(df["region"])
    df["segment_enc"] = LabelEncoder().fit_transform(df["segment"])

    # Binary behavioral flags
    spend_median = df["avg_monthly_spend"].median()
    df["is_high_spender"]   = (df["avg_monthly_spend"] > spend_median).astype(int)
    df["is_long_tenure"]    = (df["tenure_months"] > 24).astype(int)
    df["is_lapsed"]         = (df["last_purchase_days_ago"] > 180).astype(int)
    df["is_frequent_buyer"] = (df["purchase_frequency"] > 3).astype(int)
    df["is_premium"]        = (df["segment"] == "Premium").astype(int)

    # Interaction & transform features
    df["spend_x_frequency"] = df["avg_monthly_spend"] * df["purchase_frequency"]
    df["tenure_x_spend"]    = df["tenure_months"] * df["avg_monthly_spend"]
    df["recency_score"]     = 365 - df["last_purchase_days_ago"]
    df["log_spend"]         = np.log1p(df["avg_monthly_spend"])
    df["log_tenure"]        = np.log1p(df["tenure_months"])

    feature_cols = [
        "age", "tenure_months", "avg_monthly_spend",
        "purchase_frequency", "last_purchase_days_ago", "num_support_contacts",
        "region_enc", "segment_enc",
        "is_high_spender", "is_long_tenure", "is_lapsed",
        "is_frequent_buyer", "is_premium",
        "spend_x_frequency", "tenure_x_spend",
        "recency_score", "log_spend", "log_tenure",
    ]

    return df, feature_cols


def validate_features(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Raise ValueError if any feature contains nulls or infinite values."""
    nulls = df[feature_cols].isnull().sum()
    if nulls.any():
        raise ValueError(f"Null values found:\n{nulls[nulls > 0]}")

    infs = np.isinf(df[feature_cols].select_dtypes(include=np.number)).sum()
    if infs.any():
        raise ValueError(f"Infinite values found:\n{infs[infs > 0]}")

    print(f"✅ Validation passed — {len(feature_cols)} features, "
          f"{len(df):,} records, 0 nulls, 0 infs")


# ── 2. EDA — Unsupervised: Customer Segmentation via KMeans ──────────────────

def cluster_customers(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 4,
    save_plot: bool = True,
) -> pd.DataFrame:
    """
    Unsupervised learning — cluster customers into segments using KMeans.
    Used during EDA to understand natural groupings before causal modeling.
    """
    X = df[feature_cols].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["customer_cluster"] = kmeans.fit_predict(X)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    if save_plot:
        os.makedirs("notebooks", exist_ok=True)
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(coords[:, 0], coords[:, 1],
                              c=df["customer_cluster"], cmap="tab10",
                              alpha=0.4, s=5)
        plt.colorbar(scatter, label="Cluster")
        plt.title("Customer Segmentation (KMeans + PCA)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.tight_layout()
        plt.savefig("notebooks/customer_clusters.png", dpi=150)
        plt.close()
        print("   Plot saved → notebooks/customer_clusters.png")

    retention_by_cluster = df.groupby("customer_cluster")["retained"].mean()
    print(f"\n📊 KMeans Clustering ({n_clusters} clusters)")
    print(retention_by_cluster.to_string())

    return df


# ── 3. EDA — Supervised: Feature Importance via Random Forest ────────────────

def feature_importance_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_col: str = "retained",
    save_plot: bool = True,
) -> pd.Series:
    """
    Supervised learning — rank features by importance using Random Forest.
    Used during EDA to identify which features are most predictive of retention.
    """
    X = df[feature_cols].values
    y = df[outcome_col].values

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)

    if save_plot:
        os.makedirs("notebooks", exist_ok=True)
        plt.figure(figsize=(8, 6))
        importance.plot(kind="bar", color="steelblue", edgecolor="white")
        plt.title("Feature Importance (Random Forest)")
        plt.ylabel("Importance Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("notebooks/feature_importance.png", dpi=150)
        plt.close()
        print("   Plot saved → notebooks/feature_importance.png")

    print(f"\n📊 Top 5 Features by Importance:")
    print(importance.head(5).to_string())

    return importance


if __name__ == "__main__":
    from data_loader import generate_customer_data
    os.makedirs("data/processed", exist_ok=True)

    df = generate_customer_data()
    df, feature_cols = engineer_features(df)
    validate_features(df, feature_cols)

    df = cluster_customers(df, feature_cols)
    feature_importance_analysis(df, feature_cols)

    df.to_csv("data/processed/features.csv", index=False)
    print(f"\n✅ Saved → data/processed/features.csv")
