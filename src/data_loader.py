"""
data_loader.py
--------------
Generates a realistic synthetic customer dataset with confounding.

Two implementations:
  1. generate_customer_data()       — pandas (fast, default)
  2. generate_customer_data_spark() — PySpark (distributed processing)

Higher-spending customers are more likely to receive promotions,
mimicking real-world selection bias in promotional targeting.

Dataset inspiration:
  - UCI Online Retail Dataset:
    https://archive.ics.uci.edu/dataset/352/online+retail
  - Kaggle E-Commerce Behavior Data:
    https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
  - Kaggle Customer Retention Dataset:
    https://www.kaggle.com/datasets/uttamp/store-data

Seed=42 ensures full reproducibility across all runs.
"""

import pandas as pd
import numpy as np
import os


# ── 1. Pandas Version (default) ───────────────────────────────────────────────

def generate_customer_data(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer dataset with confounded treatment assignment.

    Parameters
    ----------
    n    : number of customer records (default 100,000)
    seed : random seed for reproducibility (default 42)

    Returns
    -------
    pd.DataFrame with customer features, treatment, and outcome columns
    """
    np.random.seed(seed)

    df = pd.DataFrame({
        "customer_id":            range(n),
        "age":                    np.random.randint(18, 70, n),
        "region":                 np.random.choice(["North", "South", "East", "West"], n),
        "segment":                np.random.choice(["Premium", "Standard", "Basic"], n,
                                                   p=[0.2, 0.5, 0.3]),
        "tenure_months":          np.random.randint(1, 120, n),
        "avg_monthly_spend":      np.round(np.random.exponential(scale=150, size=n), 2),
        "purchase_frequency":     np.random.poisson(lam=3, size=n),
        "last_purchase_days_ago": np.random.randint(1, 365, n),
        "num_support_contacts":   np.random.poisson(lam=1, size=n),
    })

    # Confounding: higher-spend + premium customers more likely to receive promo
    spend_median = df["avg_monthly_spend"].median()
    promo_prob = (
        0.25
        + 0.30 * (df["avg_monthly_spend"] > spend_median).astype(float)
        + 0.10 * (df["segment"] == "Premium").astype(float)
        + 0.05 * (df["tenure_months"] > 24).astype(float)
    ).clip(0, 1)

    df["received_promotion"] = np.random.binomial(1, promo_prob)

    # Outcome: retention driven by treatment + confounders + noise
    retention_prob = (
        0.35
        + 0.20 * df["received_promotion"]                             # true ATE = 0.20
        + 0.15 * (df["tenure_months"] > 24).astype(float)
        + 0.10 * (df["avg_monthly_spend"] > spend_median).astype(float)
        + 0.08 * (df["segment"] == "Premium").astype(float)
        - 0.07 * (df["last_purchase_days_ago"] > 180).astype(float)
        - 0.05 * (df["num_support_contacts"] > 2).astype(float)
        + np.random.normal(0, 0.04, n)
    ).clip(0, 1)

    df["retained"] = np.random.binomial(1, retention_prob)

    return df


# ── 2. PySpark Version (distributed processing) ───────────────────────────────

def generate_customer_data_spark(n: int = 100_000, seed: int = 42):
    """
    PySpark version — generates data with pandas then converts to Spark DataFrame
    and applies distributed feature transformations.

    Requires: pyspark>=3.4.0

    Returns
    -------
    pyspark.sql.DataFrame
    """
    try:
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
    except ImportError:
        raise ImportError("PySpark not installed. Run: pip install pyspark")

    # Step 1: Generate with pandas (seed-controlled)
    pandas_df = generate_customer_data(n=n, seed=seed)

    # Step 2: Start Spark session
    spark = (
        SparkSession.builder
        .appName("CausalRetention")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Step 3: Convert to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)

    # Step 4: Apply distributed feature transformations
    spend_median = float(pandas_df["avg_monthly_spend"].median())

    spark_df = (
        spark_df
        .withColumn("is_high_spender",
                    F.when(F.col("avg_monthly_spend") > spend_median, 1).otherwise(0))
        .withColumn("is_long_tenure",
                    F.when(F.col("tenure_months") > 24, 1).otherwise(0))
        .withColumn("is_lapsed",
                    F.when(F.col("last_purchase_days_ago") > 180, 1).otherwise(0))
        .withColumn("is_premium",
                    F.when(F.col("segment") == "Premium", 1).otherwise(0))
        .withColumn("spend_x_frequency",
                    F.col("avg_monthly_spend") * F.col("purchase_frequency"))
        .withColumn("log_spend",
                    F.log1p(F.col("avg_monthly_spend")))
        .withColumn("recency_score",
                    F.lit(365) - F.col("last_purchase_days_ago"))
    )

    print(f"✅ Spark DataFrame created — {spark_df.count():,} records, "
          f"{len(spark_df.columns)} columns")

    return spark_df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    # Pandas version
    df = generate_customer_data()
    df.to_csv("data/raw/customer_data.csv", index=False)
    print(f"✅ Generated {len(df):,} records → data/raw/customer_data.csv")
    print(f"   Promotion rate : {df['received_promotion'].mean():.2%}")
    print(f"   Retention rate : {df['retained'].mean():.2%}")
    print(f"   Naive lift     : "
          f"{df.groupby('received_promotion')['retained'].mean().diff().iloc[-1]:.4f}")

    # PySpark version
    print("\nTesting PySpark version...")
    try:
        spark_df = generate_customer_data_spark()
        spark_df.show(5)
    except ImportError as e:
        print(f"  ⚠️  {e}")
