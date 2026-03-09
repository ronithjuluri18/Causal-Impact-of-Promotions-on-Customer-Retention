"""
tests/test_pipeline.py
----------------------
Unit tests for the causal retention pipeline.
Run: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader         import generate_customer_data
from feature_engineering import engineer_features, validate_features


class TestDataLoader:

    def test_record_count(self):
        assert len(generate_customer_data(n=500)) == 500

    def test_required_columns(self):
        df = generate_customer_data(n=200)
        for col in ["customer_id", "age", "region", "segment",
                    "tenure_months", "avg_monthly_spend",
                    "purchase_frequency", "last_purchase_days_ago",
                    "num_support_contacts", "received_promotion", "retained"]:
            assert col in df.columns

    def test_no_nulls(self):
        assert generate_customer_data(n=500).isnull().sum().sum() == 0

    def test_treatment_binary(self):
        df = generate_customer_data(n=500)
        assert set(df["received_promotion"].unique()).issubset({0, 1})

    def test_outcome_binary(self):
        df = generate_customer_data(n=500)
        assert set(df["retained"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        df1 = generate_customer_data(n=200, seed=42)
        df2 = generate_customer_data(n=200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_age_range(self):
        df = generate_customer_data(n=500)
        assert df["age"].between(18, 70).all()

    def test_promotion_rate_reasonable(self):
        df = generate_customer_data(n=5000)
        assert 0.2 < df["received_promotion"].mean() < 0.9


class TestFeatureEngineering:

    def setup_method(self):
        self.df = generate_customer_data(n=500)

    def test_new_columns_created(self):
        df, _ = engineer_features(self.df)
        for col in ["is_high_spender", "is_long_tenure", "is_lapsed",
                    "spend_x_frequency", "log_spend", "region_enc", "segment_enc"]:
            assert col in df.columns

    def test_feature_list_length(self):
        _, features = engineer_features(self.df)
        assert len(features) >= 10

    def test_binary_flags(self):
        df, _ = engineer_features(self.df)
        for col in ["is_high_spender", "is_long_tenure",
                    "is_lapsed", "is_frequent_buyer", "is_premium"]:
            assert set(df[col].unique()).issubset({0, 1})

    def test_no_nulls_after_engineering(self):
        df, features = engineer_features(self.df)
        assert df[features].isnull().sum().sum() == 0

    def test_log_transform_non_negative(self):
        df, _ = engineer_features(self.df)
        assert (df["log_spend"] >= 0).all()
        assert (df["log_tenure"] >= 0).all()

    def test_validate_passes(self):
        df, features = engineer_features(self.df)
        validate_features(df, features)

    def test_validate_catches_nulls(self):
        df, features = engineer_features(self.df)
        df.loc[0, features[0]] = np.nan
        with pytest.raises(ValueError, match="Null values"):
            validate_features(df, features)
