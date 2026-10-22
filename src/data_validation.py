
import pandas as pd
from pathlib import Path
from expectations.customer_suite import validate_customer_data

DATA_PATH = Path("data/customers.csv")

EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "churn"
]


def validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    validate_customer_data()

    df = pd.read_csv(DATA_PATH)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Customer dataset schema does not match")

    if df.empty:
        raise ValueError("Customer dataset is empty")

    if df["customer_id"].isna().any():
        raise ValueError("customer_id contains missing values")

    if df["customer_id"].duplicated().any():
        raise ValueError("customer_id contains duplicate values")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")

    if not df["age"].between(18, 100).all():
        raise ValueError("age is outside the allowed range")

    if not df["tenure_months"].between(1, 120).all():
        raise ValueError("tenure_months is outside the allowed range")

    if not (df["monthly_charges"] > 0).all():
        raise ValueError("monthly_charges must be positive")

    if not df["support_calls"].between(0, 50).all():
        raise ValueError("support_calls is outside the allowed range")

    if not df["contract_type"].isin(
        ["Monthly", "Annual", "Two-Year"]
    ).all():
        raise ValueError("contract_type contains invalid values")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("churn must contain only 0 or 1")

    print("Customer data validation passed")
    print("Rows validated:", len(df))


if __name__ == "__main__":
    validate_data()
