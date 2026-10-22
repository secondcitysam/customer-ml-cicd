import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
rows = 1000

age = rng.integers(18, 75, rows)
tenure = rng.integers(1, 73, rows)
monthly_charges = rng.uniform(20, 120, rows).round(2)
support_calls = rng.integers(0, 10, rows)
contract_type = rng.choice(
    ["Monthly", "Annual", "Two-Year"],
    rows,
    p=[0.55, 0.30, 0.15]
)

score = (
    (contract_type == "Monthly").astype(int) * 1.2
    + (support_calls >= 5).astype(int) * 1.0
    + (tenure < 12).astype(int) * 0.8
    + (monthly_charges > 90).astype(int) * 0.4
    + rng.normal(0, 0.8, rows)
)

churn = (score > 1.8).astype(int)

df = pd.DataFrame({
    "customer_id": range(10001, 10001 + rows),
    "age": age,
    "tenure_months": tenure,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "churn": churn
})

Path("data").mkdir(exist_ok=True)
df.to_csv("data/customers.csv", index=False)

print("Customer dataset generated")
print("Rows:", len(df))
print(df.head())
print("Churn distribution:")
print(df["churn"].value_counts())
