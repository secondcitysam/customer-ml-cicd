
import pandas as pd
from src.data_validation import validate_data
from src.model import create_model


def test_customer_pipeline_integration():
    validate_data()

    df = pd.read_csv("data/customers.csv")

    features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_calls",
        "contract_type"
    ]

    X = df[features].copy()
    X["contract_type"] = X["contract_type"].map({
        "Monthly": 0,
        "Annual": 1,
        "Two-Year": 2
    })

    y = df["churn"]

    model = create_model()
    model.fit(X, y)

    predictions = model.predict(X.iloc[:10])

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})
