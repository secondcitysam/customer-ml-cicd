
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.model import create_model

DATA_PATH = "data/customers.csv"
MAX_PREDICTION_LATENCY = 0.5


def load_data():
    df = pd.read_csv(DATA_PATH)

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

    return X, y


def test_model_accuracy():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.70, (
        f"Model accuracy {accuracy:.3f} is below 0.70"
    )


def test_prediction_latency():
    X, y = load_data()

    model = create_model()
    model.fit(X, y)

    sample = X.iloc[[0]]

    start = time.perf_counter()
    prediction = model.predict(sample)
    elapsed = time.perf_counter() - start

    assert len(prediction) == 1
    assert elapsed < MAX_PREDICTION_LATENCY, (
        f"Prediction took {elapsed:.4f}s; "
        f"limit is {MAX_PREDICTION_LATENCY}s"
    )


def test_predictions_are_valid():
    X, y = load_data()

    model = create_model()
    model.fit(X, y)

    predictions = model.predict(X.iloc[:20])

    assert len(predictions) == 20
    assert np.isin(predictions, [0, 1]).all()
