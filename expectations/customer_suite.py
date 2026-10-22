
import pandas as pd
import great_expectations as gx

DATA_PATH = "data/customers.csv"

EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "churn"
]


def validate_customer_data():
    df = pd.read_csv(DATA_PATH)

    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(
        name="customer_data_source"
    )

    data_asset = data_source.add_dataframe_asset(
        name="customer_data"
    )

    batch_definition = (
        data_asset.add_batch_definition_whole_dataframe(
            "customer_batch"
        )
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    expectations = [
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=EXPECTED_COLUMNS
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_id"
        ),
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="customer_id"
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="age"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age",
            min_value=18,
            max_value=100
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="tenure_months",
            min_value=1,
            max_value=120
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="monthly_charges",
            min_value=0,
            strict_min=True
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="support_calls",
            min_value=0,
            max_value=50
        ),
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="contract_type",
            value_set=["Monthly", "Annual", "Two-Year"]
        ),
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            column="churn",
            value_set=[0, 1]
        )
    ]

    failed = []

    for expectation in expectations:
        result = batch.validate(expectation)
        print(
            expectation.__class__.__name__,
            result.success
        )

        if not result.success:
            failed.append(expectation.__class__.__name__)

    if failed:
        raise ValueError(
            f"Great Expectations validation failed: {failed}"
        )

    print("Great Expectations validation passed")


if __name__ == "__main__":
    validate_customer_data()
