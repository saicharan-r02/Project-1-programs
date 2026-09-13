import pandas as pd
import numpy as np

def compute_monthly_cohorts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates monthly customer retention cohorts.
    Returns:
      - cohort_counts: Absolute active customer count per cohort index
      - retention_matrix: Percentage retention rate per cohort index
    """
    valid_df = df[df["is_returned"] == 0].copy()
    
    # Extract year-month of transaction
    valid_df["order_month"] = valid_df["order_date"].dt.to_period("M")

    # Customer acquisition cohort month
    cohort_map = valid_df.groupby("customer_id")["order_month"].min().reset_index()
    cohort_map.rename(columns={"order_month": "cohort_month"}, inplace=True)

    valid_df = pd.merge(valid_df, cohort_map, on="customer_id")

    # Calculate cohort index (months since acquisition)
    year_diff = valid_df["order_month"].dt.year - valid_df["cohort_month"].dt.year
    month_diff = valid_df["order_month"].dt.month - valid_df["cohort_month"].dt.month
    valid_df["cohort_index"] = year_diff * 12 + month_diff

    # Group by cohort_month and cohort_index
    cohort_data = valid_df.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()

    # Pivot table
    cohort_counts = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_id")
    cohort_sizes = cohort_counts.iloc[:, 0]
    
    # Retention rate (%)
    retention_matrix = cohort_counts.divide(cohort_sizes, axis=0) * 100
    retention_matrix = retention_matrix.round(1)

    return cohort_counts, retention_matrix
