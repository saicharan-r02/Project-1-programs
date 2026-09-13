import numpy as np
import pandas as pd

def compute_rfm_segments(df: pd.DataFrame, reference_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Customer RFM (Recency, Frequency, Monetary) segmentation and scoring.
    """
    valid_df = df[df["is_returned"] == 0]
    if reference_date is None:
        reference_date = valid_df["order_date"].max() + pd.Timedelta(days=1)

    # 1. Aggregate customer metrics
    rfm = valid_df.groupby("customer_id").agg(
        Recency=("order_date", lambda d: (reference_date - d.max()).days),
        Frequency=("transaction_id", "count"),
        Monetary=("net_revenue", "sum"),
        Profit=("gross_profit", "sum")
    ).reset_index()

    # 2. Vectorized Quantile Scoring (1 to 5)
    # Recency: lower is better -> inverse labels
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    # 3. Customer Segment Classification
    def map_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New / Promising"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m >= 3:
            return "High-Value Sleepers"
        else:
            return "Hibernating / Lost"

    rfm["Segment"] = rfm.apply(map_segment, axis=1)

    # 4. Segment Summary KPIs
    total_rev = rfm["Monetary"].sum()
    segment_summary = rfm.groupby("Segment").agg(
        Customer_Count=("customer_id", "count"),
        Total_Revenue=("Monetary", "sum"),
        Total_Profit=("Profit", "sum"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    ).reset_index()

    segment_summary["Revenue_Share_Pct"] = (segment_summary["Total_Revenue"] / total_rev * 100).round(2)
    segment_summary["Avg_Recency_Days"] = segment_summary["Avg_Recency_Days"].round(1)
    segment_summary["Avg_Frequency"] = segment_summary["Avg_Frequency"].round(1)
    segment_summary["Avg_Monetary"] = segment_summary["Avg_Monetary"].round(2)
    segment_summary = segment_summary.sort_values("Total_Revenue", ascending=False)

    return rfm, segment_summary
