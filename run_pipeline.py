import os
import pandas as pd
from src.dataset_generator import load_retail_data
from src.vectorized_metrics import compute_clv_vectorized, compute_product_affinity_matrix, compute_price_elasticity
from src.rfm_segmentation import compute_rfm_segments
from src.cohort_analysis import compute_monthly_cohorts
from src.supply_chain import compute_supply_chain_kpis
from src.dashboard_generator import generate_visual_charts, generate_executive_html_dashboard

def main():
    print("\n" + "="*70)
    print(" [DataVoyager] Global E-Commerce & Retail Supply Chain Analytics")
    print("="*70)

    # 1. Load Transaction Dataset
    print("\n[Step 1/6] Ingesting multi-year retail transactions...")
    df = load_retail_data()
    print(f"[OK] Ingested {len(df):,} transactions across {df['customer_id'].nunique():,} unique customers.")

    # 2. Vectorized CLV & Price Elasticity
    print("\n[Step 2/6] Executing NumPy vectorized CLV and demand elasticity...")
    clv_df = compute_clv_vectorized(df)
    elasticity_df = compute_price_elasticity(df)
    print(f"[OK] Computed CLV for {len(clv_df):,} customers. Top 10% Avg Annual CLV: ${clv_df['projected_annual_clv'].quantile(0.90):,.2f}")
    print("\n--- Price Elasticity by Category ---")
    for _, row in elasticity_df.iterrows():
        print(f"  {row['Category']:<20}: {row['Price_Elasticity']:<6} ({row['Demand_Sensitivity']})")

    # 3. RFM Segmentation
    print("\n[Step 3/6] Computing Customer RFM segmentation...")
    rfm_df, rfm_summary = compute_rfm_segments(df)
    print("\n--- RFM Segment Breakdown ---")
    for _, row in rfm_summary.iterrows():
        print(f"  {row['Segment']:<22}: {row['Customer_Count']:>5} customers ({row['Revenue_Share_Pct']:>5.1f}% rev) | Avg Spend: ${row['Avg_Monetary']:>8.2f}")

    # 4. Cohort Retention Matrix
    print("\n[Step 4/6] Computing Monthly Retention Cohort matrices...")
    cohort_counts, retention_matrix = compute_monthly_cohorts(df)
    print(f"[OK] Generated {retention_matrix.shape[0]}x{retention_matrix.shape[1]} cohort retention matrix.")

    # 5. Supply Chain & Safety Stock Optimization
    print("\n[Step 5/6] Calculating Inventory Turnover, GMROI and Safety Stock...")
    sc_kpis = compute_supply_chain_kpis(df)
    print(f"[OK] Optimized inventory and Reorder Points (ROP) across {len(sc_kpis)} product lines.")

    # 6. Visual Charts & Executive Dashboard
    print("\n[Step 6/6] Generating visual charts and interactive executive dashboard...")
    affinity_matrix = compute_product_affinity_matrix(df)
    generate_visual_charts(df, rfm_summary, retention_matrix, affinity_matrix)

    total_rev = float(df[df["is_returned"] == 0]["net_revenue"].sum())
    total_prof = float(df[df["is_returned"] == 0]["gross_profit"].sum())
    kpis = {
        "total_revenue": total_rev,
        "total_profit": total_prof,
        "profit_margin": (total_prof / total_rev * 100) if total_rev > 0 else 0,
        "total_customers": int(df["customer_id"].nunique()),
        "aov": float(df[df["is_returned"] == 0]["net_revenue"].mean()),
        "return_rate": float(df["is_returned"].mean() * 100)
    }
    dashboard_path = generate_executive_html_dashboard(kpis, rfm_summary, sc_kpis)

    print("\n" + "="*70)
    print(" [OK] DataVoyager Analytics Pipeline Executed Successfully!")
    print(f" [INFO] Open the executive dashboard at: {dashboard_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
