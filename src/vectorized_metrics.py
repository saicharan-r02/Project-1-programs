import numpy as np
import pandas as pd

def compute_clv_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Customer Lifetime Value metrics using vectorized NumPy operations.
    Calculates total spend, order count, lifespan days, and projected CLV.
    """
    valid_df = df[df["is_returned"] == 0]
    
    cust_summary = valid_df.groupby("customer_id").agg(
        total_revenue=("net_revenue", "sum"),
        total_profit=("gross_profit", "sum"),
        order_count=("transaction_id", "count"),
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max")
    ).reset_index()

    revenues = cust_summary["total_revenue"].to_numpy()
    profits = cust_summary["total_profit"].to_numpy()
    orders = cust_summary["order_count"].to_numpy()
    
    lifespan_days = (cust_summary["last_purchase"] - cust_summary["first_purchase"]).dt.days.to_numpy()
    lifespan_months = np.maximum(lifespan_days / 30.44, 1.0)

    aov = revenues / orders
    monthly_frequency = orders / lifespan_months
    profit_margin_pct = profits / np.maximum(revenues, 1.0)

    projected_annual_clv = aov * (monthly_frequency * 12) * profit_margin_pct

    cust_summary["aov"] = np.round(aov, 2)
    cust_summary["monthly_frequency"] = np.round(monthly_frequency, 2)
    cust_summary["profit_margin_pct"] = np.round(profit_margin_pct * 100, 2)
    cust_summary["projected_annual_clv"] = np.round(projected_annual_clv, 2)

    return cust_summary

def compute_product_affinity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Product Co-occurrence Basket Affinity Matrix using vectorized matrix dot product (M^T @ M).
    """
    df["basket_key"] = df["customer_id"] + "_" + df["order_date"].dt.strftime("%Y-%m-%d")
    basket_dummies = pd.crosstab(df["basket_key"], df["product_category"])
    
    binary_matrix = (basket_dummies.to_numpy() > 0).astype(int)
    
    co_occurrence = binary_matrix.T @ binary_matrix
    categories = basket_dummies.columns.tolist()

    affinity_df = pd.DataFrame(co_occurrence, index=categories, columns=categories)
    return affinity_df

def compute_price_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Price Elasticity of Demand across product categories using NumPy regression.
    Elasticity = (% Delta Quantity) / (% Delta Price)
    """
    results = []
    for cat, group in df.groupby("product_category"):
        prices = group["unit_price"].to_numpy() * (1 - group["discount_rate"].to_numpy())
        quantities = group["quantity"].to_numpy()

        log_p = np.log(prices)
        log_q = np.log(quantities)

        p_mean = np.mean(log_p)
        q_mean = np.mean(log_q)
        numerator = np.sum((log_p - p_mean) * (log_q - q_mean))
        denominator = np.sum((log_p - p_mean) ** 2)

        elasticity = numerator / denominator if denominator != 0 else -1.0
        
        if elasticity < -1.2:
            behavior = "Highly Elastic (Sensitive to Discounts)"
        elif elasticity > -0.8:
            behavior = "Inelastic (Brand Loyal / Essential)"
        else:
            behavior = "Unitary Elastic"

        results.append({
            "Category": cat,
            "Price_Elasticity": round(float(elasticity), 3),
            "Demand_Sensitivity": behavior
        })

    return pd.DataFrame(results).sort_values("Price_Elasticity")
