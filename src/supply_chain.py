import numpy as np
import pandas as pd

def compute_supply_chain_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Supply Chain, Inventory Turnover, and Safety Stock Optimization metrics.
    """
    results = []

    for (cat, name), group in df.groupby(["product_category", "product_name"]):
        total_cogs = group["cogs"].sum()
        total_profit = group["gross_profit"].sum()
        total_units_sold = group["quantity"].sum()
        avg_lead_time_days = group["delivery_days"].mean()
        lead_time_std = group["delivery_days"].std() if len(group) > 1 else 1.0

        # Daily sales velocity
        total_days = (group["order_date"].max() - group["order_date"].min()).days
        total_days = max(total_days, 30)
        daily_demand = total_units_sold / total_days
        demand_std = group.groupby(group["order_date"].dt.date)["quantity"].sum().std()
        demand_std = 1.0 if np.isnan(demand_std) else demand_std

        # Simulated Average Holding Inventory (25 days buffer)
        avg_inventory_units = max(daily_demand * 25, 10.0)
        avg_inventory_cost = avg_inventory_units * group["unit_cost"].iloc[0]

        # 1. Inventory Turnover Ratio (ITR) = COGS / Avg Inventory Value
        itr = total_cogs / avg_inventory_cost if avg_inventory_cost > 0 else 0
        # 2. Days Sales of Inventory (DSI) = 365 / ITR
        dsi = (365.0 / itr) if itr > 0 else 0
        # 3. GMROI = Gross Profit / Avg Inventory Cost
        gmroi = (total_profit / avg_inventory_cost) if avg_inventory_cost > 0 else 0
        
        # 4. Safety Stock (95% Service Level -> Z = 1.65)
        # SS = Z * sqrt(LeadTime * Var(Demand) + Demand^2 * Var(LeadTime))
        variance_term = (avg_lead_time_days * (demand_std ** 2)) + ((daily_demand ** 2) * (lead_time_std ** 2))
        safety_stock = 1.65 * np.sqrt(max(variance_term, 0.1))

        # 5. Reorder Point (ROP) = (Daily Demand * Avg Lead Time) + Safety Stock
        reorder_point = (daily_demand * avg_lead_time_days) + safety_stock

        results.append({
            "Category": cat,
            "Product_Name": name,
            "Total_Units_Sold": int(total_units_sold),
            "COGS": round(float(total_cogs), 2),
            "Gross_Profit": round(float(total_profit), 2),
            "Inventory_Turnover_Ratio": round(float(itr), 2),
            "Days_Sales_Inventory": round(float(dsi), 1),
            "GMROI": round(float(gmroi), 2),
            "Safety_Stock_Units": int(np.ceil(safety_stock)),
            "Reorder_Point_Units": int(np.ceil(reorder_point))
        })

    return pd.DataFrame(results).sort_values("GMROI", ascending=False)
