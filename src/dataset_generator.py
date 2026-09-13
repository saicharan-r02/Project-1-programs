import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "retail_transactions.csv")

def generate_retail_dataset(n_records: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a rich, realistic global e-commerce & retail supply chain dataset.
    Features: transaction_id, customer_id, order_date, region, product_category,
              product_name, unit_cost, unit_price, quantity, discount_rate,
              payment_method, shipping_cost, delivery_days, return_status.
    """
    np.random.seed(random_state)
    os.makedirs(DATA_DIR, exist_ok=True)

    n_customers = 2500
    customer_pool = [f"CUST-{1000 + i}" for i in range(n_customers)]
    customer_ids = np.random.choice(customer_pool, size=n_records, p=np.random.dirichlet(np.ones(n_customers) * 0.4))

    # Dates spanning 24 months
    start_date = datetime(2024, 1, 1)
    date_offsets = np.random.randint(0, 730, size=n_records)
    order_dates = [start_date + timedelta(days=int(d)) for d in date_offsets]

    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    region_probs = [0.40, 0.28, 0.18, 0.08, 0.06]
    selected_regions = np.random.choice(regions, size=n_records, p=region_probs)

    catalog = [
        {"category": "Electronics", "name": "UltraHD 4K Monitor", "cost": 180.0, "base_price": 320.0},
        {"category": "Electronics", "name": "Wireless Noise-Cancelling Headphones", "cost": 65.0, "base_price": 149.0},
        {"category": "Electronics", "name": "Mechanical Gaming Keyboard", "cost": 38.0, "base_price": 89.0},
        {"category": "Apparel", "name": "Merino Wool Thermal Jacket", "cost": 42.0, "base_price": 115.0},
        {"category": "Apparel", "name": "Performance Running Shoes", "cost": 35.0, "base_price": 95.0},
        {"category": "Apparel", "name": "Organic Cotton Hoodie", "cost": 18.0, "base_price": 55.0},
        {"category": "Home & Kitchen", "name": "Espresso Coffee Machine", "cost": 110.0, "base_price": 240.0},
        {"category": "Home & Kitchen", "name": "Smart Air Purifier HEPA", "cost": 75.0, "base_price": 170.0},
        {"category": "Home & Kitchen", "name": "Cast Iron Dutch Oven 6Qt", "cost": 30.0, "base_price": 78.0},
        {"category": "Health & Beauty", "name": "Advanced Anti-Aging Peptide Serum", "cost": 14.0, "base_price": 48.0},
        {"category": "Health & Beauty", "name": "Sonic Electric Toothbrush", "cost": 22.0, "base_price": 65.0},
        {"category": "Sports & Outdoors", "name": "Ultralight Camping Tent 2-Person", "cost": 60.0, "base_price": 145.0},
        {"category": "Sports & Outdoors", "name": "Adjustable Dumbbell Set 50lbs", "cost": 95.0, "base_price": 210.0}
    ]

    item_indices = np.random.randint(0, len(catalog), size=n_records)
    categories = [catalog[i]["category"] for i in item_indices]
    product_names = [catalog[i]["name"] for i in item_indices]
    unit_costs = np.array([catalog[i]["cost"] for i in item_indices])
    unit_prices = np.array([catalog[i]["base_price"] for i in item_indices])

    # Quantities: Poisson distribution skewed towards 1-3 items
    quantities = np.random.poisson(1.6, size=n_records).clip(1, 10)

    # Discounts: 0%, 5%, 10%, 15%, 20%, 25%
    discount_rates = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], size=n_records, p=[0.45, 0.20, 0.15, 0.10, 0.06, 0.04])

    gross_revenue = unit_prices * quantities
    discount_amount = gross_revenue * discount_rates
    net_revenue = gross_revenue - discount_amount
    total_cogs = unit_costs * quantities
    gross_profit = net_revenue - total_cogs
    profit_margin = (gross_profit / net_revenue).round(4)

    payment_methods = np.random.choice(["Credit Card", "Digital Wallet", "Debit Card", "Buy Now Pay Later", "Bank Transfer"],
                                       size=n_records, p=[0.48, 0.26, 0.14, 0.08, 0.04])

    shipping_costs = (np.random.normal(12, 4, size=n_records) + quantities * 1.5).clip(4, 45).round(2)
    delivery_days = np.random.choice([1, 2, 3, 4, 5, 6, 7, 10], size=n_records, p=[0.12, 0.32, 0.28, 0.14, 0.08, 0.03, 0.02, 0.01])
    is_returned = np.random.binomial(1, 0.048, size=n_records)

    df = pd.DataFrame({
        "transaction_id": [f"TXN-{200000 + i}" for i in range(n_records)],
        "customer_id": customer_ids,
        "order_date": order_dates,
        "region": selected_regions,
        "product_category": categories,
        "product_name": product_names,
        "unit_cost": unit_costs,
        "unit_price": unit_prices,
        "quantity": quantities,
        "discount_rate": discount_rates,
        "gross_revenue": gross_revenue.round(2),
        "discount_amount": discount_amount.round(2),
        "net_revenue": net_revenue.round(2),
        "cogs": total_cogs.round(2),
        "gross_profit": gross_profit.round(2),
        "profit_margin": profit_margin,
        "payment_method": payment_methods,
        "shipping_cost": shipping_costs,
        "delivery_days": delivery_days,
        "is_returned": is_returned
    })

    df = df.sort_values("order_date").reset_index(drop=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"[OK] Retail transaction dataset generated: {DATA_FILE} ({len(df):,} transactions)")
    return df

def load_retail_data() -> pd.DataFrame:
    """Loads transaction dataset with parsed datetime."""
    if not os.path.exists(DATA_FILE):
        df = generate_retail_dataset()
    else:
        df = pd.read_csv(DATA_FILE)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df

if __name__ == "__main__":
    generate_retail_dataset()
